# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "bbos",
#   "numpy",
#   "opencv-python",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Keep track of where people are and turn in place to face one on request.

Runs ON THE ROBOT as a long-lived helper of the voice assistant, speaking
JSON lines over stdin/stdout:

    {"id": 1, "cmd": "acquire", "purpose": "scan", "hint_deg": null}
    {"cmd": "cancel"}
    {"id": 2, "cmd": "status"}
    {"id": 3, "cmd": "turn", "delta_deg": 12.0}

While idle it looks for faces on the head camera about twice a second and
remembers the heading of the last person it saw. That memory is the first
place it turns when asked to find someone.

Motion is deliberately narrow: it only turns in place (zero linear speed),
slowly, and only while an ``acquire`` or ``turn`` request is running. ``drive.ctrl`` is
opened for that request alone, so nav and teleop keep the base otherwise. The
drive daemon zeroes the base 0.1 s after the last command, so a crash or kill
stops the turn. It refuses to move when the robot is not upright, is in lean
or twist mode, or another app is already driving. It never drives toward a
person; it reports the distance so the assistant can ask them to move.

Stdin closing (the assistant exiting) ends the process.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from camera_geometry import (  # noqa: E402
    DEFAULT_D,
    DEFAULT_K,
    DEFAULT_T_POINTS_CAMERA,
    camera_ray_to_points,
    fisheye_pixel_to_ray,
)

CAMERA_TOPIC = "camera.head.rgb"    # raw 2560x960 stereo, left eye is the first half
FACE_MODEL = "face_detection_yunet_2026may.onnx"
FACE_SCORE = 0.75
FACE_WIDTH_M = 0.16               # typical adult face width for a rough range
IDLE_PERIOD_S = 0.5
MEMORY_S = 120.0

MAX_REQUESTED_TURN_DEG = 45.0     # "turn" lines a gesture up; it is not a way to spin
MAX_TURN_RAD_S = 0.6              # drive limit is 1.0 rad/s; stay well under
MIN_TURN_RAD_S = 0.2
TURN_KP = 0.02                    # rad/s per degree of heading error
TURN_ACCEL = 1.5                  # rad/s^2 slew limit, gentle on a balancing base
TURN_TOLERANCE_DEG = 4.0
CENTER_TOLERANCE_DEG = 6.0
SEARCH_STEP_DEG = 60.0            # the fisheye sees well past +/-50 deg, so steps overlap
SETTLE_S = 0.6
DRIVE_PERIOD_S = 0.02
UPRIGHT_DEG = 25.0
BALANCE_MODE = 0

# Face distance the next action works best at, in metres.
DISTANCE_BANDS = {"scan": (0.4, 1.2), "gesture": (0.45, 0.9)}


class Cancelled(Exception):
    pass


class Refused(Exception):
    """The robot is not in a state where turning is allowed."""


# ---------------------------------------------------------------------------
# Pure helpers (tested without a robot)
# ---------------------------------------------------------------------------

def wrap_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def pixel_bearing_deg(u, v, K=DEFAULT_K, D=DEFAULT_D, T=DEFAULT_T_POINTS_CAMERA) -> float:
    """Horizontal bearing of a raw left-eye pixel. Positive is to the robot's left."""
    _, direction = camera_ray_to_points(fisheye_pixel_to_ray(u, v, K, D), T)
    lateral_right, forward = direction[0], direction[1]
    return math.degrees(math.atan2(-lateral_right, forward))


def estimate_distance_m(face_width_px: float, fx: float = float(DEFAULT_K[0, 0])) -> float:
    return FACE_WIDTH_M * fx / max(1.0, float(face_width_px))


def distance_band(distance_m: float, purpose: str) -> str:
    near, far = DISTANCE_BANDS.get(purpose, DISTANCE_BANDS["scan"])
    if distance_m > far:
        return "far"
    if distance_m < near:
        return "close"
    return "ok"


def pick_face(faces):
    """The largest confident face, which is usually the nearest person."""
    faces = [face for face in faces if face[4] >= FACE_SCORE]
    return max(faces, key=lambda face: face[2]) if faces else None


def search_plan(hint_deg=None, step=SEARCH_STEP_DEG, max_total=360.0):
    """Relative turns for one look-around, trying the hinted direction first.

    The last step that would bring the robot back to its starting view is
    left out because that view has already been checked.
    """
    turns = []
    direction = -1.0 if hint_deg is not None and hint_deg < 0 else 1.0
    total = 0.0
    if hint_deg is not None and abs(hint_deg) >= step / 2:
        turns.append(float(hint_deg))
        total = abs(hint_deg)
    while total + step < max_total - step / 2:
        turns.append(direction * step)
        total += step
    return turns


def turn_command(error_deg: float) -> float:
    """Angular speed (rad/s, positive = left) for the remaining heading error."""
    if abs(error_deg) <= TURN_TOLERANCE_DEG:
        return 0.0
    speed = min(MAX_TURN_RAD_S, max(MIN_TURN_RAD_S, TURN_KP * abs(error_deg)))
    return math.copysign(speed, error_deg)


def slew(previous: float, target: float, dt: float, accel: float = TURN_ACCEL) -> float:
    step = accel * dt
    return previous + max(-step, min(step, target - previous))


# ---------------------------------------------------------------------------
# Robot side
# ---------------------------------------------------------------------------

def _log(message):
    print(f"[person-tracker] {message}", file=sys.stderr, flush=True)


class Robot:
    def __init__(self, bbos, model_path: Path):
        import cv2

        self.cv2 = cv2
        self.bbos = bbos
        self.K, self.D, self.T = self._camera_model()
        self.camera = bbos.Reader(CAMERA_TOPIC, keeptime=False).__enter__()
        self.imu = bbos.Reader("imu.orientation", keeptime=False).__enter__()
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), FACE_SCORE, 0.3, 50)
        self.detector_size = None
        self.yaw_sign = None          # IMU yaw direction vs. positive twist, learned on the first turn
        self.last_yaw = None

    def _camera_model(self):
        K, D, T = DEFAULT_K, DEFAULT_D, DEFAULT_T_POINTS_CAMERA
        try:
            calibration = self.bbos.Config("depth").camera_cal()
            K = np.asarray(calibration[0], dtype=np.float64)
            D = np.asarray(calibration[1], dtype=np.float64).reshape(-1)[:4]
        except Exception as exc:  # noqa: BLE001 - documented values are a safe fallback
            _log(f"camera intrinsics fallback ({exc})")
        try:
            T = np.asarray(self.bbos.Config("depth").camera_to_base_3x4, dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass
        return K, D, T

    def close(self):
        for reader in (self.camera, self.imu):
            try:
                reader.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    # -- sensing ------------------------------------------------------------

    def rpy(self, timeout=1.0):
        deadline = time.monotonic() + timeout
        while not self.imu.ready():
            if time.monotonic() > deadline:
                if self.last_yaw is None:
                    raise Refused("no IMU heading is available")
                break
            time.sleep(0.002)
        else:
            rpy = np.asarray(self.imu.data["rpy"], dtype=np.float64).reshape(-1)
            self.last_rpy = rpy
            self.last_yaw = float(rpy[2])
        return self.last_rpy

    def yaw(self):
        return float(self.rpy()[2])

    def frame_after(self, t, timeout=1.0):
        """Newest left-eye BGR frame captured after monotonic time ``t``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.camera.ready() and time.monotonic() >= t:
                stereo = self.camera.data["rgb"]
                return self.cv2.cvtColor(stereo[:, : stereo.shape[1] // 2], self.cv2.COLOR_RGB2BGR)
            time.sleep(0.003)
        return None

    def detect(self, frame):
        h, w = frame.shape[:2]
        if self.detector_size != (w, h):
            self.detector.setInputSize((w, h))
            self.detector_size = (w, h)
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])) for f in faces]

    def look(self, frames=2):
        """Best face over a few fresh frames, as a dict with bearing and range."""
        best = None
        start = time.monotonic()
        for _ in range(frames):
            frame = self.frame_after(start)
            if frame is None:
                break
            face = pick_face(self.detect(frame))
            if face is not None and (best is None or face[2] > best[2]):
                best = face
            start = time.monotonic()
        if best is None:
            return None
        x, y, w, h, score = best
        return {
            "bearing_deg": round(pixel_bearing_deg(x + w / 2, y + h / 2, self.K, self.D, self.T), 1),
            "face_width_px": round(w),
            "distance_m": round(estimate_distance_m(w, float(self.K[0, 0])), 2),
            "score": round(score, 2),
        }

    # -- motion -------------------------------------------------------------

    def preflight(self):
        roll, pitch = self.rpy()[:2]
        if abs(roll) >= UPRIGHT_DEG or abs(pitch) >= UPRIGHT_DEG:
            raise Refused("the robot is not upright")
        with self.bbos.Reader("base.mode", keeptime=False) as mode:
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                if mode.ready():
                    if int(mode.data["mode"]) != BALANCE_MODE:
                        raise Refused("the base is in lean or twist mode")
                    break
                time.sleep(0.01)
        with self.bbos.Reader("drive.ctrl", keeptime=False) as drive:
            deadline = time.monotonic() + 0.3
            while time.monotonic() < deadline:
                if drive.ready():
                    raise Refused("another app is already driving the base")
                time.sleep(0.01)

    def turn_by(self, writer, delta_deg, cancel):
        """Closed-loop turn in place on IMU yaw. Always leaves the base at zero twist."""
        start_yaw = self.yaw()
        started = last = time.monotonic()
        timeout = abs(delta_deg) / math.degrees(MIN_TURN_RAD_S) + 3.0
        speed = 0.0
        try:
            while True:
                if cancel.is_set():
                    raise Cancelled()
                now = time.monotonic()
                raw = wrap_deg(self.yaw() - start_yaw)
                if self.yaw_sign is None and now - started > 0.6 and abs(speed) >= MIN_TURN_RAD_S:
                    if abs(raw) < 1.5:
                        raise Refused("the base did not turn when commanded")
                    self.yaw_sign = 1.0 if raw * delta_deg > 0 else -1.0
                    _log(f"IMU yaw sign learned: {self.yaw_sign:+.0f}")
                done = raw * (self.yaw_sign or 1.0)
                error = delta_deg - done
                target = turn_command(error)
                if target == 0.0:
                    break
                if now - started > timeout:
                    raise Refused("the turn did not finish in time")
                speed = slew(speed, target, now - last)
                last = now
                writer["twist"] = np.array([0.0, speed], dtype=np.float32)
                time.sleep(DRIVE_PERIOD_S)
        finally:
            for _ in range(10):
                writer["twist"] = np.zeros(2, dtype=np.float32)
                time.sleep(DRIVE_PERIOD_S)
        if cancel.wait(SETTLE_S):
            raise Cancelled()


class Tracker:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.last_seen = None         # {"yaw": world deg, "time": monotonic, ...}

    def remember(self, face):
        yaw = self.robot.yaw()
        self.last_seen = {"yaw": wrap_deg(yaw + face["bearing_deg"] * (self.robot.yaw_sign or 1.0)),
                          "time": time.monotonic(), **face}

    def observe(self):
        face = self.robot.look(frames=1)
        if face is not None:
            self.remember(face)
        return face

    def memory_hint(self):
        if self.last_seen is None or time.monotonic() - self.last_seen["time"] > MEMORY_S:
            return None
        hint = wrap_deg(self.last_seen["yaw"] - self.robot.yaw()) * (self.robot.yaw_sign or 1.0)
        return hint if abs(hint) > CENTER_TOLERANCE_DEG else None

    def status(self):
        seen = None
        if self.last_seen is not None:
            seen = {key: value for key, value in self.last_seen.items() if key not in ("yaw", "time")}
            seen["age_s"] = round(time.monotonic() - self.last_seen["time"], 1)
        return {"ok": True, "last_seen": seen}

    def acquire(self, purpose, hint_deg, cancel):
        turned = 0.0
        face = self.robot.look()
        writer = None
        try:
            if face is None or abs(face["bearing_deg"]) > CENTER_TOLERANCE_DEG:
                try:
                    self.robot.preflight()
                    try:
                        writer = self.robot.bbos.Writer(
                            "drive.ctrl", self.robot.bbos.Type("drive_ctrl"), keeptime=False
                        ).__enter__()
                    except RuntimeError as exc:
                        # An idle teleop or nav process holds the writer without
                        # publishing, so preflight cannot see it. Same answer.
                        raise Refused("another app is already driving the base") from exc
                except Refused as exc:
                    if face is None:
                        raise
                    # Someone is already in view; use them without turning.
                    _log(f"not centering: {exc}")
                    self.remember(face)
                    return {"found": True, "centered": False, "turned_deg": 0,
                            "distance": distance_band(face["distance_m"], purpose), **face}
            if face is None:
                hint = hint_deg if hint_deg is not None else self.memory_hint()
                source = "voice" if hint_deg is not None else ("memory" if hint is not None else "sweep")
                _log(f"no face in view; searching ({source}, hint={hint})")
                for step in search_plan(hint):
                    self.robot.turn_by(writer, step, cancel)
                    turned += step
                    face = self.robot.look()
                    if face is not None:
                        break
            if face is None:
                return {"found": False, "reason": "nobody in view after looking around",
                        "turned_deg": round(turned)}
            for _ in range(3):
                if abs(face["bearing_deg"]) <= CENTER_TOLERANCE_DEG:
                    break
                self.robot.turn_by(writer, face["bearing_deg"], cancel)
                turned += face["bearing_deg"]
                face = self.robot.look() or face
            self.remember(face)
            return {
                "found": True,
                "centered": abs(face["bearing_deg"]) <= CENTER_TOLERANCE_DEG,
                "distance": distance_band(face["distance_m"], purpose),
                "turned_deg": round(turned),
                **face,
            }
        finally:
            if writer is not None:
                writer.__exit__(None, None, None)


    def turn(self, delta_deg, cancel):
        """Turn in place by a caller-chosen angle (positive left), with the usual refusals."""
        delta = float(delta_deg)
        if not math.isfinite(delta) or abs(delta) > MAX_REQUESTED_TURN_DEG:
            raise Refused(f"a {delta:.0f} degree turn is more than a gesture may ask for")
        self.robot.preflight()
        try:
            writer = self.robot.bbos.Writer(
                "drive.ctrl", self.robot.bbos.Type("drive_ctrl"), keeptime=False
            ).__enter__()
        except RuntimeError as exc:
            raise Refused("another app is already driving the base") from exc
        try:
            self.robot.turn_by(writer, delta, cancel)
        finally:
            writer.__exit__(None, None, None)
        return {"ok": True, "turned_deg": round(delta, 1)}


def _read_commands(commands: queue.Queue, cancel: threading.Event):
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if request.get("cmd") == "cancel":
            cancel.set()
        else:
            commands.put(request)
    commands.put(None)                # stdin closed: the assistant is gone
    cancel.set()


def main():
    from bbos import Config, Reader, Type, Writer
    from types import SimpleNamespace

    if "--check-deps" in sys.argv:    # used by the launcher to install the env
        import cv2  # noqa: F401
        return

    model = next((p for p in (HERE / FACE_MODEL, HERE.parent / "assets" / "models" / FACE_MODEL) if p.exists()), None)
    if model is None:
        sys.exit(f"{FACE_MODEL} not found next to this script")
    bbos = SimpleNamespace(Config=Config, Reader=Reader, Type=Type, Writer=Writer)
    robot = Robot(bbos, model)
    tracker = Tracker(robot)
    commands: queue.Queue = queue.Queue()
    cancel = threading.Event()
    threading.Thread(target=_read_commands, args=(commands, cancel), daemon=True).start()

    def reply(request, body):
        print(json.dumps({"id": request.get("id"), **body}), flush=True)

    print(json.dumps({"ready": True}), flush=True)
    _log("ready")
    try:
        while True:
            try:
                request = commands.get(timeout=IDLE_PERIOD_S)
            except queue.Empty:
                try:
                    tracker.observe()
                except Exception as exc:  # noqa: BLE001 - idle observation is best effort
                    _log(f"observe failed: {exc}")
                continue
            if request is None:
                return
            if request.get("cmd") == "status":
                reply(request, tracker.status())
            elif request.get("cmd") == "turn":
                cancel.clear()
                try:
                    result = tracker.turn(request.get("delta_deg", 0.0), cancel)
                except Cancelled:
                    result = {"ok": False, "reason": "cancelled"}
                except Refused as exc:
                    result = {"ok": False, "refused": True, "reason": str(exc)}
                except Exception as exc:  # noqa: BLE001 - report, never crash the helper
                    _log(f"turn failed: {type(exc).__name__}: {exc}")
                    result = {"ok": False, "error": True, "reason": str(exc)}
                _log(f"turn -> {result}")
                reply(request, result)
            elif request.get("cmd") == "acquire":
                cancel.clear()
                try:
                    result = tracker.acquire(request.get("purpose", "scan"), request.get("hint_deg"), cancel)
                except Cancelled:
                    result = {"found": False, "reason": "cancelled"}
                except Refused as exc:
                    result = {"found": False, "refused": True, "reason": str(exc)}
                except Exception as exc:  # noqa: BLE001 - report, never crash the helper
                    _log(f"acquire failed: {type(exc).__name__}: {exc}")
                    result = {"found": False, "error": True, "reason": str(exc)}
                _log(f"acquire -> {result}")
                reply(request, result)
            else:
                reply(request, {"ok": False, "reason": "unknown command"})
    finally:
        robot.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
