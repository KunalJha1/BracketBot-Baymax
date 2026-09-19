# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "bbos",
#   "bbai",
#   "google-genai",
#   "python-dotenv",
#   "numpy",
#   "soxr",
#   "Pillow",
#   "opencv-python",
#   "pycuda",
#   "fastapi",
#   "uvicorn",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# bbai = { path = "/home/bracketbot/bbai", editable = true }
# ///
"""
Greeter: person detection + Gemini Live audio + deterministic voice actions.

Detects people via YOLO26 TensorRT and plays the 'wave' arm movement when a
face is detected. Human speech can use Gemini Live or fully local Whisper and
eSpeak I/O. It is sent through the local action allowlist; non-action
conversation can use OpenRouter and Browserbase.

Usage:
  uv run main.py
  uv run main.py --voice Puck --volume 0.5
"""

import argparse
import asyncio
import cv2
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import soxr
from dotenv import load_dotenv
from PIL import Image

from google import genai
from google.genai import types

from bbos import Reader, Writer, Type, Config

try:
    from .gesture_safety import depth_clearance, spoken_safety_refusal
    from .gesture_runtime import (
        arm_motion_reserved,
        prepare_recorded_movement,
        play_recorded_movement,
        recorded_movement_name,
        set_arms_limp,
    )
    from .voice_router import (
        OpenRouterClient,
        RouteKind,
        VoiceRouter,
        default_question_response_cache,
        default_seed_pairs,
        utterances_match,
    )
    from .pointing import (
        PersonTargetTracker,
        pointing_goal,
        quaternion_from_z,
        quaternion_slerp,
    )
except ImportError:  # ``uv run main.py`` executes this as a standalone script.
    from gesture_safety import depth_clearance, spoken_safety_refusal
    from gesture_runtime import (
        arm_motion_reserved,
        prepare_recorded_movement,
        play_recorded_movement,
        recorded_movement_name,
        set_arms_limp,
    )
    from voice_router import (
        OpenRouterClient,
        RouteKind,
        VoiceRouter,
        default_question_response_cache,
        default_seed_pairs,
        utterances_match,
    )
    from pointing import PersonTargetTracker, pointing_goal, quaternion_from_z, quaternion_slerp

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
MOVEMENTS_DIR = SCRIPT_DIR / "movements"
MOVEMENTS_DIR.mkdir(exist_ok=True)

# ── Audio config ──────────────────────────────────────────────────────
CFG = Config("speaker")
CHUNK = CFG.chunk_size
BBOS_RATE = Config("mic").sample_rate
GEMINI_RATE = 24000                   # Gemini outputs 24kHz PCM16
JITTER_BUFFER_CHUNKS = 4


def resample_24k_to_16k(pcm_24k):
    return soxr.resample(pcm_24k.astype(np.float32), GEMINI_RATE, BBOS_RATE).astype(np.int16)


# ── Arm movement ─────────────────────────────────────────────────────
DOF = 8
_arm_shm = {}
_arm_lock = threading.Lock()
_movement_playback_lock = threading.Lock()
_movement_cancel_event = threading.Event()
_saved_movements = {}
_person_targets = PersonTargetTracker(max_age=1.0)

POINT_RAMP_SECONDS = 2.0
POINT_HOLD_SECONDS = 1.0
POINT_MAX_ENTRY_TURNS = 0.45
POINT_UPRIGHT_DEGREES = 25.0
POINT_IK_SAMPLES = 40
POINT_MAX_STEP_TURNS = 0.08


def _load_movements():
    """Load all saved movements from disk."""
    for f in MOVEMENTS_DIR.glob("*.json"):
        if f.name.startswith("."):
            continue
        try:
            data = json.loads(f.read_text())
            _saved_movements[f.stem] = data
            n = len(data)
            dur = data[-1]["t"] if n else 0
            print(f"[arm] Loaded '{f.stem}': {n} frames, {dur:.1f}s", flush=True)
        except Exception as e:
            print(f"[arm] Failed to load {f}: {e}", flush=True)
    dance_path = SCRIPT_DIR.parent / "mimic" / "recordings" / "dance.json"
    if dance_path.is_file():
        try:
            data = json.loads(dance_path.read_text())
            _saved_movements["dance"] = data
            print(f"[arm] Loaded 'dance': {len(data)} frames", flush=True)
        except Exception as exc:
            print(f"[arm] Failed to load {dance_path}: {exc}", flush=True)


def _get_arm_shm():
    """Lazy-init arm SHM readers/writers."""
    with _arm_lock:
        if not _arm_shm:
            r_l = Reader("arm_left.state", keeptime=False)
            r_r = Reader("arm_right.state", keeptime=False)
            w_l = Writer("arm_left.ctrl", Type("arm_ctrl"), keeptime=False)
            w_r = Writer("arm_right.ctrl", Type("arm_ctrl"), keeptime=False)
            w_tl = Writer("arm_left.torque", Type("arm_torque"), keeptime=False)
            w_tr = Writer("arm_right.torque", Type("arm_torque"), keeptime=False)
            r_l.__enter__(); r_r.__enter__()
            w_l.__enter__(); w_r.__enter__()
            w_tl.__enter__(); w_tr.__enter__()
            _arm_shm.update(r_left=r_l, r_right=r_r, w_left=w_l, w_right=w_r,
                            w_torque_left=w_tl, w_torque_right=w_tr)
            print("[arm] SHM initialized", flush=True)
    return _arm_shm


def _cleanup_arm_shm():
    with _arm_lock:
        for key in list(_arm_shm):
            try:
                _arm_shm[key].__exit__(None, None, None)
            except Exception:
                pass
        _arm_shm.clear()


def _prepare_movement(name):
    """Read fresh safety inputs before opening any arm control writer."""
    return prepare_recorded_movement(_saved_movements[name])


def _play_movement_plan(name, plan):
    """Play one preflighted plan, return to measured starts, then torque off."""
    play_recorded_movement(
        name, plan, _movement_cancel_event, stop_event
    )


def start_movement(name):
    """Safety-check and start at most one allowlisted recorded movement."""
    if arm_motion_reserved():
        return False, "My arms are busy packing, but I can still answer questions."
    movement_name = recorded_movement_name(name)
    if movement_name not in _saved_movements:
        return False, f"Movement '{movement_name}' is not installed"
    if not _saved_movements[movement_name]:
        return False, f"Movement '{movement_name}' has no frames"
    if not _movement_playback_lock.acquire(blocking=False):
        return False, "Another movement is already running"

    try:
        plan = _prepare_movement(movement_name)
    except Exception as exc:
        print(f"[arm] {name} preflight rejected: {exc}", flush=True)
        _movement_playback_lock.release()
        return False, spoken_safety_refusal(name, exc)
    _movement_cancel_event.clear()
    _debug["movement_running"] = True

    def run():
        try:
            _play_movement_plan(name, plan)
        except Exception as exc:
            print(f"[arm] Gesture failed safely: {exc}", flush=True)
        finally:
            if name == "goodbye":
                try:
                    set_arms_limp()
                except Exception as exc:
                    print(f"[arm] Could not make both arms limp: {exc}", flush=True)
            _debug["movement_running"] = False
            _movement_playback_lock.release()

    threading.Thread(target=run, name=f"voice-movement-{name}", daemon=True).start()
    return True, f"Started {name}"


def _fresh(reader, timeout=2.0):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started >= timeout:
            raise RuntimeError("No fresh robot state is available")
        time.sleep(0.005)
    return reader.data


def _smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _play_arm_path(writer, path, duration, cancel_event=None):
    """Play a collision-checked motor path and return progress when cancelled."""
    began = time.monotonic()
    last_pose = np.asarray(path[0], dtype=np.float64)
    last_position = 0.0
    while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
        alpha = min((time.monotonic() - began) / duration, 1.0)
        eased = _smoothstep(alpha)
        path_position = eased * (len(path) - 1)
        lower = min(int(path_position), len(path) - 1)
        upper = min(lower + 1, len(path) - 1)
        fraction = path_position - lower
        pose = (1.0 - fraction) * path[lower] + fraction * path[upper]
        writer["pos"] = pose.astype(np.float32)
        last_pose = pose
        last_position = path_position
        if alpha >= 1.0:
            break
        time.sleep(0.015)
    return last_pose, last_position


def _plan_point_path(cfg, start, position, quaternion):
    """Solve the entire Cartesian route, rejecting discontinuous IK branches."""
    current_urdf = np.asarray(cfg.q2urdf(start.copy()), dtype=np.float64)
    cfg.ik.reset(list(current_urdf[:7]))
    start_position, start_quaternion = cfg.ik.fk(list(current_urdf[:7]))
    start_position = np.asarray(start_position, dtype=np.float64)
    start_quaternion = np.asarray(start_quaternion, dtype=np.float64)
    path = [start.copy()]

    for index in range(1, POINT_IK_SAMPLES + 1):
        alpha = index / POINT_IK_SAMPLES
        waypoint = (1.0 - alpha) * start_position + alpha * position
        orientation = quaternion_slerp(start_quaternion, quaternion, alpha)
        solution = cfg.ik.solve(waypoint.tolist(), orientation.tolist())
        if solution is None or len(solution) < 7:
            raise RuntimeError(
                f"Collision-aware IK missed pointing waypoint {index}/{POINT_IK_SAMPLES}"
            )
        solved_urdf = current_urdf.copy()
        solved_urdf[:7] = np.asarray(solution[:7], dtype=np.float64)
        pose = np.asarray(cfg.urdf2q(solved_urdf), dtype=np.float64)
        pose[7:] = start[7:]
        if not np.isfinite(pose).all():
            raise RuntimeError("Pointing IK returned a non-finite arm pose")
        step = float(np.max(np.abs(pose - path[-1])))
        if step > POINT_MAX_STEP_TURNS:
            raise RuntimeError(
                f"Pointing path rejected at waypoint {index}: "
                f"{step:.3f}-turn IK branch jump"
            )
        path.append(pose)

    entry = float(np.max(np.abs(np.asarray(path) - start)))
    if entry > POINT_MAX_ENTRY_TURNS:
        raise RuntimeError(
            f"Pointing path rejected: largest joint move is {entry:.3f} turns"
        )
    return path, entry


def play_pointing_gesture(target):
    """Aim one arm at a camera-selected person, hold briefly, then return."""
    side = target.arm
    cfg = Config(f"arm_{side}")
    shm = _get_arm_shm()
    reader = shm[f"r_{side}"]
    control = shm[f"w_{side}"]
    torque = shm[f"w_torque_{side}"]
    torque_enabled = False

    with Reader("imu.orientation", keeptime=False) as imu:
        rpy = np.asarray(_fresh(imu)["rpy"], dtype=np.float64)
        if abs(rpy[0]) >= POINT_UPRIGHT_DEGREES or abs(rpy[1]) >= POINT_UPRIGHT_DEGREES:
            raise RuntimeError("Pointing cancelled because the robot is not upright")

        start = np.asarray(_fresh(reader)["pos"], dtype=np.float64).copy()
        cfg.ik.init()
        shoulder_height = float(Config("quest").robot_shoulder_height)
        position, direction = pointing_goal(target, shoulder_height)
        quaternion = quaternion_from_z(direction)
        path, entry = _plan_point_path(cfg, start, position, quaternion)

        print(
            f"[point] {side} arm -> image x={target.x_offset:+.2f}, "
            f"goal={np.round(position, 3)}, entry={entry:.3f} turns",
            flush=True,
        )
        try:
            # Seed the controller at the measured pose before enabling torque.
            control["pos"] = start.astype(np.float32)
            time.sleep(0.10)
            torque_enabled = True
            torque["enable"] = np.ones(DOF, dtype=np.bool_)
            time.sleep(0.70)
            last_pose, path_position = _play_arm_path(
                control,
                path,
                POINT_RAMP_SECONDS,
                cancel_event=_movement_cancel_event,
            )
            _movement_cancel_event.wait(POINT_HOLD_SECONDS)
            # Return only through the portion actually traversed. If Stop was
            # pressed halfway out, jumping to the full endpoint would be unsafe.
            reached_index = min(int(path_position), len(path) - 1)
            return_path = [last_pose, *reversed(path[: reached_index + 1])]
            _play_arm_path(control, return_path, POINT_RAMP_SECONDS)
        finally:
            if torque_enabled:
                # The arm is back at its measured start before becoming limp.
                control["pos"] = start.astype(np.float32)
                time.sleep(0.10)
                torque["enable"] = np.zeros(DOF, dtype=np.bool_)
        print("[point] gesture complete; arm returned and torque is off", flush=True)


def start_robot_action(name):
    """Dispatch either a recorded movement or the camera-guided point gesture."""
    if arm_motion_reserved():
        return False, "My arms are busy packing, but I can still answer questions."
    point_preferences = {
        "point": "primary",
        "point-left": "left",
        "point-right": "right",
    }
    if name not in point_preferences:
        return start_movement(name)

    target = _person_targets.select(point_preferences[name])
    if target is None:
        return False, "I cannot see a person clearly enough to point at right now."
    if not _movement_playback_lock.acquire(blocking=False):
        return False, "Another movement is already running"
    try:
        with (
            Reader("imu.orientation", keeptime=False) as imu,
            Reader("camera.points", keeptime=False) as depth,
        ):
            rpy = np.asarray(_fresh(imu)["rpy"], dtype=np.float64)
            if (
                abs(rpy[0]) >= POINT_UPRIGHT_DEGREES
                or abs(rpy[1]) >= POINT_UPRIGHT_DEGREES
            ):
                raise RuntimeError("the robot is not upright")
            depth_data = _fresh(depth)
            count = int(depth_data["num_points"])
            depth_clearance(depth_data["points"][:count], (target.arm,))
    except Exception as exc:
        print(f"[point] preflight rejected: {exc}", flush=True)
        _movement_playback_lock.release()
        return False, spoken_safety_refusal(name, exc)
    _movement_cancel_event.clear()
    _debug["movement_running"] = True

    def run():
        try:
            play_pointing_gesture(target)
        except Exception as exc:
            print(f"[point] Gesture failed safely: {exc}", flush=True)
        finally:
            _debug["movement_running"] = False
            _movement_playback_lock.release()

    threading.Thread(target=run, name=f"voice-{name}", daemon=True).start()
    return True, f"Started {name}"


def stop_robot_action():
    """Request a safe return for whichever voice movement is active."""
    if not _movement_playback_lock.locked():
        return False, "No movement is running."
    _movement_cancel_event.set()
    return True, "Okay. Stopping safely."


# ── Drive tool ────────────────────────────────────────────────────────
LINEAR_SPEED = 0.15   # m/s
TURN_SPEED = 1.0      # rad/s

DIRECTION_MAP = {
    "forward":  ( 1,  0),
    "backward": (-1,  0),
    "left":     ( 0,  1),
    "right":    ( 0, -1),
    "stop":     ( 0,  0),
}


def drive_robot(direction, speed, duration):
    """Send drive commands to the drive daemon via SHM."""
    lin_sign, ang_sign = DIRECTION_MAP.get(direction, (0, 0))
    twist = np.array([
        lin_sign * LINEAR_SPEED * speed,
        ang_sign * TURN_SPEED * speed,
    ], dtype=np.float32)

    print(f"[drive] START {direction} twist={twist} for {duration}s", flush=True)
    try:
        w_ctrl = _get_drive_writer()
        start = time.time()
        while time.time() - start < duration and not stop_event.is_set():
            w_ctrl["twist"] = twist
            time.sleep(0.05)
        w_ctrl["twist"] = np.zeros(2, dtype=np.float32)
    except Exception as e:
        print(f"[drive] Error: {e}", flush=True)
    print(f"[drive] DONE {direction}", flush=True)


# ── WASD drive ───────────────────────────────────────────────────────
WHEEL_VEL_COMBOS = {
    'w':  (0.20, 0.20),    'wa': (0.05, 0.28),
    's':  (-0.20, -0.20),  'wd': (0.28, 0.05),
    'a':  (-0.15, 0.15),   'sa': (-0.05, -0.28),
    'd':  (0.15, -0.15),   'sd': (-0.28, -0.05),
    '':   (0.0, 0.0),
}

_drive_writer = None
_drive_lock = threading.Lock()
_wasd_cmd = {'keys': '', 'shift': False, 'gain': 1.0}
_wasd_lock = threading.Lock()


def _get_drive_writer():
    global _drive_writer
    with _drive_lock:
        if _drive_writer is None:
            _drive_writer = Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False)
            _drive_writer.__enter__()
    return _drive_writer


def _cleanup_drive_writer():
    global _drive_writer
    with _drive_lock:
        if _drive_writer is not None:
            try:
                _drive_writer.__exit__(None, None, None)
            except Exception:
                pass
            _drive_writer = None


def _wheel_vels_to_twist(v_left, v_right):
    R = Config("drive").robot_width * 0.5
    v = (v_left + v_right) / 2.0
    w = (v_right - v_left) / (2.0 * R)
    return float(v), float(w)


def wasd_drive_loop():
    """Background thread: read WASD state, write twist to drive.ctrl at 20Hz."""
    print("[wasd] Drive loop started", flush=True)
    try:
        w_ctrl = _get_drive_writer()
    except Exception as e:
        print(f"[wasd] Failed to get drive writer: {e}", flush=True)
        return

    while not stop_event.is_set():
        with _wasd_lock:
            keys = _wasd_cmd['keys']
            shift = _wasd_cmd['shift']
            gain = _wasd_cmd['gain']

        if keys in WHEEL_VEL_COMBOS:
            v_left, v_right = WHEEL_VEL_COMBOS[keys]
        else:
            v_left, v_right = 0.0, 0.0

        speed_mult = 1.0 if shift else 0.5
        total_mult = speed_mult * gain
        v_left *= total_mult
        v_right *= total_mult
        v, w = _wheel_vels_to_twist(v_left, v_right)

        try:
            w_ctrl["twist"] = np.array([v, w], dtype=np.float32)
        except Exception:
            pass
        time.sleep(0.05)


# ── Shared state ──────────────────────────────────────────────────────
mic_queue = queue.Queue(maxsize=50)
speaker_queue = queue.Queue(maxsize=200)
stop_event = threading.Event()
interrupt_flag = threading.Event()

# Gemini session refs for cross-thread greeting
_gemini_session = None
_gemini_loop = None

# ── Debug state ──────────────────────────────────────────────────────
_debug = {
    "mic_rms": 0, "mic_peak": 0,
    "mic_chunks_in": 0, "spk_chunks_out": 0,
    "mic_q": 0, "spk_q": 0,
    "gemini_connected": False, "gemini_attempt": 0,
    "turns_completed": 0, "current_turn": 0,
    "last_tool_call": "", "camera_frames": 0,
    "mic_waveform": [], "spk_waveform": [],
    "mic_gain": 3.0, "volume": 0.45,
    "persons_detected": 0, "greetings_sent": 0,
    "current_detections": 0,
    "last_voice_route": "",
    "openrouter_configured": False,
    "browserbase_configured": False,
    "voice_backend": "",
    "last_transcript": "",
    "point_target": "none",
    "movement_running": False,
}

# ── MJPEG feed queue ─────────────────────────────────────────────────
_web_jpeg_queue = queue.Queue(maxsize=3)

# ── Greeting logic ───────────────────────────────────────────────────
GREET_COOLDOWN = 30.0  # seconds before re-greeting
_last_greet_time = 0
_greet_lock = threading.Lock()


def _greet_person():
    """Send a greeting prompt to Gemini and start the allowlisted wave."""
    global _last_greet_time

    with _greet_lock:
        now = time.time()
        if now - _last_greet_time < GREET_COOLDOWN:
            return
        if not (_gemini_session and _gemini_loop):
            return
        _last_greet_time = now

    _debug["greetings_sent"] = _debug.get("greetings_sent", 0) + 1

    # This application event is trusted and does not need model-selected tools.
    start_movement("wave")

    prompt = (
        "[SYSTEM EVENT: PERSON APPEARED] "
        "Say hello warmly. Keep it short — 1-2 sentences max."
    )

    print(f"[greeter] Sending greeting to Gemini", flush=True)
    try:
        future = asyncio.run_coroutine_threadsafe(
            _gemini_session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
                turn_complete=True,
            ),
            _gemini_loop,
        )
        future.result(timeout=5)
    except Exception as e:
        print(f"[greeter] Error sending greeting: {e}", flush=True)


# ── YOLO detection loop ─────────────────────────────────────────────
def detector_loop():
    """Run YOLO person detection + face gating + greeting trigger."""
    import pycuda.driver as cuda
    cuda.init()
    ctx = cuda.Device(0).make_context()

    try:
        from bbai import Detector
        from bbai.detector import COCO_NAMES
        print("[detector] Loading YOLO model...", flush=True)
        engine_path = Path.home() / ".cache/bracketbot-ai/yolo26n/yolo26n.engine"
        if not engine_path.exists():
            print(f"[detector] Engine not found at {engine_path}", flush=True)
            return
        model = Detector.__new__(Detector)
        model.device = 0
        model.verbose = False
        model.engine = None
        model.context = None
        model.names = {i: n for i, n in enumerate(COCO_NAMES)}
        model.model_dir = engine_path.parent
        model.model_path = engine_path
        model._load_model()
        print(f"[detector] Model loaded: {engine_path}", flush=True)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cam = Reader("camera.head", keeptime=False)
        cam.__enter__()
        time.sleep(0.5)

        cam_cfg = Config("cam_head")
        img_w = cam_cfg.width // 2

        print("[detector] Running (end2end output: 1x300x6)", flush=True)

        COLOR_NEW = (0, 160, 255)       # orange BGR
        COLOR_GREETED = (0, 200, 80)    # green BGR
        COLOR_NO_FACE = (180, 180, 180) # gray BGR
        CONF_THRESH = 0.40
        PERSON_CLS = 0

        total_persons = 0
        frame_count = 0
        last_det_log = time.time()

        while not stop_event.is_set():
            if not cam.ready():
                time.sleep(0.05)
                continue

            stereo = cam.data["rgb"].copy()
            left = stereo[:, :img_w, :]  # RGB
            frame_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
            orig_h, orig_w = frame_bgr.shape[:2]

            # ── Run inference manually (bypass Detector._postprocess) ──
            try:
                img, meta = model._preprocess(left)
                np.copyto(model.inputs[0]['host'], img.ravel())
                import pycuda.driver as drv
                drv.memcpy_htod_async(model.inputs[0]['device'], model.inputs[0]['host'], model.stream)
                for inp in model.inputs:
                    model.context.set_tensor_address(inp['name'], int(inp['device']))
                for out in model.outputs:
                    model.context.set_tensor_address(out['name'], int(out['device']))
                model.context.execute_async_v3(stream_handle=model.stream.handle)
                for out in model.outputs:
                    drv.memcpy_dtoh_async(out['host'], out['device'], model.stream)
                model.stream.synchronize()
                raw = model.outputs[0]['host'].reshape(model.outputs[0]['shape'])
            except Exception as e:
                print(f"[detector] Inference error: {e}", flush=True)
                raw = None

            # ── Parse end2end output (1, 300, 6): x1,y1,x2,y2,conf,cls ──
            detections = []
            if raw is not None:
                preds = raw[0]  # (300, 6) — DO NOT transpose
                # Vectorized filter: conf >= threshold AND class == person
                mask = (preds[:, 4] >= CONF_THRESH) & (preds[:, 5].astype(int) == PERSON_CLS)
                valid = preds[mask]
                if len(valid) > 0:
                    px, py = meta['pad']
                    r = meta['ratio']
                    coords = valid[:, :4].copy()
                    coords[:, [0, 2]] = (coords[:, [0, 2]] - px) / r
                    coords[:, [1, 3]] = (coords[:, [1, 3]] - py) / r
                    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, orig_w)
                    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, orig_h)
                    for i in range(len(valid)):
                        detections.append((
                            int(coords[i, 0]), int(coords[i, 1]),
                            int(coords[i, 2]), int(coords[i, 3]),
                            float(valid[i, 4]),
                        ))

            num_dets = len(detections)
            _debug["current_detections"] = num_dets
            observed_at = time.monotonic()
            _person_targets.update(
                detections,
                orig_w,
                orig_h,
                observed_at=observed_at,
            )
            primary_target = _person_targets.select("primary", now=observed_at)
            _debug["point_target"] = (
                "none"
                if primary_target is None
                else f"{primary_target.arm} arm, x={primary_target.x_offset:+.2f}"
            )

            has_face_this_frame = False

            for x1, y1, x2, y2, conf in detections:
                # Face detection within person bbox
                person_crop = frame_bgr[y1:y2, x1:x2]
                face_found = False
                if person_crop.size > 0:
                    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=1, minSize=(20, 20)
                    )
                    face_found = len(faces) > 0

                if face_found:
                    has_face_this_frame = True
                    with _greet_lock:
                        recently_greeted = (time.time() - _last_greet_time) < GREET_COOLDOWN
                    color = COLOR_GREETED if recently_greeted else COLOR_NEW
                    label = "Greeted" if recently_greeted else f"Person {conf:.0%}"
                else:
                    color = COLOR_NO_FACE
                    label = f"Person {conf:.0%}"

                if primary_target is not None and (
                    x1, y1, x2, y2
                ) == (
                    primary_target.x1,
                    primary_target.y1,
                    primary_target.x2,
                    primary_target.y2,
                ):
                    label = f"Point target | {label}"

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4), font, 0.55,
                            (0, 0, 0), 1, cv2.LINE_AA)

            # Trigger greeting
            if has_face_this_frame:
                with _greet_lock:
                    should_greet = (time.time() - _last_greet_time) >= GREET_COOLDOWN
                if should_greet:
                    total_persons += 1
                    _debug["persons_detected"] = total_persons
                    threading.Thread(target=_greet_person, daemon=True).start()

            # Encode annotated frame for MJPEG feed
            _, encoded = cv2.imencode('.jpg', frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = encoded.tobytes()
            try:
                _web_jpeg_queue.put_nowait(frame_bytes)
            except queue.Full:
                try:
                    _web_jpeg_queue.get_nowait()
                except queue.Empty:
                    pass
                _web_jpeg_queue.put_nowait(frame_bytes)

            frame_count += 1
            _debug["camera_frames"] = frame_count

            now = time.time()
            if now - last_det_log >= 10.0:
                print(f"[detector] {frame_count} frames, {num_dets} persons", flush=True)
                last_det_log = now

            time.sleep(0.1)  # ~10 Hz

    except Exception as e:
        print(f"[detector] Error: {e}\n{traceback.format_exc()}", flush=True)
    finally:
        try:
            cam.__exit__(None, None, None)
        except Exception:
            pass
        try:
            ctx.pop()
        except Exception:
            pass


# ── Web dashboard ────────────────────────────────────────────────────
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

web_app = FastAPI()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head><title>Greeter Debug</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #0d1117; color: #e6edf3;
       max-width: 900px; margin: 0 auto; padding: 16px; }
h1 { font-size: 18px; text-align: center; margin-bottom: 4px; }
.sub { text-align: center; color: #484f58; font-size: 12px; margin-bottom: 16px; }

.card { background: #161b22; border: 1px solid #21262d; border-radius: 8px;
        padding: 12px; margin-bottom: 12px; }
.card h2 { font-size: 13px; color: #8b949e; text-transform: uppercase;
           letter-spacing: 1px; margin-bottom: 8px; }

.row { display: flex; gap: 12px; margin-bottom: 12px; }
.row > .card { flex: 1; margin-bottom: 0; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
         font-size: 12px; font-weight: 600; }
.badge.on { background: #238636; color: #fff; }
.badge.off { background: #da3633; color: #fff; }

.vu-bg { height: 24px; background: #0d1117; border-radius: 4px; overflow: hidden;
         margin: 4px 0; position: relative; }
.vu-fill { height: 100%; border-radius: 4px; transition: width 0.15s; }
.vu-fill.green { background: #238636; }
.vu-fill.yellow { background: #d29922; }
.vu-fill.red { background: #da3633; }
.vu-label { position: absolute; right: 6px; top: 3px; font-size: 12px;
            font-family: monospace; color: #e6edf3; }

.q-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; }
.q-name { width: 30px; color: #8b949e; }
.q-bg { flex: 1; height: 16px; background: #0d1117; border-radius: 3px; overflow: hidden; }
.q-fill { height: 100%; transition: width 0.15s; border-radius: 3px; }
.q-val { width: 50px; text-align: right; font-family: monospace; font-size: 11px; }

canvas { width: 100%; height: 80px; background: #0d1117; border: 1px solid #21262d;
         border-radius: 6px; display: block; margin: 4px 0; }

.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; font-size: 12px; }
.stats .k { color: #8b949e; }
.stats .v { font-family: monospace; text-align: right; }

.ctrl-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 12px; }
.ctrl-row label { color: #8b949e; width: 70px; }
.ctrl-row input[type=range] { flex: 1; }
.ctrl-row .val { font-family: monospace; width: 40px; text-align: right; }

#feed { width: 100%; border-radius: 6px; border: 1px solid #21262d; display: block; }

.wasd-wrap { display: flex; align-items: center; gap: 16px; margin-top: 6px; }
.key-grid { display: flex; flex-direction: column; align-items: center; gap: 3px; }
.key-row { display: flex; gap: 3px; }
.key {
  width: 32px; height: 32px; background: #1a1a1a; border: 1px solid #333;
  border-radius: 4px; display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 600; color: #666; user-select: none;
  transition: all 0.1s;
}
.key.active { background: #ff9800; color: #000; border-color: #ff9800; }
.key.shift-key { width: 70px; font-size: 10px; }
.speed-info { font-size: 11px; color: #666; }
.speed-info .sval { color: #e0e0e0; font-weight: 500; }
.speed-info .fast { color: #ff9800; font-weight: 600; }
.drive-gain-wrap { display: flex; align-items: center; gap: 6px; margin-top: 4px; }
.drive-gain-wrap input[type=range] { width: 80px; accent-color: #ff9800; }
</style>
</head>
<body>
<h1>Greeter Debug</h1>
<div class="sub">YOLO person detection + Gemini greeting</div>

<div class="card">
  <h2>Camera Feed</h2>
  <img id="feed" alt="Camera Feed" src="/feed">
</div>

<div class="row">
  <div class="card">
    <h2>Gemini</h2>
    <span class="badge off" id="gBadge">disconnected</span>
    <span style="font-size:12px;color:#8b949e;margin-left:8px" id="gAttempt"></span>
  </div>
  <div class="card">
    <h2>Detection</h2>
    <div class="stats">
      <div class="k">Current</div><div class="v" id="curDets">0</div>
      <div class="k">Total persons</div><div class="v" id="totalPersons">0</div>
      <div class="k">Greetings sent</div><div class="v" id="greetsSent">0</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>Mic Input</h2>
  <div class="vu-bg">
    <div class="vu-fill green" id="micBar" style="width:0%"></div>
    <div class="vu-label" id="micRms">0</div>
  </div>
  <canvas id="micCanvas" width="1600" height="160"></canvas>
</div>

<div class="card">
  <h2>Speaker Output</h2>
  <canvas id="spkCanvas" width="1600" height="160"></canvas>
</div>

<div class="card">
  <h2>Queue Health</h2>
  <div class="q-row">
    <div class="q-name">MIC</div>
    <div class="q-bg"><div class="q-fill" id="mqFill" style="width:0%;background:#58a6ff"></div></div>
    <div class="q-val" id="mqVal">0/50</div>
  </div>
  <div class="q-row">
    <div class="q-name">SPK</div>
    <div class="q-bg"><div class="q-fill" id="sqFill" style="width:0%;background:#3fb950"></div></div>
    <div class="q-val" id="sqVal">0/200</div>
  </div>
</div>

<div class="card">
  <h2>Stats</h2>
  <div class="stats">
    <div class="k">Turn</div><div class="v" id="turnNum">0</div>
    <div class="k">Turns done</div><div class="v" id="turnsDone">0</div>
  </div>
</div>

<div class="card">
  <h2>Controls</h2>
  <div class="ctrl-row">
    <label>Mic gain</label>
    <input type="range" id="gainSlider" min="0.5" max="5.0" step="0.1" value="3.0">
    <div class="val" id="gainVal">3.0</div>
  </div>
  <div class="ctrl-row">
    <label>Volume</label>
    <input type="range" id="volSlider" min="0.1" max="1.0" step="0.05" value="0.45">
    <div class="val" id="volVal">0.45</div>
  </div>
</div>

<div class="card">
  <h2>Drive (WASD)</h2>
  <div class="wasd-wrap">
    <div class="key-grid">
      <div class="key-row"><div class="key" id="key-w">W</div></div>
      <div class="key-row">
        <div class="key" id="key-a">A</div>
        <div class="key" id="key-s">S</div>
        <div class="key" id="key-d">D</div>
      </div>
      <div class="key-row"><div class="key shift-key" id="key-shift">Shift</div></div>
    </div>
    <div class="speed-info">
      <div>Linear: <span class="sval" id="linear-speed">0.00</span> m/s</div>
      <div>Angular: <span class="sval" id="angular-speed">0.00</span> rad/s</div>
      <div>Mode: <span id="speed-mode">Half Speed</span></div>
      <div class="drive-gain-wrap">
        Gain: <input type="range" id="driveGainSlider" min="0.2" max="2.0" step="0.1" value="1.0">
        <span class="sval" id="driveGainVal">1.0x</span>
      </div>
    </div>
  </div>
</div>

<script>
function drawWave(id, samples, color) {
  const c = document.getElementById(id);
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#21262d';
  ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
  if (!samples || !samples.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i=0; i<w; i++) {
    const idx = Math.floor(i * samples.length / w);
    const y = h/2 - (samples[idx]/32768) * h/2;
    if (i===0) ctx.moveTo(i,y); else ctx.lineTo(i,y);
  }
  ctx.stroke();
}

async function poll() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const b = document.getElementById('gBadge');
    b.textContent = d.gemini_connected ? 'connected' : 'disconnected';
    b.className = 'badge ' + (d.gemini_connected ? 'on' : 'off');
    document.getElementById('gAttempt').textContent =
      d.gemini_attempt > 1 ? 'attempt #'+d.gemini_attempt : '';
    document.getElementById('turnNum').textContent = d.current_turn;
    document.getElementById('turnsDone').textContent = d.turns_completed;
    document.getElementById('curDets').textContent = d.current_detections;
    document.getElementById('totalPersons').textContent = d.persons_detected;
    document.getElementById('greetsSent').textContent = d.greetings_sent;
    const pct = Math.min(100, d.mic_rms / 8000 * 100);
    const bar = document.getElementById('micBar');
    bar.style.width = pct + '%';
    bar.className = 'vu-fill ' + (pct > 70 ? 'red' : pct > 30 ? 'yellow' : 'green');
    document.getElementById('micRms').textContent = d.mic_rms;
    drawWave('micCanvas', d.mic_waveform, '#58a6ff');
    drawWave('spkCanvas', d.spk_waveform, '#3fb950');
    const mq = d.mic_q, sq = d.spk_q;
    document.getElementById('mqFill').style.width = (mq/50*100)+'%';
    document.getElementById('mqFill').style.background = mq > 40 ? '#da3633' : '#58a6ff';
    document.getElementById('mqVal').textContent = mq+'/50';
    document.getElementById('sqFill').style.width = (sq/200*100)+'%';
    document.getElementById('sqVal').textContent = sq+'/200';
    document.getElementById('gainVal').textContent = d.mic_gain.toFixed(1);
    document.getElementById('volVal').textContent = d.volume.toFixed(2);
  } catch(e) {}
}

document.getElementById('gainSlider').addEventListener('input', async function() {
  const v = this.value;
  document.getElementById('gainVal').textContent = parseFloat(v).toFixed(1);
  await fetch('/api/set?mic_gain='+v, {method:'POST'});
});
document.getElementById('volSlider').addEventListener('input', async function() {
  const v = this.value;
  document.getElementById('volVal').textContent = parseFloat(v).toFixed(2);
  await fetch('/api/set?volume='+v, {method:'POST'});
});

setInterval(poll, 200);
poll();

// --- WASD Drive ---
const wasdKeys = { w:false, a:false, s:false, d:false };
let wasdShift = false, wasdDriveGain = 1.0;

function sendDriveCmd() {
  let combo = '';
  if (wasdKeys.w) combo += 'w';
  if (wasdKeys.s) combo += 's';
  if (wasdKeys.a) combo += 'a';
  if (wasdKeys.d) combo += 'd';
  const comboDisplay = {
    'w':{lin:0.20,ang:0},'s':{lin:-0.20,ang:0},
    'a':{lin:0,ang:0.92},'d':{lin:0,ang:-0.92},
    'wa':{lin:0.165,ang:0.70},'wd':{lin:0.165,ang:-0.70},
    'sa':{lin:-0.165,ang:-0.70},'sd':{lin:-0.165,ang:0.70},
    '':{lin:0,ang:0},
  };
  const base = comboDisplay[combo] || comboDisplay[''];
  const m = (wasdShift ? 1.0 : 0.5) * wasdDriveGain;
  document.getElementById("linear-speed").textContent = (base.lin*m).toFixed(2);
  document.getElementById("angular-speed").textContent = (base.ang*m).toFixed(2);
  const modeEl = document.getElementById("speed-mode");
  modeEl.textContent = wasdShift ? "Full Speed" : "Half Speed";
  modeEl.className = wasdShift ? "fast" : "";
  fetch("/api/drive", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({keys:combo, shift:wasdShift, gain:wasdDriveGain})
  }).catch(() => {});
}

function setWasdKey(key, state) {
  const k = key.toLowerCase();
  if (k in wasdKeys) {
    wasdKeys[k] = state;
    const el = document.getElementById("key-"+k);
    if (el) el.classList.toggle("active", state);
    sendDriveCmd();
  }
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.repeat) return;
  if (e.key === "Shift") { wasdShift=true; document.getElementById("key-shift").classList.add("active"); sendDriveCmd(); return; }
  setWasdKey(e.key, true);
});
document.addEventListener("keyup", e => {
  if (e.key === "Shift") { wasdShift=false; document.getElementById("key-shift").classList.remove("active"); sendDriveCmd(); return; }
  setWasdKey(e.key, false);
});
window.addEventListener("blur", () => {
  Object.keys(wasdKeys).forEach(k => { wasdKeys[k]=false; const el=document.getElementById("key-"+k); if(el) el.classList.remove("active"); });
  wasdShift=false; document.getElementById("key-shift").classList.remove("active");
  sendDriveCmd();
});

document.getElementById("driveGainSlider").addEventListener("input", function() {
  wasdDriveGain = parseFloat(this.value);
  document.getElementById("driveGainVal").textContent = wasdDriveGain.toFixed(1) + "x";
  sendDriveCmd();
});
</script>
</body>
</html>
"""

@web_app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML

@web_app.get("/api/status")
def api_status():
    return _debug

@web_app.post("/api/set")
def api_set(mic_gain: float = None, volume: float = None):
    if mic_gain is not None:
        _debug["mic_gain"] = mic_gain
    if volume is not None:
        _debug["volume"] = volume
    return {"ok": True}

@web_app.post("/api/drive")
async def api_drive(request: Request):
    body = await request.json()
    with _wasd_lock:
        _wasd_cmd['keys'] = body.get('keys', '')
        _wasd_cmd['shift'] = body.get('shift', False)
        _wasd_cmd['gain'] = body.get('gain', 1.0)
    return JSONResponse({"status": "ok"})


def _robot_local_request(request: Request):
    return request.client is not None and request.client.host in {"127.0.0.1", "::1"}


@web_app.post("/api/action")
async def api_action(request: Request):
    """Start one camera-backed action for trusted robot-local callers."""
    if not _robot_local_request(request):
        return JSONResponse(
            {"ok": False, "message": "Camera actions are robot-local only"},
            status_code=403,
        )
    body = await request.json()
    action = str(body.get("action", ""))
    if action not in {"point", "point-left", "point-right"}:
        return JSONResponse(
            {"ok": False, "message": "Unknown camera action"},
            status_code=404,
        )
    started, message = start_robot_action(action)
    return JSONResponse(
        {"ok": started, "message": message},
        status_code=202 if started else 409,
    )


@web_app.post("/api/action/stop")
def api_action_stop(request: Request):
    """Ask a camera-backed gesture to return safely to its start pose."""
    if not _robot_local_request(request):
        return JSONResponse(
            {"ok": False, "message": "Camera actions are robot-local only"},
            status_code=403,
        )
    if not _movement_playback_lock.locked():
        return JSONResponse(
            {"ok": False, "message": "No camera action is running"},
            status_code=409,
        )
    _movement_cancel_event.set()
    return {"ok": True, "message": "Stop requested"}

async def _generate_frames():
    while not stop_event.is_set():
        try:
            frame = _web_jpeg_queue.get_nowait()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except queue.Empty:
            await asyncio.sleep(0.05)

@web_app.get("/feed")
async def feed():
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Audio I/O thread ──────────────────────────────────────────────────
def audio_io_loop(volume):
    """Read mic from SHM, write speaker audio to SHM."""
    playing = False

    with Reader("mic.audio") as r_mic, \
         Writer("speaker.audio", Type("speaker_audio")) as w_spk:

        while not w_spk.ready():
            pass
        for _ in range(3):
            with w_spk.buf() as b:
                b["audio"] = np.zeros((CHUNK, CFG.channels), dtype=np.int16)

        print("[audio] I/O loop running", flush=True)
        mic_chunks_sent = 0
        spk_chunks_played = 0
        last_stats = time.time()

        while not stop_event.is_set():
            did_work = False

            # Read mic
            if r_mic.ready():
                did_work = True
                audio = r_mic.data["audio"].copy()
                mic_chunks_sent += 1
                flat = audio.flatten()
                _debug["mic_rms"] = int(np.sqrt(np.mean(flat.astype(np.float64)**2)))
                _debug["mic_peak"] = int(np.max(np.abs(flat)))
                _debug["mic_chunks_in"] = mic_chunks_sent
                step = max(1, len(flat) // 800)
                _debug["mic_waveform"] = flat[::step].tolist()
                try:
                    mic_queue.put_nowait(audio)
                except queue.Full:
                    try:
                        mic_queue.get_nowait()
                    except queue.Empty:
                        pass
                    mic_queue.put_nowait(audio)

            # Write speaker
            if w_spk._update():
                did_work = True

                if interrupt_flag.is_set():
                    while not speaker_queue.empty():
                        try:
                            speaker_queue.get_nowait()
                        except queue.Empty:
                            break
                    interrupt_flag.clear()
                    playing = False

                if not playing and speaker_queue.qsize() >= JITTER_BUFFER_CHUNKS:
                    playing = True

                if playing:
                    try:
                        chunk = speaker_queue.get_nowait()
                        spk_chunks_played += 1
                    except queue.Empty:
                        chunk = np.zeros(CHUNK, dtype=np.int16)
                else:
                    chunk = np.zeros(CHUNK, dtype=np.int16)

                _debug["spk_chunks_out"] = spk_chunks_played
                step = max(1, len(chunk) // 800)
                _debug["spk_waveform"] = chunk[::step].tolist()

                with w_spk.buf() as b:
                    b["audio"] = chunk.reshape(-1, CFG.channels)

            _debug["mic_q"] = mic_queue.qsize()
            _debug["spk_q"] = speaker_queue.qsize()

            # Periodic stats
            now = time.time()
            if now - last_stats >= 10.0:
                print(f"[audio] mic_in={mic_chunks_sent} spk_out={spk_chunks_played} "
                      f"mic_q={mic_queue.qsize()} spk_q={speaker_queue.qsize()}", flush=True)
                mic_chunks_sent = 0
                spk_chunks_played = 0
                last_stats = now

            if not did_work:
                time.sleep(0.001)


def _queue_local_speech(synthesizer, text, args):
    """Synthesize one reply and feed it through the existing BBOS speaker queue."""
    try:
        from .local_voice import speaker_chunks
    except ImportError:
        from local_voice import speaker_chunks

    pcm = synthesizer.synthesize(text, BBOS_RATE)
    volume = _debug.get("volume", args.volume)
    pcm = (pcm.astype(np.float32) * volume).clip(-32768, 32767).astype(np.int16)
    chunks = speaker_chunks(pcm, CHUNK, CFG.channels)
    for chunk in chunks:
        while not stop_event.is_set():
            try:
                speaker_queue.put(chunk, timeout=0.1)
                break
            except queue.Full:
                continue
    # Avoid feeding the robot's own reply straight back into transcription.
    stop_event.wait(len(pcm) / BBOS_RATE + 0.5)
    while not mic_queue.empty():
        try:
            mic_queue.get_nowait()
        except queue.Empty:
            break


def local_voice_session(args):
    """Wake word -> whisper.cpp -> VoiceRouter -> espeak-ng, with no Gemini."""
    try:
        from .local_voice import (
            EspeakSynthesizer,
            LocalVoiceError,
            SpeechSegmenter,
            WhisperCppTranscriber,
        )
    except ImportError:
        from local_voice import (
            EspeakSynthesizer,
            LocalVoiceError,
            SpeechSegmenter,
            WhisperCppTranscriber,
        )

    llm = OpenRouterClient(
        model=args.openrouter_model,
        timeout=args.openrouter_timeout,
        response_cache=default_question_response_cache(),
        seed_pairs=default_seed_pairs(),
    )
    voice_router = VoiceRouter(
        llm,
        action_executor=start_robot_action,
        stop_executor=stop_robot_action,
    )
    _debug["openrouter_configured"] = llm.configured
    _debug["browserbase_configured"] = llm.web_search.configured
    _debug["voice_backend"] = "local"

    transcriber = WhisperCppTranscriber(
        args.whisper_bin,
        args.whisper_model,
        threads=args.whisper_threads,
    )
    synthesizer = EspeakSynthesizer(
        args.espeak_bin,
        voice=args.espeak_voice,
        words_per_minute=args.espeak_speed,
    )
    transcriber.validate()
    synthesizer.validate()
    segmenter = SpeechSegmenter(
        BBOS_RATE,
        threshold_db=args.vad_threshold_db,
        pre_roll_s=args.pre_roll,
        trailing_silence_s=args.trailing_silence,
        start_grace_s=args.wake_grace,
        max_utterance_s=args.max_utterance,
    )

    print(
        "[local-voice] Ready: say the configured wake phrase, then your question",
        flush=True,
    )
    last_wake_active = False
    pending_wake = False
    with Reader("wakeword.state", keeptime=False) as wakeword:
        while not stop_event.is_set():
            triggered = False
            if wakeword.ready():
                active = bool(wakeword.data["active"])
                triggered = active and not last_wake_active
                last_wake_active = active
                if triggered:
                    pending_wake = True
                    print("[local-voice] Wake phrase detected; recording...", flush=True)

            try:
                audio = mic_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            flat = audio.reshape(-1)
            gain = _debug.get("mic_gain", args.mic_gain)
            boosted = (
                flat.astype(np.float32) * gain
            ).clip(-32768, 32767).astype(np.int16)
            if pending_wake:
                # A wake event and a mic frame are produced by independent
                # daemons. Keep the wake latched until a real audio frame
                # arrives instead of dropping it when the mic queue is empty.
                triggered = True
                pending_wake = False
            if args.local_always_listen and not segmenter.recording:
                rms = max(1.0, float(np.sqrt(np.mean(boosted.astype(np.float64) ** 2))))
                dbfs = 20.0 * np.log10(rms / 32768.0)
                triggered = dbfs >= args.vad_threshold_db

            utterance_audio = segmenter.push(boosted, triggered=triggered)
            if utterance_audio is None:
                continue

            try:
                transcript = transcriber.transcribe(utterance_audio, BBOS_RATE)
                _debug["last_transcript"] = transcript
                if not transcript:
                    print("[local-voice] No speech recognized", flush=True)
                    segmenter.reset()
                    continue
                print(f"[local-voice] Heard: {transcript}", flush=True)
                decision = voice_router.route(transcript)
                _debug["last_voice_route"] = decision.kind.value
                if decision.kind == RouteKind.ACTION and decision.action_started is None:
                    started, status = start_robot_action(decision.action)
                    reply = decision.reply if started else status
                else:
                    reply = decision.reply or "I did not understand that."
                print(f"[local-voice] Reply: {reply}", flush=True)
                _queue_local_speech(synthesizer, reply, args)
                _debug["turns_completed"] += 1
            except LocalVoiceError as exc:
                print(f"[local-voice] {exc}", flush=True)
            finally:
                segmenter.reset()


# ── Gemini session ────────────────────────────────────────────────────
async def gemini_session(args):
    global _gemini_session, _gemini_loop

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[!] GEMINI_API_KEY not set in .env", flush=True)
        return

    client = genai.Client(api_key=api_key)
    llm = OpenRouterClient(
        model=args.openrouter_model,
        timeout=args.openrouter_timeout,
        response_cache=default_question_response_cache(),
        seed_pairs=default_seed_pairs(),
    )
    voice_router = VoiceRouter(
        llm,
        action_executor=start_robot_action,
        stop_executor=stop_robot_action,
    )
    _debug["openrouter_configured"] = llm.configured
    _debug["browserbase_configured"] = llm.web_search.configured

    tools = [
        types.FunctionDeclaration(
            name="route_utterance",
            description=(
                "Pass every human utterance to the deterministic Baymax router. "
                "It safely selects an allowlisted gesture or sends non-action "
                "conversation to OpenRouter. Preserve the person's exact words."
            ),
            behavior="NON_BLOCKING",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "utterance": types.Schema(
                        type=types.Type.STRING,
                        description="The exact words spoken by the person",
                    ),
                },
                required=["utterance"],
            ),
        ),
    ]

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(mode="VERBATIM"),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=args.voice,
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=args.system_prompt)]
        ),
        tools=[types.Tool(function_declarations=tools)],
    )

    attempt = 0
    try:
      while not stop_event.is_set():
        attempt += 1
        try:
            # Drain stale mic data before (re)connect
            while not mic_queue.empty():
                try: mic_queue.get_nowait()
                except queue.Empty: break

            _debug["gemini_attempt"] = attempt
            if attempt > 1:
                delay = min(2 ** (attempt - 2), 10)
                print(f"[gemini] Reconnecting (attempt #{attempt}) in {delay}s...", flush=True)
                await asyncio.sleep(delay)

            async with client.aio.live.connect(model=args.model, config=config) as session:
                _gemini_session = session
                _gemini_loop = asyncio.get_event_loop()
                # Final transcripts are the authority for motion. Gemini's
                # function argument must match one before the router sees it.
                transcript_queue = asyncio.Queue(maxsize=4)
                print(f"[gemini] Connected (attempt #{attempt})", flush=True)
                _debug["gemini_connected"] = True
                _debug["gemini_attempt"] = attempt
                attempt = 1  # reset on successful connect

                # ── Handle tool calls ─────────────────────────────
                async def _handle_tools(tool_call):
                    for fc in tool_call.function_calls:
                        _debug["last_tool_call"] = fc.name
                        print(f"[tool] {fc.name}({fc.args})", flush=True)
                        try:
                            if fc.name == "route_utterance":
                                claimed = str(fc.args.get("utterance", ""))
                                utterance = None
                                deadline = asyncio.get_event_loop().time() + 2.0
                                while utterance is None:
                                    remaining = deadline - asyncio.get_event_loop().time()
                                    if remaining <= 0:
                                        break
                                    try:
                                        stamp, transcript = await asyncio.wait_for(
                                            transcript_queue.get(), timeout=remaining
                                        )
                                    except asyncio.TimeoutError:
                                        break
                                    if time.monotonic() - stamp > 3.0:
                                        continue
                                    if utterances_match(transcript, claimed):
                                        utterance = transcript
                                if utterance is None:
                                    result = {
                                        "kind": "error",
                                        "reply": (
                                            "I could not verify that voice command. "
                                            "Please say it again."
                                        ),
                                    }
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                name=fc.name,
                                                id=fc.id,
                                                response=result,
                                                scheduling="WHEN_IDLE",
                                            )
                                        ]
                                    )
                                    continue
                                decision = await asyncio.to_thread(
                                    voice_router.route, utterance
                                )
                                _debug["last_voice_route"] = decision.kind.value
                                if (
                                    decision.kind == RouteKind.ACTION
                                    and decision.action_started is None
                                ):
                                    started, status = start_robot_action(decision.action)
                                    if not started:
                                        result = {
                                            "kind": "error",
                                            "reply": status,
                                        }
                                    else:
                                        result = {
                                            "kind": decision.kind.value,
                                            "status": "started",
                                            "action": decision.action,
                                            "reply": decision.reply,
                                        }
                                elif decision.kind == RouteKind.ACTION:
                                    result = {
                                        "kind": decision.kind.value,
                                        "status": "started",
                                        "action": decision.action,
                                        "reply": decision.reply,
                                    }
                                else:
                                    result = {
                                        "kind": decision.kind.value,
                                        "reply": decision.reply or "",
                                    }
                            else:
                                result = {"error": f"Unknown tool: {fc.name}"}
                        except Exception as e:
                            result = {"error": str(e)}

                        await session.send_tool_response(
                            function_responses=[
                                types.FunctionResponse(
                                    name=fc.name,
                                    id=fc.id,
                                    response=result,
                                    scheduling="WHEN_IDLE",
                                )
                            ]
                        )

                # ── Send audio ────────────────────────────────────
                async def send_audio():
                    sent = 0
                    last_log = time.time()
                    while not stop_event.is_set():
                        try:
                            audio = mic_queue.get_nowait()
                            if audio.ndim > 1:
                                audio = audio.flatten()
                            gain = _debug.get("mic_gain", args.mic_gain)
                            boosted = (
                                audio.astype(np.float32) * gain
                            ).clip(-32768, 32767).astype(np.int16)
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=boosted.tobytes(),
                                    mime_type="audio/pcm;rate=16000",
                                )
                            )
                            sent += 1
                            now = time.time()
                            if now - last_log >= 10.0:
                                print(f"[mic->gemini] {sent} chunks sent", flush=True)
                                sent = 0
                                last_log = now
                        except queue.Empty:
                            await asyncio.sleep(0.02)

                # ── Receive audio ─────────────────────────────────
                async def receive():
                    pcm_buffer = bytearray()
                    turn_count = 0
                    audio_chunks_recv = 0
                    turn_start = None
                    while not stop_event.is_set():
                        turn_count += 1
                        _debug["current_turn"] = turn_count
                        turn_start = None
                        audio_chunks_recv = 0
                        turn = session.receive()
                        async for response in turn:
                            input_transcription = (
                                getattr(response.server_content, "input_transcription", None)
                                if response.server_content else None
                            )
                            if input_transcription and input_transcription.text:
                                item = (time.monotonic(), input_transcription.text.strip())
                                if transcript_queue.full():
                                    try:
                                        transcript_queue.get_nowait()
                                    except asyncio.QueueEmpty:
                                        pass
                                transcript_queue.put_nowait(item)

                            if (response.server_content
                                    and response.server_content.model_turn
                                    and response.server_content.model_turn.parts):
                                if turn_start is None:
                                    turn_start = time.time()
                                for part in response.server_content.model_turn.parts:
                                    if (hasattr(part, 'inline_data')
                                            and part.inline_data
                                            and part.inline_data.data):
                                        audio_data = part.inline_data.data
                                        audio_chunks_recv += 1
                                        pcm_24k = np.frombuffer(audio_data, dtype=np.int16)
                                        pcm_16k = resample_24k_to_16k(pcm_24k)
                                        vol = _debug.get("volume", args.volume)
                                        pcm_16k = (
                                            pcm_16k.astype(np.float32) * vol
                                        ).clip(-32768, 32767).astype(np.int16)
                                        pcm_buffer.extend(pcm_16k.tobytes())

                                        while len(pcm_buffer) >= CHUNK * 2:
                                            chunk_bytes = bytes(pcm_buffer[:CHUNK * 2])
                                            del pcm_buffer[:CHUNK * 2]
                                            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
                                            try:
                                                speaker_queue.put_nowait(chunk)
                                            except queue.Full:
                                                try: speaker_queue.get_nowait()
                                                except queue.Empty: pass
                                                speaker_queue.put_nowait(chunk)

                                    if hasattr(part, 'text') and part.text:
                                        print(f"[gemini] Text: {part.text}", flush=True)

                            if (response.server_content
                                    and response.server_content.turn_complete):
                                _debug["turns_completed"] += 1
                                dur = f" ({time.time()-turn_start:.1f}s)" if turn_start else ""
                                print(f"[gemini] Turn #{turn_count} complete{dur}, "
                                      f"{audio_chunks_recv} audio chunks", flush=True)

                            if (response.server_content
                                    and hasattr(response.server_content, 'interrupted')
                                    and response.server_content.interrupted):
                                interrupt_flag.set()

                            if response.tool_call:
                                asyncio.create_task(_handle_tools(response.tool_call))

                # ── Run tasks ─────────────────────────────────────
                tasks = [asyncio.create_task(c) for c in [send_audio(), receive()]]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()

            # Session exited — loop will reconnect
            _debug["gemini_connected"] = False
            _gemini_session = None
            print("[gemini] Session ended, will reconnect...", flush=True)

        except Exception as e:
            _debug["gemini_connected"] = False
            _gemini_session = None
            if stop_event.is_set():
                break
            print(f"\n[gemini] Connection error: {e}", flush=True)

    finally:
        _gemini_session = None
        _gemini_loop = None
        _cleanup_arm_shm()


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Greeter: YOLO + local or Gemini voice")
    parser.add_argument(
        "--voice-backend",
        choices=("auto", "local", "gemini"),
        default=os.environ.get("BAYMAX_VOICE_BACKEND", "auto"),
        help="voice transport; auto uses Gemini only when GEMINI_API_KEY is set",
    )
    parser.add_argument("--model", default="gemini-2.5-flash-native-audio-preview-12-2025")
    parser.add_argument(
        "--voice",
        default=os.environ.get("BAYMAX_GEMINI_VOICE", "Puck"),
        help=(
            "Gemini voice (Aoede, Charon, Fenrir, Iapetus, Kore, Orus, "
            "Puck, etc.)"
        ),
    )
    parser.add_argument("--volume", type=float, default=0.45)
    parser.add_argument("--mic-gain", type=float, default=3.0)
    parser.add_argument(
        "--openrouter-model",
        default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b"),
        help="OpenRouter model slug used for non-action speech",
    )
    parser.add_argument("--openrouter-timeout", type=float, default=20.0)
    parser.add_argument(
        "--whisper-bin",
        default=os.environ.get(
            "WHISPER_CPP_BIN",
            "/home/bracketbot/.local/share/whisper.cpp/build/bin/whisper-cli",
        ),
    )
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get(
            "WHISPER_CPP_MODEL",
            "/home/bracketbot/.local/share/whisper.cpp/models/ggml-base.en.bin",
        ),
    )
    parser.add_argument("--whisper-threads", type=int, default=4)
    parser.add_argument("--espeak-bin", default="espeak-ng")
    parser.add_argument("--espeak-voice", default="en-us")
    parser.add_argument("--espeak-speed", type=int, default=165)
    parser.add_argument("--vad-threshold-db", type=float, default=-38.0)
    parser.add_argument("--pre-roll", type=float, default=1.5)
    parser.add_argument(
        "--trailing-silence",
        type=float,
        default=1.5,
        help="seconds of silence after speech before submitting the turn",
    )
    parser.add_argument(
        "--wake-grace",
        type=float,
        default=1.5,
        help="minimum listening time after the wake-word trigger",
    )
    parser.add_argument("--max-utterance", type=float, default=8.0)
    parser.add_argument(
        "--local-always-listen",
        action="store_true",
        help="start on any speech instead of requiring wakeword.state",
    )
    parser.add_argument("--system-prompt", default=(
        "You are the speech interface for a cheerful, gentle robot called BracketBot. "
        "For EVERY human utterance, call route_utterance exactly once with the "
        "person's exact words. Never answer human speech yourself and never infer "
        "or perform a physical action. After the tool responds, speak only its "
        "reply with a warm, upbeat, reassuring delivery while staying natural and "
        "concise. Text beginning '[SYSTEM EVENT:' is trusted "
        "application input: respond to that directly without calling the tool."
    ))
    args = parser.parse_args()
    if args.voice_backend == "auto":
        args.voice_backend = "gemini" if os.environ.get("GEMINI_API_KEY") else "local"

    # Pre-import so pycuda.autoinit's context lives on the main thread
    import pycuda.autoinit  # noqa: F401

    # Load saved movements from disk
    _load_movements()

    print("=" * 50, flush=True)
    print("  BracketBot Greeter", flush=True)
    print(f"  Voice backend: {args.voice_backend}", flush=True)
    if args.voice_backend == "gemini":
        print(f"  Model: {args.model}", flush=True)
        print(f"  Voice: {args.voice}", flush=True)
    print(f"  Volume: {args.volume}", flush=True)
    print(f"  Mic gain: {args.mic_gain}x", flush=True)
    print(f"  OpenRouter model: {args.openrouter_model}", flush=True)
    print(
        f"  OpenRouter: {'configured' if os.environ.get('OPENROUTER_API_KEY') else 'not configured'}",
        flush=True,
    )
    print(
        f"  Browserbase search: "
        f"{'configured' if os.environ.get('BROWSERBASE_API_KEY') else 'not configured'}",
        flush=True,
    )
    print(f"  Movements: {len(_saved_movements)} loaded", flush=True)
    print(f"  Greet cooldown: {GREET_COOLDOWN}s", flush=True)
    print("=" * 50, flush=True)

    _debug["mic_gain"] = args.mic_gain
    _debug["volume"] = args.volume

    # Start web dashboard
    web_thread = threading.Thread(
        target=lambda: uvicorn.run(web_app, host="0.0.0.0", port=8017, log_level="error"),
        daemon=True)
    web_thread.start()
    print(f"  Dashboard: http://0.0.0.0:8017", flush=True)

    # Start audio I/O thread
    audio_thread = threading.Thread(target=audio_io_loop, args=(args.volume,), daemon=True)
    audio_thread.start()

    # Start YOLO detector thread
    det_thread = threading.Thread(target=detector_loop, daemon=True)
    det_thread.start()
    print("  Detector: YOLO26n + face cascade", flush=True)

    # Start WASD drive thread
    wasd_thread = threading.Thread(target=wasd_drive_loop, daemon=True)
    wasd_thread.start()

    try:
        if args.voice_backend == "local":
            local_voice_session(args)
        else:
            asyncio.run(gemini_session(args))
    except KeyboardInterrupt:
        print("\n[+] Stopping...", flush=True)
    finally:
        stop_event.set()
        _cleanup_arm_shm()
        _cleanup_drive_writer()
        print("[+] Done", flush=True)


if __name__ == "__main__":
    main()
