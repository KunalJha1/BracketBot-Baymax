"""Person-follow runner for BracketBot. Copied to /tmp by robot_dashboard.py.

People are person-sized clusters in camera.points with optional aligned torso
colour cues; no separate camera image or neural network. Needs only BBOS
and numpy, so it runs in the BBOS venv like robot_base_mode.py. While running it
is the only writer of drive.ctrl and led.ctrl. Every decision lives in
follow_core.FollowLoop; this file only moves data between BBOS topics and the loop.

    ~/bbos/.venv/bin/python /tmp/robot_follow.py --pid-file /tmp/f.pid     # from the dashboard
    ~/bbos/.venv/bin/python /tmp/robot_follow.py --check                   # gate G0: list clusters
    ~/bbos/.venv/bin/python /tmp/robot_follow.py --dry-run --no-heartbeat  # gate G2
    ~/bbos/.venv/bin/python /tmp/robot_follow.py --rotate-only ...         # gate G3
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from follow_core import (
    FollowConfig, FollowLoop, Perception, PersonObservation, TickInputs, clamp, led_color,
    parse_command, start_refusal, status_line, wheel_twist,
)
from follow_calibration import Calibration, DEFAULT_PATH, load_calibration
from follow_perception import (
    ClusterConfig, base_to_local, find_people, floor_line, level_floor, outside_self_mask, without_self,
)
from ground_approach import GroundApproachLoop, GROUND_LINE, STANDOFF, approach_config, target_from_payload

PERIOD = 0.02  # 50 Hz control loop; drive.ctrl times out after 0.1 s
STATUS_PERIOD = 0.2
CSV_PERIOD = 0.05
STATE_TIMEOUT = 0.3
DRIVE_WRITER_PATTERNS = (
    "bbapps/greeter/main.py", "nav/main.py", "bbapps/teleop.py", "quest_teleop/main.py",
    "leader_follower_teleop.py", "live_inference.py", "robot_follow.py", "robot_base_mode.py",
    "person_tracker.py",
)
CSV_FIELDS = (
    "t", "state", "rule", "gap", "range", "bearing", "error", "v_cmd", "omega_cmd", "v", "omega",
    "measured_v", "measured_omega", "blocked", "corridor_points", "track_age", "people",
    "tick_ms", "perception_ms",
    "association", "candidates",
)
STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def build_parser():
    cfg = FollowConfig()
    parser = argparse.ArgumentParser(description="Follow one person at a held distance")
    parser.add_argument("--gap", type=float, default=cfg.gap_default)
    parser.add_argument("--v-max", type=float, default=cfg.v_max)
    parser.add_argument("--ground-approach", action="store_true",
                        help="approach one confirmed ground pose at <=0.05 m/s and speak once")
    parser.add_argument("--ground-alert-file", type=Path,
                        default=Path("/tmp/bracketbot_ground_alert.json"))
    parser.add_argument("--no-speech", action="store_true",
                        help="with --ground-approach: print the check-in line on arrival instead of "
                             "playing it (the voice assistant holds BBOS's only speaker.audio writer)")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="compute and log; never open drive.ctrl")
    mode.add_argument("--rotate-only", action="store_true", help="forward speed held at 0 (gate G3)")
    parser.add_argument("--no-heartbeat", action="store_true", help="only with --dry-run: no dashboard needed")
    mode.add_argument("--check", action="store_true", help="gate G0: print the clusters seen for 3 s, then exit")
    mode.add_argument("--preflight", action="store_true", help="validate calibration and live inputs; opens no writers")
    parser.add_argument("--no-led", action="store_true",
                        help="leave led.ctrl alone: BBOS allows one writer and the voice assistant holds it")
    parser.add_argument("--ignore-writer", action="append", default=[], metavar="PATTERN",
                        help="a DRIVE_WRITER_PATTERNS entry the caller guarantees is idle (the voice "
                             "assistant passes person_tracker.py, which it keeps from turning meanwhile)")
    parser.add_argument("--no-odom-check", action="store_true",
                        help="do not exit on wheel feedback opposing the command (false alarms while "
                             "a balancing base turns); stop, heartbeat and tilt exits remain")
    parser.add_argument("--human-gate", action="store_true",
                        help="lock only onto shapes the vision app's YOLO confirms are people "
                             "(read from --ground-alert-file)")
    parser.add_argument("--relock", action="store_true",
                        help="after a long loss, search again and follow whoever stands in the "
                             "lock-on zone instead of waiting for a restart")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_PATH,
                        help="robot-specific JSON calibration (default: ~/.config/baymax/follow.json)")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = FollowConfig()
    if not cfg.gap_min <= args.gap <= cfg.gap_max:
        parser.error(f"--gap must be between {cfg.gap_min} and {cfg.gap_max} m")
    if not 0.0 < args.v_max <= 0.30:
        parser.error("--v-max must be above 0 and at most 0.30 m/s (the drive daemon's clamp)")
    if args.no_heartbeat and not args.dry_run:
        parser.error("--no-heartbeat is only allowed with --dry-run")
    return args


def loop_config(args):
    if args.ground_approach:
        return approach_config(0.0 if args.rotate_only else args.v_max)
    cfg = replace(FollowConfig(), v_max=0.0 if args.rotate_only else args.v_max,
                  relock_after_loss=args.relock, odom_check=not args.no_odom_check)
    if args.relock:
        # "Follow whoever is in front until told to stop": after a short loss, lock again
        # (0.5 s) instead of spending 10 s re-confirming clothing colour. Logged on the
        # robot: 5.5 s stuck "confirming" a person standing 1 m away. And 30-90 stray
        # points held it BLOCKED; a chair leg at 0.5 m returns several hundred.
        cfg = replace(cfg, lost_timeout=1.5, corridor_min_points=150)
    return cfg


def read_ground_target(path, wall_now, calibration):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return target_from_payload(payload, wall_now, calibration.left_sign)


def people_from_payload(payload, wall_now, left_sign, max_age=1.0):
    """The vision app's YOLO people as robot-local (forward, left), or None if not trustworthy now."""
    try:
        if payload["schema_version"] != 1 or payload["depth_aligned"] is not True:
            return None
        ages = (wall_now - payload["camera_timestamp_ns"] / 1e9, wall_now - payload["published_at"])
        if not all(np.isfinite(age) and 0 <= age <= max_age for age in ages):
            return None
        humans = []
        for obs in payload["observations"]:
            # box_position works from any side; base_position needs a visible torso pose.
            position = obs.get("box_position") or obs.get("base_position")
            if position is None:
                continue  # seen in the image, but no depth on them
            right, forward = float(position[0]), float(position[1])
            if np.isfinite([right, forward]).all():
                humans.append((forward, right * left_sign))
        return tuple(humans)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError):
        return None


class HumanGate:
    """Feeds lock-on the people YOLO currently sees, so it never locks a plant or a pillar.

    Reuses the always-on vision app's detections: a second YOLO would cost ~0.2 s a
    frame on an already saturated CPU. If that app goes quiet the gate opens after
    ``patience`` seconds, so follow degrades to shape-only lock instead of never starting.
    """

    def __init__(self, path, left_sign, patience=3.0):
        self.path, self.left_sign, self.patience = path, left_sign, patience
        self._last_ok = None
        self._open = False
        self._explained = -math.inf

    def humans(self, wall_now):
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        humans = None if payload is None else people_from_payload(payload, wall_now, self.left_sign)
        if self._last_ok is None:
            self._last_ok = wall_now       # patience counts from the first look
        if humans is not None:
            if self._open:
                print("[follow] human gate back; locking only onto detected people", flush=True)
            self._last_ok, self._open = wall_now, False
            return humans
        if wall_now - self._last_ok > self.patience:
            if not self._open:
                print("[follow] human gate unavailable; shape-only lock", flush=True)
            self._open = True
            return None
        return ()

    def explain(self, t, perception, gate_dist, every=2.0):
        """Say why nothing is lockable, so one test run shows a shape/detector disagreement."""
        if perception.humans is None or not perception.people or t - self._explained < every:
            return
        if any(math.hypot(o.forward - f, o.left - l) <= gate_dist
               for o in perception.people for f, l in perception.humans):
            return
        self._explained = t
        shapes = [(round(o.forward, 2), round(o.left, 2)) for o in perception.people]
        seen = [(round(f, 2), round(l, 2)) for f, l in perception.humans]
        print(f"[follow] gate: shapes at {shapes} but YOLO people at {seen}", flush=True)


def check_ground_service(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("approach_schema_version") == 1
                and 0 <= time.time() - payload["published_at"] <= 2.0):
            return
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    raise RuntimeError("Ground observations unavailable: deploy the updated emotion_greeter main.py "
                       "and ground_safety.py, then start vision before the ground check-in")


def wait_fresh(reader, timeout, topic):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started > timeout:
            raise RuntimeError(f"no fresh {topic} sample; is its daemon running?")
        time.sleep(0.002)
    return reader.data


def other_drive_writers(ignore=()):
    # robot_follow.py and robot_base_mode.py match our own pattern, and the
    # `uv run ... python /tmp/robot_follow.py` parent also matches: ignore both.
    mine = {os.getpid(), os.getppid()}
    found = []
    for pattern in DRIVE_WRITER_PATTERNS:
        if pattern in ignore:
            continue
        result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
        found.extend(
            line.strip() for line in result.stdout.splitlines()
            if line.strip() and int(line.split()[0]) not in mine
        )
    return found


def perceive(points_base, t, cluster_cfg=ClusterConfig(), calibration=None, colors=None):
    """One camera.points frame -> Perception with every person-sized cluster."""
    calibration = calibration or Calibration()
    local = base_to_local(points_base, calibration.left_sign)
    local = level_floor(local, floor_line(local))
    keep = outside_self_mask(local, calibration.self_mask)
    if colors is not None:
        colors = np.asarray(colors)
        colors = colors[keep] if colors.shape == local.shape else None
    local = local[keep]
    people = tuple(PersonObservation(c.forward, c.left, hist=c.hist)
                   for c in find_people(local, cluster_cfg, colors=colors))
    return Perception(t, people, local)


def point_colors(data, n):
    names = getattr(getattr(data, "dtype", None), "names", None)
    names = names if names is not None else data.keys()
    if "colors" not in names:
        return None
    colors = np.asarray(data["colors"])
    if colors.dtype != np.uint8 or colors.ndim != 2 or colors.shape[1] != 3 or len(colors) < n:
        return None
    return colors[:n].copy()


def cloud(data):
    n = int(data["num_points"])
    points = np.asarray(data["points"])
    if points.ndim != 2 or points.shape[1] != 3 or not 0 <= n <= len(points):
        raise RuntimeError("invalid camera.points count or shape; expected num_points and Nx3 points")
    points = points[:n].copy()
    if not np.isfinite(points).all():
        raise RuntimeError("non-finite camera.points; check the depth daemon")
    return points


def ground_perception(data, t, wall_now, calibration):
    """Use capture time, not read time, to reject frozen or delayed depth."""
    try:
        captured = int(np.datetime64(data["timestamp"], "ns").astype(np.int64)) / 1e9
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    age = wall_now - captured
    if not 0 <= age <= approach_config().points_stale:
        return None
    local = without_self(base_to_local(cloud(data), calibration.left_sign), calibration.self_mask)
    if len(local) < 100:
        return None
    # The cloud's floor is a ramp (~0.10 m per metre on this robot). Unlevelled it
    # put ~11,000 "obstacle" points in the corridor and held every ground approach
    # BLOCKED. With too little floor to fit, the cloud stays as it is: still blocked.
    return Perception(t - age, (), level_floor(local, floor_line(local)))


def state_vector(data, field, size, topic):
    value = np.asarray(data[field], dtype=float).copy()
    if value.shape != (size,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid {topic}.{field}; expected {size} finite values")
    return value


def start_command_reader(commands):
    def read():
        for line in sys.stdin:
            commands.put(line)
        commands.put(None)  # EOF: the dashboard or SSH is gone

    threading.Thread(target=read, name="follow-stdin", daemon=True).start()


class PerceptionWorker:
    """Clusters camera.points off the control thread.

    Inline it took ~50 ms a frame, so the loop ticked at ~11 Hz with gaps beyond the
    base's 0.1 s drive.ctrl timeout: the base kept zeroing the twist mid-follow.
    """

    def __init__(self, points, process):
        self.points, self.process = points, process
        self._lock = threading.Lock()
        self._latest = self._error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="follow-perception", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def step(self):
        """Process one frame if there is one; False when there was nothing to read."""
        if not self.points.ready():
            return False
        started = time.monotonic()
        perception, candidates = self.process(self.points.data, started)
        result = (perception, (time.monotonic() - started) * 1000.0, candidates)
        with self._lock:
            self._latest = result
        return True

    def _run(self):
        try:
            while not self._stop.is_set():
                if not self.step():
                    self._stop.wait(0.002)
        except Exception as exc:  # surfaced on the control thread, which stops the robot
            self._error = exc

    def take(self):
        """Newest unread ``(perception, ms, candidates)``, or None. Re-raises a worker failure."""
        if self._error is not None:
            raise self._error
        with self._lock:
            latest, self._latest = self._latest, None
        return latest


def write_twist(writer, v, omega):
    with writer.buf() as frame:
        frame["twist"] = np.array([v, omega], dtype=np.float32)


def clamped_twist(out, cfg):
    """Re-clamp what the loop asks for; the runner never writes a negative speed
    even if the controller or supervisor were ever wrong (Global Constraints)."""
    return clamp(out.v, 0.0, cfg.v_max), clamp(out.omega, -cfg.omega_max, cfg.omega_max)


def write_led(writer, rgb):
    with writer.buf() as frame:
        frame["rgb"] = np.asarray(rgb, dtype=np.uint8)
        frame["brightness"] = np.int16(-1)
        frame["period_ms"] = np.uint16(0)


def run_check(reader_cls, seconds=3.0, calibration=None):
    """Read-only: print the person-sized clusters in each camera.points frame."""
    with reader_cls("camera.points", keeptime=False) as points:
        wait_fresh(points, 3.0, "camera.points")
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if points.ready():
                cal = calibration or Calibration()
                data = points.data
                xyz = cloud(data)
                local = base_to_local(xyz, cal.left_sign)
                local = level_floor(local, floor_line(local))   # same view the follow loop gets
                keep = outside_self_mask(local, cal.self_mask)
                colors = point_colors(data, len(xyz))
                people = find_people(local[keep], colors=None if colors is None else colors[keep])
                print(json.dumps({"clusters": [
                    {"forward": round(c.forward, 2), "left": round(c.left, 2), "points": c.points,
                     "top": round(c.top, 2), "depth": round(c.depth, 2), "width": round(c.width, 2),
                     "appearance": c.hist is not None}
                    for c in people
                ]}), flush=True)
            time.sleep(0.05)


def control_loop(args, cfg, readers, drive, led, wheel_diam, robot_width, calibration=None, speech=None):
    points, imu, drive_state = readers
    calibration = calibration or Calibration()
    loop = GroundApproachLoop(cfg) if args.ground_approach else FollowLoop(cfg, args.gap)
    commands = queue.Queue()
    if not args.no_heartbeat:
        start_command_reader(commands)
    now = time.monotonic()
    last_heartbeat = now
    command_stop = False
    last_imu = last_drive = now
    rpy = state_vector(imu.data, "rpy", 3, "imu.orientation")
    vel = state_vector(drive_state.data, "vel", 2, "drive.state")
    measured = wheel_twist(vel[list(calibration.wheel_order)], wheel_diam, robot_width, calibration.wheel_signs)
    last_status = last_csv = 0.0
    last_flush = now
    candidates = "[]"
    state, state_since = None, now

    gate = HumanGate(args.ground_alert_file, calibration.left_sign) if args.human_gate else None

    def process(data, t):
        if args.ground_approach:
            return ground_perception(data, t, time.time(), calibration), None
        xyz = cloud(data)
        perception = perceive(xyz, t, calibration=calibration, colors=point_colors(data, len(xyz)))
        if gate is not None:
            perception = replace(perception, humans=gate.humans(time.time()))
            gate.explain(t, perception, cfg.human_gate_dist)
        return perception, json.dumps([
            {"forward": round(p.forward, 3), "left": round(p.left, 3), "appearance": p.hist is not None}
            for p in perception.people
        ], separators=(",", ":"))

    log_path = args.log_dir / f"baymax_follow_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with log_path.open("w", newline="") as log_file, PerceptionWorker(points, process) as worker:
        log = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
        log.writeheader()
        print(f"[follow] logging to {log_path}", flush=True)
        while True:
            tick_started = t = time.monotonic()
            while not commands.empty():
                line = commands.get()
                if line is None:
                    command_stop = True
                    break
                command = parse_command(line, cfg)
                if command is None:
                    print(f"[follow] ignored command: {line.strip()[:80]}", flush=True)
                    continue
                last_heartbeat = t
                if command.kind == "stop":
                    command_stop = True
                elif command.kind == "gap":
                    print(f"[follow] gap {loop.set_gap(command.gap):.2f} m", flush=True)
            if args.no_heartbeat:
                last_heartbeat = t

            if imu.ready():
                rpy = state_vector(imu.data, "rpy", 3, "imu.orientation")
                last_imu = t
            if drive_state.ready():
                vel = state_vector(drive_state.data, "vel", 2, "drive.state")
                measured = wheel_twist(vel[list(calibration.wheel_order)], wheel_diam, robot_width, calibration.wheel_signs)
                last_drive = t
            if t - last_imu > STATE_TIMEOUT or t - last_drive > STATE_TIMEOUT:
                raise RuntimeError("stale imu.orientation or drive.state; stopping follow")
            perception = perception_ms = None
            latest = worker.take()
            if latest is not None:
                perception, perception_ms, frame_candidates = latest
                if frame_candidates is not None:
                    candidates = frame_candidates

            # Include perception work in freshness and heartbeat checks.
            t = time.monotonic()
            if t - last_imu > STATE_TIMEOUT or t - last_drive > STATE_TIMEOUT:
                raise RuntimeError("stale imu.orientation or drive.state; stopping follow")

            inputs = TickInputs(
                t=t, heartbeat_age=t - last_heartbeat,
                stop_requested=STOP_REQUESTED or command_stop,
                roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]),
                measured_v=measured[0], measured_omega=measured[1], perception=perception,
            )
            if args.ground_approach:
                wall_now = time.time()
                loop.observe(read_ground_target(args.ground_alert_file, wall_now, calibration))
                out = loop.tick(inputs, wall_now)
            else:
                out = loop.tick(inputs)
            if drive is not None:
                v, omega = clamped_twist(out, cfg)
                write_twist(drive, v, calibration.omega_sign * omega)
            if out.state != state:
                print(f"[follow] state {out.state}", flush=True)
                state, state_since = out.state, t
            led_state = {"WAITING": "SEARCHING", "APPROACHING": "FOLLOWING", "ARRIVED": "FOLLOWING"}.get(out.state, out.state)
            if led is not None:
                write_led(led, led_color(led_state, t - state_since))
            if args.ground_approach and loop.arrived and not out.exit:
                if speech is None and args.no_speech and not args.dry_run:
                    print(f"[follow] say {GROUND_LINE}", flush=True)
                    print("[follow] ground approach complete", flush=True)
                    return
                if speech is None:
                    print(f"[follow] dry run would say: {GROUND_LINE}", flush=True)
                    return
                if out.rule in ("ok", "settling"):
                    speech.start()
                if speech.done.is_set():
                    if speech.error:
                        raise RuntimeError(f"ground check-in speech failed: {speech.error}")
                    print("[follow] ground approach complete", flush=True)
                    return
            if t - last_status >= STATUS_PERIOD:
                print(status_line(out), flush=True)
                last_status = t
            if t - last_csv >= CSV_PERIOD:
                log.writerow({
                    "t": round(t, 3), "state": out.state, "rule": out.rule, "gap": out.gap,
                    "range": out.range, "bearing": out.bearing, "error": out.error,
                    "v_cmd": out.v_cmd, "omega_cmd": out.omega_cmd, "v": out.v, "omega": out.omega,
                    "measured_v": measured[0], "measured_omega": measured[1], "blocked": out.blocked,
                    "corridor_points": out.corridor_points, "track_age": out.track_age,
                    "people": "" if perception is None else len(perception.people),
                    "tick_ms": round((time.monotonic() - tick_started) * 1000, 3),
                    "perception_ms": "" if perception_ms is None else round(perception_ms, 3),
                    "association": out.association, "candidates": candidates,
                })
                last_csv = t
            if t - last_flush >= 1.0:
                log_file.flush()
                last_flush = t
            if out.exit:
                print(f"[follow] exit: {out.rule}", flush=True)
                return
            time.sleep(max(0.0, PERIOD - (time.monotonic() - tick_started)))


def run(args):
    calibration = load_calibration(args.calibration, required=args.preflight or not (args.check or args.dry_run))
    if calibration is None:
        print("[follow] UNCALIBRATED diagnostic: assumed left sign, no self mask; not motion-ready", flush=True)
    elif loop_config(args).v_max > calibration.motion_speed_limit:
        raise RuntimeError(f"requested speed exceeds calibrated limit {calibration.motion_speed_limit:.2f} m/s")

    from bbos import Config, Reader, Type, Writer

    if args.check:
        run_check(Reader, calibration=calibration)
        return
    cfg = loop_config(args)
    if calibration is not None:
        cfg = replace(cfg, self_mask=calibration.self_mask)
    drive_cfg = Config("drive")
    wheel_diam = float(drive_cfg.wheel_diam)
    robot_width = float(getattr(drive_cfg, "robot_width", cfg.robot_width))
    if not np.isfinite([wheel_diam, robot_width]).all() or min(wheel_diam, robot_width) <= 0:
        raise RuntimeError("invalid drive wheel diameter or robot width")
    low_battery_v = getattr(Config("base"), "low_battery_v", None)
    with ExitStack() as stack:
        speech = None
        if args.ground_approach and not (args.dry_run or args.preflight or args.no_speech):
            from ground_speech import prepare_speech
            speech = prepare_speech(Config, Writer, Type)
            stack.callback(speech.close)
        points = stack.enter_context(Reader("camera.points", keeptime=False))
        imu = stack.enter_context(Reader("imu.orientation", keeptime=False))
        drive_state = stack.enter_context(Reader("drive.state", keeptime=False))
        drive_status = stack.enter_context(Reader("drive.status", keeptime=False))

        rpy = state_vector(wait_fresh(imu, 2.0, "imu.orientation"), "rpy", 3, "imu.orientation")
        state_vector(wait_fresh(drive_state, 2.0, "drive.state"), "vel", 2, "drive.state")
        try:
            cloud(wait_fresh(points, 2.0, "camera.points"))
            points_fresh = True
        except RuntimeError:
            points_fresh = False
        try:
            voltage = float(wait_fresh(drive_status, 5.0, "drive.status")["voltage"])
        except RuntimeError:
            voltage = None
        if low_battery_v is None or not np.isfinite(float(low_battery_v)):
            raise RuntimeError("base.low_battery_v unavailable; cannot check battery")
        if voltage is None or not np.isfinite(voltage):
            raise RuntimeError("drive.status.voltage unavailable; cannot check battery")
        refusal = start_refusal(
            cfg, roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]), voltage=voltage,
            low_battery_v=None if low_battery_v is None else float(low_battery_v),
            drive_writers=other_drive_writers(args.ignore_writer), points_fresh=points_fresh,
        )
        if refusal:
            raise RuntimeError(f"refusing to start: {refusal}")

        # The slow battery topic may have taken several seconds. Refresh fast
        # inputs immediately before allowing writers, including the upright test.
        rpy = state_vector(wait_fresh(imu, 2.0, "imu.orientation"), "rpy", 3, "imu.orientation")
        state_vector(wait_fresh(drive_state, 2.0, "drive.state"), "vel", 2, "drive.state")
        cloud(wait_fresh(points, 2.0, "camera.points"))
        if max(abs(rpy[0]), abs(rpy[1])) > cfg.upright_deg:
            raise RuntimeError("refusing to start: robot is not upright")
        if args.preflight:
            if args.ground_approach:
                check_ground_service(args.ground_alert_file)
            print("[follow] PREFLIGHT OK: calibration and live inputs checked; no writers opened. Physical gates still required.", flush=True)
            return

        drive = None
        if args.ground_approach:
            check_ground_service(args.ground_alert_file)
        if not args.dry_run:
            drive = stack.enter_context(Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False))
        led = None if args.no_led else stack.enter_context(Writer("led.ctrl", Type("led_ctrl"), keeptime=False))
        mode = "dry run" if args.dry_run else "rotate only" if args.rotate_only else f"v_max {cfg.v_max:.2f} m/s"
        if args.ground_approach:
            print(f"[follow] follow active (ground approach, {mode}, {STANDOFF:.1f} m body standoff)", flush=True)
        else:
            print(f"[follow] follow active ({mode}, gap {args.gap:.2f} m) - stand in front of the robot", flush=True)
            if not cfg.odom_check:
                print("[follow] odometry check off", flush=True)
        try:
            control_loop(args, cfg, (points, imu, drive_state), drive, led, wheel_diam, robot_width, calibration, speech)
        finally:
            if drive is not None:
                for _ in range(6):
                    write_twist(drive, 0.0, 0.0)
                    time.sleep(PERIOD)
            if led is not None:
                write_led(led, (0, 0, 0))
            if drive is not None:
                print("[follow] stopped; zero twist sent", flush=True)


def main(argv=None):
    args = parse_args(argv)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_stop)
    if args.pid_file is not None:
        args.pid_file.write_text(f"{os.getpid()}\n")
    try:
        run(args)
    except RuntimeError as exc:
        print(f"[follow] {exc}", flush=True)
        raise SystemExit(1) from exc
    finally:
        if args.pid_file is not None:
            try:
                if args.pid_file.read_text().strip() == str(os.getpid()):
                    args.pid_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
