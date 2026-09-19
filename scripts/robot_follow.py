# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "bbos",
#   "bbai",
#   "numpy",
#   "opencv-python",
#   "pycuda",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# bbai = { path = "/home/bracketbot/bbai", editable = true }
# ///
"""Person-follow runner for BracketBot. Copied to /tmp by robot_dashboard.py.

The dependency block mirrors bbapps/greeter/main.py, the app already proven to
run TensorRT through PyCUDA on this robot (bbai brings the TensorRT bindings).
While running it is the only writer of drive.ctrl and led.ctrl. Every decision
lives in follow_core.FollowLoop; this file only moves data between BBOS topics,
the pose engine, and the loop.

    uv run --script /tmp/robot_follow.py --pid-file /tmp/f.pid     # from the dashboard
    uv run --script /tmp/robot_follow.py --check                   # gate G1
    uv run --script /tmp/robot_follow.py --dry-run --no-heartbeat  # gate G2
    uv run --script /tmp/robot_follow.py --rotate-only ...         # gate G3
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import replace
import json
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
    FollowConfig, FollowLoop, Perception, PersonObservation, TickInputs, led_color,
    parse_command, start_refusal, status_line, timestamp_to_seconds, wheel_twist,
)
from follow_perception import (
    PoseEngine, base_to_local, hand_raised, mask_to_image_pixels, person_position,
    torso_histogram, torso_rect,
)

PERIOD = 0.02  # 50 Hz control loop; drive.ctrl times out after 0.1 s
STATUS_PERIOD = 0.2
CSV_PERIOD = 0.05
PAIR_TOLERANCE = 0.05  # max seconds between the RGB frame and the point cloud it is paired with
# drive.state.vel -> (left, right) turns/s, forward-positive. Verified at gates G3/G4a.
WHEEL_ORDER = (0, 1)
WHEEL_SIGNS = (1.0, 1.0)
DRIVE_WRITER_PATTERNS = (
    "greeter/main.py", "nav/main.py", "bbapps/teleop.py", "quest_teleop/main.py",
    "leader_follower_teleop.py", "live_inference.py",
)
CSV_FIELDS = (
    "t", "state", "rule", "gap", "range", "bearing", "error", "v_cmd", "omega_cmd", "v", "omega",
    "measured_v", "measured_omega", "blocked", "corridor_points", "track_age", "people",
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
    parser.add_argument("--engine", type=Path, default=Path.home() / ".cache/baymax/yolo11n-pose.engine")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--dry-run", action="store_true", help="compute and log; never open drive.ctrl")
    parser.add_argument("--rotate-only", action="store_true", help="forward speed held at 0 (gate G3)")
    parser.add_argument("--no-heartbeat", action="store_true", help="only with --dry-run: no dashboard needed")
    parser.add_argument("--check", action="store_true", help="gate G1: time the engine on a live frame, then exit")
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
    return replace(FollowConfig(), v_max=0.0 if args.rotate_only else args.v_max)


def field(data, name):
    try:
        return data[name]
    except (KeyError, ValueError, IndexError):
        return None


def wait_fresh(reader, timeout, topic):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started > timeout:
            raise RuntimeError(f"no fresh {topic} sample; is its daemon running?")
        time.sleep(0.002)
    return reader.data


def left_eye(image):
    # A side-by-side stereo image is more than twice as wide as it is tall.
    return image[:, : image.shape[1] // 2] if image.shape[1] > 2.5 * image.shape[0] else image


def stamp(data, arrival):
    value = field(data, "timestamp")
    return arrival if value is None else timestamp_to_seconds(np.asarray(value).item())


def other_drive_writers():
    found = []
    for pattern in DRIVE_WRITER_PATTERNS:
        result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
        found.extend(
            line.strip() for line in result.stdout.splitlines()
            if line.strip() and int(line.split()[0]) != os.getpid()
        )
    return found


def perceive(engine, image, points_base, pixels, t):
    """Pose detections + the point cloud -> located people. ``pixels``: each point's (u, v) in ``image``."""
    detections = engine.infer(image)
    local = base_to_local(points_base)
    people = []
    for det in detections:
        rect = torso_rect(det)
        position = person_position(local, pixels, rect)
        if position is not None:
            people.append(PersonObservation(
                position[0], position[1], det.score, hand_raised(det), torso_histogram(image, rect)
            ))
    return Perception(t, tuple(people), local)


def make_pixel_mapper(depth_shape):
    """Where each camera.points point lands in the detection image: (points_base, mask, image_shape) -> (N, 2)."""
    return lambda points_base, mask, image_shape: mask_to_image_pixels(mask, depth_shape, image_shape)


def start_command_reader(commands):
    def read():
        for line in sys.stdin:
            commands.put(line)
        commands.put(None)  # EOF: the dashboard or SSH is gone

    threading.Thread(target=read, name="follow-stdin", daemon=True).start()


def write_twist(writer, v, omega):
    with writer.buf() as frame:
        frame["twist"] = np.array([v, omega], dtype=np.float32)


def write_led(writer, rgb):
    with writer.buf() as frame:
        frame["rgb"] = np.asarray(rgb, dtype=np.uint8)
        frame["brightness"] = np.int16(-1)
        frame["period_ms"] = np.uint16(0)


def run_check(engine, reader_cls, image_topic):
    with reader_cls(image_topic, keeptime=False) as source:
        image = left_eye(np.asarray(wait_fresh(source, 3.0, image_topic)["rgb"]).copy())
    timings, detections = [], []
    for _ in range(20):
        started = time.perf_counter()
        detections = engine.infer(image)
        timings.append((time.perf_counter() - started) * 1000)
    best = detections[0] if detections else None
    print(json.dumps({
        "image_shape": list(image.shape),
        "latency_ms_p50": round(float(np.median(timings[5:])), 1),
        "latency_ms_max": round(float(np.max(timings[5:])), 1),
        "people": len(detections),
        "best": None if best is None else {
            "box": [round(float(v)) for v in best.box], "score": round(best.score, 3),
            "hand_raised": hand_raised(best),
        },
    }), flush=True)


def control_loop(args, cfg, engine, readers, drive, led, to_pixels, wheel_diam, robot_width):
    camera, points, imu, drive_state = readers
    loop = FollowLoop(cfg, args.gap)
    commands = queue.Queue()
    if not args.no_heartbeat:
        start_command_reader(commands)
    now = time.monotonic()
    last_heartbeat = now
    command_stop = False
    rpy = np.zeros(3)
    measured = (0.0, 0.0)
    rgb_frame = None  # (arrival, stamp, image)
    last_status = last_csv = 0.0
    state, state_since = None, now
    log_path = args.log_dir / f"baymax_follow_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with log_path.open("w", newline="") as log_file:
        log = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
        log.writeheader()
        print(f"[follow] logging to {log_path}", flush=True)
        while True:
            t = time.monotonic()
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
                rpy = np.asarray(imu.data["rpy"], dtype=float)
            if drive_state.ready():
                vel = np.asarray(drive_state.data["vel"], dtype=float)
                measured = wheel_twist(vel[list(WHEEL_ORDER)], wheel_diam, robot_width, WHEEL_SIGNS)
            if camera.ready():
                data = camera.data
                rgb_frame = (t, stamp(data, t), left_eye(np.asarray(data["rgb"]).copy()))
            perception = None
            if points.ready():
                data = points.data
                n = int(data["num_points"])
                cloud_stamp = stamp(data, t)
                if rgb_frame is not None and abs(rgb_frame[1] - cloud_stamp) <= PAIR_TOLERANCE \
                        and t - rgb_frame[0] <= PAIR_TOLERANCE:
                    image, points_base = rgb_frame[2], np.asarray(data["points"][:n])
                    pixels = to_pixels(points_base, np.asarray(data["mask"][:n]), image.shape)
                    perception = perceive(engine, image, points_base, pixels, t)

            out = loop.tick(TickInputs(
                t=t, heartbeat_age=t - last_heartbeat,
                stop_requested=STOP_REQUESTED or command_stop,
                roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]),
                measured_v=measured[0], measured_omega=measured[1], perception=perception,
            ))
            if drive is not None:
                write_twist(drive, out.v, out.omega)
            if out.state != state:
                print(f"[follow] state {out.state}", flush=True)
                state, state_since = out.state, t
            write_led(led, led_color(out.state, t - state_since))
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
                })
                last_csv = t
            if out.exit:
                print(f"[follow] exit: {out.rule}", flush=True)
                return
            time.sleep(max(0.0, PERIOD - (time.monotonic() - t)))


def run(args):
    from bbos import Config, Reader, Type, Writer

    engine = PoseEngine(args.engine)
    try:
        image_topic = "camera.rect"
        if args.check:
            run_check(engine, Reader, image_topic)
            return
        cfg = loop_config(args)
        drive_cfg = Config("drive")
        wheel_diam = float(drive_cfg.wheel_diam)
        robot_width = float(getattr(drive_cfg, "robot_width", cfg.robot_width))
        low_battery_v = getattr(Config("base"), "low_battery_v", None)
        with ExitStack() as stack:
            camera = stack.enter_context(Reader(image_topic, keeptime=False))
            points = stack.enter_context(Reader("camera.points", keeptime=False))
            depth = stack.enter_context(Reader("camera.depth", keeptime=False))
            imu = stack.enter_context(Reader("imu.orientation", keeptime=False))
            drive_state = stack.enter_context(Reader("drive.state", keeptime=False))
            drive_status = stack.enter_context(Reader("drive.status", keeptime=False))

            depth_shape = np.asarray(wait_fresh(depth, 3.0, "camera.depth")["depth"]).shape
            rpy = np.asarray(wait_fresh(imu, 2.0, "imu.orientation")["rpy"], dtype=float)
            try:
                wait_fresh(points, 2.0, "camera.points")
                points_fresh = True
            except RuntimeError:
                points_fresh = False
            try:
                voltage = float(wait_fresh(drive_status, 2.0, "drive.status")["voltage"])
            except RuntimeError:
                voltage = None
            if low_battery_v is None:
                print("[follow] warning: base.low_battery_v unknown; battery not checked", flush=True)
            refusal = start_refusal(
                cfg, roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]), voltage=voltage,
                low_battery_v=None if low_battery_v is None else float(low_battery_v),
                drive_writers=other_drive_writers(), points_fresh=points_fresh,
            )
            if refusal:
                raise RuntimeError(f"refusing to start: {refusal}")
            engine.infer(np.zeros((depth_shape[0], depth_shape[1], 3), dtype=np.uint8))  # warm-up

            drive = None
            if not args.dry_run:
                drive = stack.enter_context(Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False))
            led = stack.enter_context(Writer("led.ctrl", Type("led_ctrl"), keeptime=False))
            mode = "dry run" if args.dry_run else "rotate only" if args.rotate_only else f"v_max {cfg.v_max:.2f} m/s"
            print(f"[follow] follow active ({mode}, gap {args.gap:.2f} m) - raise a hand", flush=True)
            try:
                control_loop(args, cfg, engine, (camera, points, imu, drive_state),
                             drive, led, make_pixel_mapper(depth_shape), wheel_diam, robot_width)
            finally:
                if drive is not None:
                    for _ in range(6):
                        write_twist(drive, 0.0, 0.0)
                        time.sleep(PERIOD)
                write_led(led, (0, 0, 0))
                print("[follow] stopped; zero twist sent", flush=True)
    finally:
        engine.close()


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
