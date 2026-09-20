"""Safely play a recorded BracketBot greeter gesture. Runs ON THE ROBOT.

The greeter recordings contain both arms, even when only one arm actually
moves. This runner detects the active arm(s), leaves inactive arms untouched,
preserves each active arm's current lift height, eases into and out of the
recording, and disables torque when finished.

The default is a non-moving safety check. Pass ``--execute`` to move.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from bbos import Reader, Type, Writer

TICK = 0.015
# Ease time scales with distance (smoothstep peaks at 1.5x the mean speed), so
# a short entry is quick and the longest allowed entry is as slow as before.
EASE_PEAK_TURNS_PER_S = 0.30
MIN_EASE_S = 0.5
MAX_EASE_S = 3.0
IDLE_TURNS = 0.01
IDLE_MARGIN_S = 0.15
RETURNED_TURNS = 0.08
UPRIGHT_DEG = 25.0
DOF = 8
ACTIVE_SPAN_TURNS = 0.04
MAX_ENTRY_TURNS = 0.50
SIDES = ("left", "right")

interrupts = 0


def on_signal(*_):
    global interrupts
    interrupts += 1
    action = "returning to start, then torque off" if interrupts == 1 else "TORQUE OFF NOW"
    print(f"\n[gesture] stop requested ({interrupts}): {action}", flush=True)


def fresh(reader, timeout=2.0):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started > timeout:
            sys.exit("[gesture] no fresh sample: is the daemon running?")
        time.sleep(0.002)
    return reader.data


def smooth(value):
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def load_trajectory(path):
    """Return normalized timestamps and validated motor-turn trajectories."""
    try:
        frames = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read trajectory {path}: {exc}") from exc

    # Mimic recordings wrap the same frame schema with name/save metadata.
    if isinstance(frames, dict):
        frames = frames.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("trajectory must contain at least two frames")

    try:
        times = np.asarray([frame["t"] for frame in frames], dtype=np.float64)
        poses = {
            side: np.asarray([frame[side] for frame in frames], dtype=np.float32)
            for side in SIDES
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("each frame must contain numeric 't', 'left', and 'right' values") from exc

    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("trajectory timestamps must be finite and strictly increasing")
    for side, trajectory in poses.items():
        if trajectory.shape != (len(frames), DOF):
            raise ValueError(
                f"{side}-arm poses must have shape ({len(frames)}, {DOF}), got {trajectory.shape}"
            )
        if not np.isfinite(trajectory).all():
            raise ValueError(f"{side}-arm trajectory contains non-finite values")

    return times - times[0], poses


def trim_idle(times, poses):
    """Drop the motionless lead-in/out; keep holds that are part of the gesture."""
    joints = np.concatenate([poses[side][:, 1:7] for side in SIDES], axis=1)
    moved = np.flatnonzero(np.abs(joints - joints[0]).max(axis=1) > IDLE_TURNS)
    if not len(moved):
        return times, poses
    first = int(np.searchsorted(times, times[moved[0]] - IDLE_MARGIN_S))
    last = len(times) - 1
    # A final hold away from the first pose (namaste) is intentional.
    if np.abs(joints[-1] - joints[0]).max() <= RETURNED_TURNS:
        settling = np.flatnonzero(np.abs(joints - joints[-1]).max(axis=1) > IDLE_TURNS)
        last = min(last, int(np.searchsorted(times, times[settling[-1]] + IDLE_MARGIN_S)))
    if last - first < 1:
        return times, poses
    keep = slice(first, last + 1)
    return times[keep] - times[first], {side: poses[side][keep] for side in SIDES}


def ease_seconds(from_poses, to_poses):
    distance = max(
        float(np.max(np.abs(to_poses[side] - from_poses[side]))) for side in to_poses
    )
    return min(max(1.5 * distance / EASE_PEAK_TURNS_PER_S, MIN_EASE_S), MAX_EASE_S)


def active_sides(poses):
    """Select arms with intentional motion, excluding lift and gripper noise."""
    return tuple(
        side
        for side in SIDES
        if float(np.ptp(poses[side][:, 1:7], axis=0).max()) >= ACTIVE_SPAN_TURNS
    )


def write_pid_file(path):
    if path is None:
        return
    path.write_text(f"{os.getpid()}\n")


def remove_pid_file(path):
    if path is None:
        return
    try:
        if path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Safely play a recorded arm gesture")
    parser.add_argument("trajectory", type=Path, help="greeter movement JSON")
    parser.add_argument("--name", help="display name; defaults to the JSON filename")
    parser.add_argument("--execute", action="store_true", help="actually move the active arm(s)")
    parser.add_argument("--speed", type=float, default=0.6, help="playback speed, 0.25-1 (default: 0.6)")
    parser.add_argument("--pid-file", type=Path, help="write the process ID here for a remote stop control")
    args = parser.parse_args()

    if not 0.25 <= args.speed <= 1.0:
        parser.error("--speed must be between 0.25 and 1")
    name = args.name or args.trajectory.stem

    try:
        times, poses = load_trajectory(args.trajectory.expanduser())
    except ValueError as exc:
        sys.exit(f"[gesture] {exc}")

    sides = active_sides(poses)
    if not sides:
        sys.exit("[gesture] no intentional arm movement found in this recording")
    times, poses = trim_idle(times, poses)
    playback_times = times / args.speed

    with ExitStack() as stack:
        r_imu = stack.enter_context(Reader("imu.orientation", keeptime=False))
        readers = {
            side: stack.enter_context(Reader(f"arm_{side}.state", keeptime=False))
            for side in sides
        }
        rpy = np.asarray(fresh(r_imu)["rpy"], dtype=float)
        starts = {
            side: np.asarray(fresh(readers[side])["pos"], dtype=np.float32).copy()
            for side in sides
        }

        # Lift position depends on how the robot was parked. A gesture should
        # move shoulder/wrist joints, not unexpectedly raise the full arm.
        for side in sides:
            poses[side][:, 0] = starts[side][0]

        np.set_printoptions(precision=3, suppress=True)
        print(f"[gesture] {name}: active arm(s): {', '.join(sides)}")
        print(f"[gesture] imu rpy (deg): {rpy}")
        print(f"[gesture] {len(times)} frames, {playback_times[-1]:.1f} s at x{args.speed:.2f}")
        entry_moves = {}
        for side in sides:
            entry_moves[side] = float(np.abs(poses[side][0] - starts[side]).max())
            print(
                f"[gesture] {side}: start {starts[side]}; first {poses[side][0]}; "
                f"entry {entry_moves[side]:.3f} turns"
            )

        problems = []
        if abs(rpy[0]) >= UPRIGHT_DEG or abs(rpy[1]) >= UPRIGHT_DEG:
            problems.append(
                f"robot is not upright; |roll| and |pitch| must be < {UPRIGHT_DEG:.0f} deg"
            )
        for side, move in entry_moves.items():
            if move > MAX_ENTRY_TURNS:
                problems.append(
                    f"{side} entry move is {move:.3f} turns, over the {MAX_ENTRY_TURNS:.2f} limit"
                )
        if problems:
            sys.exit("[gesture] NOT SAFE TO RUN:\n  " + "\n  ".join(problems))
        if not args.execute:
            print("[gesture] checks passed; dry run only. Add --execute to move.")
            return

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, on_signal)

        controls = {
            side: stack.enter_context(Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False))
            for side in sides
        }
        torque_writers = {
            side: stack.enter_context(
                Writer(f"arm_{side}.torque", Type("arm_torque"), keeptime=False)
            )
            for side in sides
        }

        def command(targets):
            for side in sides:
                with controls[side].buf() as buf:
                    buf["pos"][:] = targets[side]
                    buf["vel"][:] = 0
                    buf["tau"][:] = 0
                    buf["alpha"] = 0.0

        def torque(on):
            for side in sides:
                with torque_writers[side].buf() as buf:
                    buf["enable"][:] = on
                    buf["tau_mode"][:] = False
                    buf["compliance_mode"] = False

        def ease(from_poses, to_poses):
            seconds = ease_seconds(from_poses, to_poses)
            started = time.monotonic()
            last = from_poses
            while True:
                fraction = smooth((time.monotonic() - started) / seconds)
                last = {
                    side: from_poses[side] + fraction * (to_poses[side] - from_poses[side])
                    for side in sides
                }
                command(last)
                if fraction >= 1.0 or interrupts > 1:
                    return last
                time.sleep(TICK)

        last = dict(starts)
        write_pid_file(args.pid_file)
        try:
            for _ in range(10):
                command(starts)
                time.sleep(TICK)
            torque(True)
            print(f"[gesture] torque on; easing into {name}", flush=True)
            first = {side: poses[side][0] for side in sides}
            last = ease(starts, first)

            if not interrupts:
                print(f"[gesture] playing {name}", flush=True)
                started = time.monotonic()
                index = 0
                while index < len(times) - 1 and not interrupts:
                    elapsed = time.monotonic() - started
                    index = min(int(np.searchsorted(playback_times, elapsed)), len(times) - 1)
                    last = {side: poses[side][index] for side in sides}
                    command(last)
                    time.sleep(TICK)

            if interrupts < 2:
                print("[gesture] returning to the starting pose", flush=True)
                ease(last, starts)
                time.sleep(0.3)
        finally:
            torque(False)
            remove_pid_file(args.pid_file)
            print("[gesture] torque off", flush=True)


if __name__ == "__main__":
    main()
