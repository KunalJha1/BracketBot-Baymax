"""Safely play BracketBot's recorded right-arm handshake. Runs ON THE ROBOT.

    uv run --no-sync --project ~/bbos python handshake_test.py handshake.json
    uv run --no-sync --project ~/bbos python handshake_test.py handshake.json --execute

The script holds the arm's current pose before enabling torque, preserves the
current lift height (J0), eases into the recorded path, plays the handshake,
eases back to the starting pose, and then disables torque. The left arm is
never commanded.

Ctrl-C once returns to the starting pose before torque is disabled. Ctrl-C a
second time disables torque immediately. The motion is refused unless the IMU
reports that the robot is upright.
"""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
from bbos import Reader, Type, Writer

TICK = 0.015
EASE_S = 3.0
UPRIGHT_DEG = 25.0
DOF = 8

interrupts = 0


def on_signal(*_):
    global interrupts
    interrupts += 1
    action = "easing back, then torque off" if interrupts == 1 else "TORQUE OFF NOW"
    print(f"\n[handshake] Ctrl-C ({interrupts}): {action}", flush=True)


def fresh(reader, timeout=2.0):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started > timeout:
            sys.exit("[handshake] no fresh sample: is the daemon running?")
        time.sleep(0.002)
    return reader.data


def smooth(value):
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def load_trajectory(path):
    """Load and validate a greeter-format right-arm trajectory."""
    try:
        frames = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read trajectory {path}: {exc}") from exc

    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("trajectory must contain at least two frames")

    try:
        times = np.asarray([frame["t"] for frame in frames], dtype=np.float64)
        poses = np.asarray([frame["right"] for frame in frames], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("each frame must contain numeric 't' and 'right' values") from exc

    if poses.shape != (len(frames), DOF):
        raise ValueError(f"right-arm poses must have shape ({len(frames)}, {DOF}), got {poses.shape}")
    if not np.isfinite(times).all() or not np.isfinite(poses).all():
        raise ValueError("trajectory contains non-finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")

    # Playback begins immediately even if the recording's first timestamp was
    # captured a fraction of a second after recording started.
    return times - times[0], poses


def main():
    parser = argparse.ArgumentParser(description="Safely play the right-arm handshake")
    parser.add_argument("trajectory", type=Path, help="greeter handshake JSON")
    parser.add_argument("--execute", action="store_true", help="actually move the right arm")
    parser.add_argument("--speed", type=float, default=0.6, help="playback speed, 0.25-1 (default: 0.6)")
    args = parser.parse_args()
    speed = float(np.clip(args.speed, 0.25, 1.0))

    try:
        times, handshake = load_trajectory(args.trajectory.expanduser())
    except ValueError as exc:
        sys.exit(f"[handshake] {exc}")

    with Reader("imu.orientation", keeptime=False) as r_imu, \
         Reader("arm_right.state", keeptime=False) as r_state:
        rpy = np.asarray(fresh(r_imu)["rpy"], dtype=float)
        start = np.asarray(fresh(r_state)["pos"], dtype=np.float32).copy()

        # The recording does not intentionally move the vertical lift. Keeping
        # the live J0 position avoids moving the entire arm to the height at
        # which the gesture happened to be recorded.
        handshake[:, 0] = start[0]
        playback_times = times / speed

        np.set_printoptions(precision=3, suppress=True)
        print(f"[handshake] imu rpy (deg):       {rpy}")
        print(f"[handshake] right arm now:       {start}")
        print(f"[handshake] first gesture pose:  {handshake[0]}")
        print(
            f"[handshake] {len(handshake)} frames, {playback_times[-1]:.1f} s at x{speed:.2f}; "
            f"joint span {np.ptp(handshake, axis=0)}"
        )
        print(
            f"[handshake] largest move to enter: "
            f"{np.abs(handshake[0] - start).max():.3f} turns over {EASE_S:.0f} s"
        )

        upright = abs(rpy[0]) < UPRIGHT_DEG and abs(rpy[1]) < UPRIGHT_DEG
        if not upright:
            sys.exit(
                f"[handshake] robot is not upright; |roll| and |pitch| must be "
                f"< {UPRIGHT_DEG:.0f} deg. Not moving."
            )
        if not args.execute:
            print("[handshake] checks passed; dry run only. Add --execute to move the right arm.")
            return

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        with Writer("arm_right.ctrl", Type("arm_ctrl"), keeptime=False) as w_ctrl, \
             Writer("arm_right.torque", Type("arm_torque"), keeptime=False) as w_torque:

            def command(pos):
                with w_ctrl.buf() as buf:
                    buf["pos"][:] = pos
                    buf["vel"][:] = 0
                    buf["tau"][:] = 0
                    buf["alpha"] = 0.0

            def torque(on):
                with w_torque.buf() as buf:
                    buf["enable"][:] = on
                    buf["tau_mode"][:] = False
                    buf["compliance_mode"] = False

            def ease(from_pos, to_pos, seconds):
                started = time.monotonic()
                last = from_pos
                while True:
                    fraction = smooth((time.monotonic() - started) / seconds)
                    last = from_pos + fraction * (to_pos - from_pos)
                    command(last)
                    if fraction >= 1.0 or interrupts > 1:
                        return last
                    time.sleep(TICK)

            last = start
            try:
                # Establish the current-position command before enabling torque
                # so the arm cannot jump toward the first recorded frame.
                for _ in range(10):
                    command(start)
                    time.sleep(TICK)
                torque(True)

                print("[handshake] torque on; easing into the gesture", flush=True)
                last = ease(start, handshake[0], EASE_S)

                if not interrupts:
                    print("[handshake] offering hand and shaking", flush=True)
                    started = time.monotonic()
                    index = 0
                    while index < len(handshake) - 1 and not interrupts:
                        elapsed = time.monotonic() - started
                        index = min(int(np.searchsorted(playback_times, elapsed)), len(handshake) - 1)
                        last = handshake[index]
                        command(last)
                        time.sleep(TICK)

                if interrupts < 2:
                    print("[handshake] easing back to the starting pose", flush=True)
                    ease(last, start, EASE_S)
                    time.sleep(0.3)
            finally:
                torque(False)
                print("[handshake] torque off", flush=True)


if __name__ == "__main__":
    main()
