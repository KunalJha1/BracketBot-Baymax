"""First-motion test: the left arm plays the greeter's recorded wave. Runs ON THE ROBOT.

    uv run --no-sync --project ~/bbos python wave_test.py                 # dry run: prints the plan, moves nothing
    uv run --no-sync --project ~/bbos python wave_test.py --execute       # moves the LEFT arm only

Sequence: hold the arm where it is -> torque on -> ease to the wave's first frame (3 s) -> play the
wave (bbapps/greeter/movements/wave.json, motor turns, 15 ms frames) -> ease back to the starting
pose (3 s) -> torque off. The starting pose is the hanging pose, so torque off there drops nothing.

Ctrl-C once: ease back to the starting pose, then torque off. Ctrl-C twice: torque off now.
Refuses to run unless the IMU says the robot is upright. The right arm is never touched.
"""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
from bbos import Reader, Type, Writer

TICK = .015
EASE_S = 3.
UPRIGHT_DEG = 25.
WAVE = Path.home() / "bbapps/greeter/movements/wave.json"

interrupts = 0


def on_signal(*_):
    global interrupts
    interrupts += 1
    print(f"\n[wave] Ctrl-C ({interrupts}): {'easing back, then torque off' if interrupts == 1 else 'TORQUE OFF NOW'}", flush=True)


def fresh(reader, timeout=2.):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started > timeout:
            sys.exit("no fresh sample: is the daemon running?")
        time.sleep(.002)
    return reader.data


def smooth(s):
    s = min(max(s, 0.), 1.)
    return s * s * (3 - 2 * s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true", help="actually move the left arm")
    p.add_argument("--speed", type=float, default=1., help="wave playback speed (0.25-1)")
    a = p.parse_args()
    speed = float(np.clip(a.speed, .25, 1.))

    frames = json.loads(WAVE.read_text())
    wave = np.array([f["left"] for f in frames], dtype=np.float32)
    times = np.array([f["t"] for f in frames]) / speed

    with Reader("imu.orientation", keeptime=False) as r_imu, Reader("arm_left.state", keeptime=False) as r_state:
        rpy = np.asarray(fresh(r_imu)["rpy"], dtype=float)
        start = np.asarray(fresh(r_state)["pos"], dtype=np.float32).copy()
        # The wave never moves the lift (J0 span 0). Keep the lift wherever it is now instead of
        # driving it to the recording's height: after standing up it can sit tens of cm lower.
        wave[:, 0] = start[0]
        np.set_printoptions(precision=3, suppress=True)
        print(f"[wave] imu rpy (deg): {rpy}")
        print(f"[wave] left arm now (turns): {start}")
        print(f"[wave] wave first frame:     {wave[0]}")
        print(f"[wave] largest move to reach it: {np.abs(wave[0] - start).max():.3f} turns over {EASE_S:.0f} s")
        print(f"[wave] wave: {len(wave)} frames, {times[-1]:.1f} s at speed {speed}; joint span {np.ptp(wave, 0)}")
        upright = abs(rpy[0]) < UPRIGHT_DEG and abs(rpy[1]) < UPRIGHT_DEG
        if not upright:
            sys.exit(f"[wave] robot is not upright (|roll| and |pitch| must be < {UPRIGHT_DEG} deg). Not moving.")
        if not a.execute:
            print("[wave] dry run only. Add --execute to move the left arm.")
            return

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        with Writer("arm_left.ctrl", Type("arm_ctrl"), keeptime=False) as w_ctrl, \
             Writer("arm_left.torque", Type("arm_torque"), keeptime=False) as w_torque:

            def command(pos):
                with w_ctrl.buf() as b:
                    b["pos"][:] = pos
                    b["vel"][:] = 0
                    b["tau"][:] = 0
                    b["alpha"] = 0.

            def torque(on):
                with w_torque.buf() as b:
                    b["enable"][:] = on
                    b["tau_mode"][:] = False
                    b["compliance_mode"] = False

            def ease(a_pos, b_pos, seconds):
                t0 = time.monotonic()
                while True:
                    s = (time.monotonic() - t0) / seconds
                    command(a_pos + smooth(s) * (b_pos - a_pos))
                    if s >= 1 or interrupts > 1:
                        return
                    time.sleep(TICK)

            try:
                # Command the pose the arm is already in BEFORE torque comes on, so nothing jumps.
                for _ in range(10):
                    command(start)
                    time.sleep(TICK)
                torque(True)
                print("[wave] torque on, easing to the wave start", flush=True)
                ease(start, wave[0], EASE_S)
                last = wave[0]
                if not interrupts:
                    print("[wave] waving", flush=True)
                    t0 = time.monotonic()
                    for i in range(len(wave)):
                        if interrupts:
                            break
                        time.sleep(max(0., t0 + times[i] - time.monotonic()))
                        command(wave[i])
                        last = wave[i]
                if interrupts < 2:
                    print("[wave] easing back to the starting pose", flush=True)
                    ease(last, start, EASE_S)
                    time.sleep(.3)
            finally:
                torque(False)
                print("[wave] torque off", flush=True)


if __name__ == "__main__":
    main()
