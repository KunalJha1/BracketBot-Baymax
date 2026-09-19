"""Replay the sim's fox-fold arm motion on the real robot, in the air. Runs ON THE ROBOT.

    uv run --no-sync --project ~/bbos python fold_replay.py fox-seed7.npz                  # dry run: checks only
    uv run --no-sync --project ~/bbos python fold_replay.py fox-seed7.npz --execute        # both arms move
    ... --speed 0.5 (default) --until 20 (stop after 20 s of the recording)

The .npz comes from scripts/export_fold_trajectory.py: URDF joint set points on the 15 ms tick. The sim's
joints use the real URDF convention (checked by FK), so each arm converts them with its own
Config(...).urdf2q. Jaws: closed -> -0.15 rad, open -> 0.8 rad (quest_teleop's values), rate-limited.

Sequence: hold both arms where they are -> torque on -> ease into the recording's first pose (lift at
<= 5 cm/s, joints at <= 0.5 rad/s) -> play at --speed -> ease back to the starting pose -> torque off.
The hands go where the sim's paper was, about 0.74 m above the floor in front of the robot:
KEEP THE SPACE IN FRONT OF THE ROBOT EMPTY.

Ctrl-C once: ease back to the starting pose, then torque off. Ctrl-C twice: torque off now.
"""

import argparse
from pathlib import Path
import signal
import sys
import time

import numpy as np
from bbos import Config, Reader, Type, Writer

TICK = .015
UPRIGHT_DEG = 25.
LIFT_SPEED, JOINT_SPEED, GRIP_SPEED = .05, .5, 2.  # m/s, rad/s, rad/s (URDF)
GRIP_CLOSED, GRIP_OPEN = -.15, .8
CAL_MARGIN = .02  # fraction of each joint's calibrated span a target may sit outside it
SIDES = ("left", "right")

interrupts = 0


def on_signal(*_):
    global interrupts
    interrupts += 1
    print(f"\n[fold] Ctrl-C ({interrupts}): {'easing back, then torque off' if interrupts == 1 else 'TORQUE OFF NOW'}", flush=True)


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


def jaw_track(closed, dt):
    """Sim open/closed flags -> URDF jaw angle moving at GRIP_SPEED."""
    out = np.empty(len(closed))
    angle = GRIP_CLOSED if closed[0] else GRIP_OPEN
    for i, c in enumerate(closed):
        goal = GRIP_CLOSED if c else GRIP_OPEN
        angle += np.clip(goal - angle, -GRIP_SPEED * dt, GRIP_SPEED * dt)
        out[i] = angle
    return out


def ease_seconds(cfg, a, b):
    """Time to move between two motor-turn poses at the ease speed limits (at least 3 s)."""
    d = np.abs(cfg.q2urdf(b.astype(np.float64).copy()) - cfg.q2urdf(a.astype(np.float64).copy()))
    return max(3., d[0] / LIFT_SPEED, d[1:7].max() / JOINT_SPEED, d[7] / GRIP_SPEED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory")
    p.add_argument("--execute", action="store_true", help="actually move both arms")
    p.add_argument("--speed", type=float, default=.5, help="playback speed, 0.1-1")
    p.add_argument("--until", type=float, default=None, help="stop after this many seconds of the recording")
    a = p.parse_args()
    speed = float(np.clip(a.speed, .1, 1.))

    z = np.load(Path(a.trajectory).expanduser())
    t, q, grip = z["t"].astype(np.float64), z["q"].astype(np.float64), z["grip"]
    if a.until is not None:
        keep = t <= a.until
        t, q, grip = t[keep], q[keep], grip[keep]
    cfgs = {s: Config(f"arm_{s}") for s in SIDES}
    cal = {s: {k: np.array(v) for k, v in __import__("json").loads(
        (Path(__import__("bbos").__file__).parent / "daemons" / f"arm_{s}" / "ranges.calibration.json").read_text()).items()} for s in SIDES}

    # URDF -> motor turns, per arm, per sample.
    turns = {}
    for k, s in enumerate(SIDES):
        urdf = np.zeros((len(t), 8))
        urdf[:, :7] = q[:, 7 * k:7 * k + 7]
        urdf[:, 7] = jaw_track(grip[:, k] > .5, TICK)
        turns[s] = np.stack([cfgs[s].urdf2q(row.copy()) for row in urdf]).astype(np.float32)

    np.set_printoptions(precision=3, suppress=True)
    problems = []
    for s in SIDES:
        lo, hi = np.minimum(cal[s]["cal_min"], cal[s]["cal_max"]), np.maximum(cal[s]["cal_min"], cal[s]["cal_max"])
        margin = CAL_MARGIN * (hi - lo)
        tr = turns[s][:, :7]
        below, above = (lo[:7] - margin[:7]) - tr.min(0), tr.max(0) - (hi[:7] + margin[:7])
        for j in range(7):
            if below[j] > 0 or above[j] > 0:
                problems.append(f"{s} J{j}: recording {tr[:, j].min():.3f}..{tr[:, j].max():.3f} turns, calibrated {lo[j]:.3f}..{hi[j]:.3f}")
        rate = np.abs(np.diff(np.stack([cfgs[s].q2urdf(r.astype(np.float64).copy()) for r in turns[s]]), axis=0)).max(0) / TICK * speed
        print(f"[fold] {s}: peak speed at x{speed}: lift {rate[0]:.3f} m/s, joints {rate[1:7].max():.2f} rad/s, jaw {rate[7]:.2f} rad/s")

    with Reader("imu.orientation", keeptime=False) as r_imu, \
         Reader("arm_left.state", keeptime=False) as r_l, Reader("arm_right.state", keeptime=False) as r_r:
        readers = {"left": r_l, "right": r_r}
        rpy = np.asarray(fresh(r_imu)["rpy"], dtype=float)
        start = {s: np.asarray(fresh(readers[s])["pos"], dtype=np.float32).copy() for s in SIDES}
        ease_in = max(ease_seconds(cfgs[s], start[s], turns[s][0]) for s in SIDES)
        print(f"[fold] imu rpy (deg): {rpy}")
        for s in SIDES:
            print(f"[fold] {s} now (turns): {start[s]}\n[fold] {s} first pose:  {turns[s][0]}")
        print(f"[fold] recording {t[-1]:.1f} s -> {t[-1] / speed:.1f} s at x{speed}; ease in {ease_in:.1f} s, ease out about the same")
        if not (abs(rpy[0]) < UPRIGHT_DEG and abs(rpy[1]) < UPRIGHT_DEG):
            problems.append(f"robot not upright (rpy {rpy})")
        if problems:
            print("[fold] NOT SAFE TO RUN:\n  " + "\n  ".join(problems))
            sys.exit(1)
        print("[fold] checks passed")
        if not a.execute:
            print("[fold] dry run only. Add --execute to move both arms.")
            return

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        with Writer("arm_left.ctrl", Type("arm_ctrl"), keeptime=False) as c_l, \
             Writer("arm_left.torque", Type("arm_torque"), keeptime=False) as e_l, \
             Writer("arm_right.ctrl", Type("arm_ctrl"), keeptime=False) as c_r, \
             Writer("arm_right.torque", Type("arm_torque"), keeptime=False) as e_r:
            ctrl, enable = {"left": c_l, "right": c_r}, {"left": e_l, "right": e_r}

            def command(poses):
                for s in SIDES:
                    with ctrl[s].buf() as b:
                        b["pos"][:] = poses[s]
                        b["vel"][:] = 0
                        b["tau"][:] = 0
                        b["alpha"] = 0.

            def torque(on):
                for s in SIDES:
                    with enable[s].buf() as b:
                        b["enable"][:] = on
                        b["tau_mode"][:] = False
                        b["compliance_mode"] = False

            def ease(a_pose, b_pose, seconds):
                t0 = time.monotonic()
                while True:
                    f = smooth((time.monotonic() - t0) / seconds)
                    command({s: a_pose[s] + f * (b_pose[s] - a_pose[s]) for s in SIDES})
                    if f >= 1 or interrupts > 1:
                        return
                    time.sleep(TICK)

            last = dict(start)
            try:
                for _ in range(10):  # hold the current pose before torque comes on, so nothing jumps
                    command(start)
                    time.sleep(TICK)
                torque(True)
                print("[fold] torque on, easing into the first pose", flush=True)
                first = {s: turns[s][0] for s in SIDES}
                ease(start, first, ease_in)
                last = first
                if not interrupts:
                    print(f"[fold] playing at x{speed}", flush=True)
                    t0, i, n = time.monotonic(), 0, len(t)
                    while i < n - 1 and not interrupts:
                        now = (time.monotonic() - t0) * speed
                        i = min(int(np.searchsorted(t, now)), n - 1)
                        last = {s: turns[s][i] for s in SIDES}
                        command(last)
                        time.sleep(TICK)
                if interrupts < 2:
                    back = max(ease_seconds(cfgs[s], last[s], start[s]) for s in SIDES)
                    print(f"[fold] easing back to the starting pose ({back:.1f} s)", flush=True)
                    ease(last, start, back)
                    time.sleep(.3)
            finally:
                torque(False)
                print("[fold] torque off", flush=True)


if __name__ == "__main__":
    main()
