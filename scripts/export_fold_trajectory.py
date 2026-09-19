"""Run the scripted fox fold in the simulator and save the arm commands for replay on the robot.

    uv run --locked python scripts/export_fold_trajectory.py --seed 7 --out artifacts/trajectories/fox-seed7.npz

The file holds what the sim's servos were told, sampled on the real arm daemon's 15 ms tick:

    t          (N,)     seconds from the start of the episode
    q          (N, 14)  URDF joint set points [lj0..lj6, rj0..rj6] (J0 in metres, the rest radians).
                        The sim's joints use the real URDF's convention (checked by FK), so the robot
                        converts these with its own Config(...).urdf2q.
    grip       (N, 2)   1 = jaw closed, 0 = open, left then right
    phase      (N,)     index into `phases`
    fold       (N,)     fold number in progress
    tool       (N, 2, 3) commanded fingertip positions, sim world frame (for reference only)

Nothing here knows about the real table: in the sim the paper is at z = PAPER_Z in front of the robot.
Replaying it on hardware moves the hands to those heights in front of the robot, so the first real runs
must have nothing in front of the robot.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bbsim.origami_config import PHASES  # noqa: E402
from bbsim.origami_scene import OrigamiScene  # noqa: E402

TICK = .015  # arm_ctrl period on the robot (@realtime(ms=15))
PHASE_NAMES = [*PHASES, "RECREASE"]


def export(seed, pattern="fox", max_seconds=400.):
    scene = OrigamiScene(pattern, seed)
    rows, next_sample = [], 0.
    while not scene.done and scene.data.time < max_seconds:
        scene.tick()
        if scene.data.time + 1e-9 >= next_sample:
            rows.append((float(scene.data.time), scene.sim.arm_targets.copy(), [float(g) for g in scene.grippers],
                         PHASE_NAMES.index(scene.phase), scene.done_folds,
                         scene.command.copy() if scene.command is not None else np.full((2, 3), np.nan)))
            next_sample += TICK
    report = scene.report()
    scene.close()
    t, q, grip, phase, fold, tool = (np.array(c) for c in zip(*rows))
    return dict(t=t.astype(np.float32), q=q.astype(np.float32), grip=grip.astype(np.float32), phase=phase.astype(np.int64),
                fold=fold.astype(np.int64), tool=tool.astype(np.float32)), report


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pattern", default="fox")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = Path(a.out or f"artifacts/trajectories/{a.pattern}-seed{a.seed}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays, report = export(a.seed, a.pattern)
    if not report["success"]:
        sys.exit(f"sim episode failed ({report['failure']}); not exporting a failed fold")
    np.savez_compressed(out, phases=np.array(PHASE_NAMES), **arrays)
    speed = np.abs(np.diff(arrays["q"], axis=0)).max(axis=0) / TICK
    summary = dict(file=str(out), seed=a.seed, seconds=float(arrays["t"][-1]), samples=len(arrays["t"]),
                   folds=report["folds_completed"], q_min=arrays["q"].min(0).round(3).tolist(), q_max=arrays["q"].max(0).round(3).tolist(),
                   max_joint_speed=speed.round(2).tolist())
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
