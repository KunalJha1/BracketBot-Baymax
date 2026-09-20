"""Build the wrap-around hug from a planned hand path.

The teleoperated hug crossed both hands in front of the robot's own chest, so
it never went around the person. This plans the hands in the base frame
instead: open wide, reach forward past the person's sides, close in behind
their back, squeeze gently, then leave along the same path. The original
recording lives on as ``reach.json`` and supplies the resting pose.

Forward kinematics come from the robot's own arm config, so run it there:

    scripts/bot push scripts/generate_hug_asset.py bbapps/greeter/movements/reach.json --to /tmp
    scripts/bot py /tmp/generate_hug_asset.py --source /tmp/reach.json --output /tmp/hug.json
    scripts/bot pull /tmp/hug.json bbapps/greeter/movements/hug.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "bbapps" / "greeter" / "movements" / "reach.json"
DEFAULT_OUTPUT = ROOT / "bbapps" / "greeter" / "movements" / "hug.json"
SIDES = ("left", "right")
FRAME_SECONDS = 0.02
PLAN_SECONDS = 0.01
# Rounds every corner of the hand path so no joint ever stops and restarts.
CORNER_SIGMA_SECONDS = 0.16
JOINT_SIGMA_SECONDS = 0.05
# Stay inside the URDF's +/-1/3 turn joint range.
JOINT_LIMIT_TURNS = 0.30
MAX_IK_ERROR_METRES = 0.02
# Both arms share the midline while closed; never let the hands meet.
MIN_HAND_GAP_METRES = 0.22
# Left-hand waypoints in the base frame (forward, left, height) at recording
# time; playback runs at 0.6x. The right hand mirrors them. "rest" is the
# recorded hanging pose. Arms stay wider than an adult's shoulders until the
# hands are past the person, which is also what keeps the depth gate clear.
WAYPOINTS = (
    (0.0, "rest"),
    (0.9, (0.24, 0.38, 1.06)),   # open wide
    (1.8, (0.44, 0.34, 1.20)),   # reach past the person's sides
    (2.7, (0.52, 0.19, 1.20)),   # close in behind their back
    (3.3, (0.50, 0.15, 1.19)),   # gentle squeeze
    (3.8, (0.515, 0.18, 1.20)),
    (4.3, (0.50, 0.15, 1.19)),   # second squeeze
    (4.8, (0.52, 0.19, 1.20)),
    (5.6, (0.44, 0.34, 1.20)),   # open before pulling back
    (6.4, (0.24, 0.38, 1.06)),
    (7.3, "rest"),
)


def gaussian_smooth(values, sigma_samples):
    """Smooth rows of ``values`` while keeping both end points fixed."""
    radius = int(np.ceil(4 * sigma_samples))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate(
        [np.repeat(values[:1], radius, axis=0), values, np.repeat(values[-1:], radius, axis=0)]
    )
    return np.column_stack(
        [np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])]
    )


def hand_path(rest, mirror):
    stamps = np.asarray([stamp for stamp, _ in WAYPOINTS])
    points = np.asarray(
        [
            rest if point == "rest" else np.asarray(point) * (1.0, mirror, 1.0)
            for _, point in WAYPOINTS
        ]
    )
    # Dwell at rest so smoothing leaves zero velocity at both ends.
    lead = 4 * CORNER_SIGMA_SECONDS
    times = np.arange(0.0, stamps[-1] + 2 * lead + PLAN_SECONDS / 2, PLAN_SECONDS)
    path = np.column_stack(
        [np.interp(times - lead, stamps, points[:, axis]) for axis in range(3)]
    )
    return times, gaussian_smooth(path, CORNER_SIGMA_SECONDS / PLAN_SECONDS)


def solve_path(config, rest_motor, path):
    """Position-only damped least squares IK, continued along the path."""

    def fk(motor):
        joints = config.q2urdf(np.asarray(motor, dtype=np.float64).copy())[:7]
        position, _ = config.ik.fk(list(joints))
        return np.asarray(position, dtype=np.float64)

    motor = np.asarray(rest_motor, dtype=np.float64).copy()
    offset = path[0] - fk(motor)
    solved = []
    worst, worst_target = 0.0, path[0]
    for target in path:
        for _ in range(8):
            error = target - offset - fk(motor)
            if np.linalg.norm(error) < 5e-4:
                break
            jacobian = np.empty((3, 6))
            for joint in range(6):
                nudged = motor.copy()
                nudged[joint + 1] += 1e-4
                jacobian[:, joint] = (fk(nudged) - fk(motor)) / 1e-4
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(3), error
            )
            motor[1:7] = np.clip(
                motor[1:7] + np.clip(step, -0.02, 0.02),
                -JOINT_LIMIT_TURNS,
                JOINT_LIMIT_TURNS,
            )
        miss = float(np.linalg.norm(target - offset - fk(motor)))
        if miss > worst:
            worst, worst_target = miss, target
        solved.append(motor.copy())
    if worst > MAX_IK_ERROR_METRES:
        raise RuntimeError(
            f"hand path is out of reach by {worst:.3f} m near {np.round(worst_target, 2)}"
        )
    return np.asarray(solved), fk


def build_hug(source_frames):
    from bbos import Config

    rest = source_frames[0]
    trajectories = {}
    hands = {}
    for side, mirror in (("left", 1.0), ("right", -1.0)):
        config = Config(f"arm_{side}")
        config.ik.init()
        joints = config.q2urdf(np.asarray(rest[side], dtype=np.float64).copy())[:7]
        rest_hand = np.asarray(config.ik.fk(list(joints))[0], dtype=np.float64)
        times, path = hand_path(rest_hand, mirror)
        solved, fk = solve_path(config, rest[side], path)
        solved = gaussian_smooth(solved, JOINT_SIGMA_SECONDS / PLAN_SECONDS)
        trajectories[side] = solved
        hands[side] = np.asarray([fk(pose) for pose in solved[::10]])
    raised = hands["left"][:, 2] > 0.9
    gap = float(np.linalg.norm(hands["left"] - hands["right"], axis=1)[raised].min())
    if gap < MIN_HAND_GAP_METRES:
        raise RuntimeError(f"hands come within {gap:.3f} m of each other")

    stride = round(FRAME_SECONDS / PLAN_SECONDS)
    frames = [
        {
            "t": round(float(times[index]), 4),
            "left": [round(float(v), 5) for v in trajectories["left"][index]],
            "right": [round(float(v), 5) for v in trajectories["right"][index]],
        }
        for index in range(0, len(times), stride)
    ]
    report = {
        "seconds": frames[-1]["t"],
        "min_hand_gap_m": round(gap, 3),
        "peak_turns_per_second": round(
            max(
                float(np.abs(np.diff(trajectories[side], axis=0)).max()) / PLAN_SECONDS
                for side in SIDES
            ),
            3,
        ),
        "joint_min": {s: np.round(trajectories[s].min(axis=0), 3).tolist() for s in SIDES},
        "joint_max": {s: np.round(trajectories[s].max(axis=0), 3).tolist() for s in SIDES},
        "hand_min": {s: np.round(hands[s].min(axis=0), 3).tolist() for s in SIDES},
        "hand_max": {s: np.round(hands[s].max(axis=0), 3).tolist() for s in SIDES},
    }
    return frames, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frames, report = build_hug(json.loads(args.source.read_text()))
    args.output.write_text(json.dumps(frames) + "\n")
    print(json.dumps(report, indent=1))
    print(f"wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
