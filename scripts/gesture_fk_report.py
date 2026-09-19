"""Report end-effector positions for recorded gesture poses. Runs on the robot.

This tool is read-only: it loads motor-turn recordings, converts them through
the installed arm calibration, and uses BBOS forward kinematics. It never opens
an arm writer or enables torque.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from bbos import Config


def main():
    parser = argparse.ArgumentParser(description="Inspect gesture reach with BBOS FK")
    parser.add_argument("side", choices=("left", "right"))
    parser.add_argument("trajectories", nargs="+", type=Path)
    args = parser.parse_args()

    config = Config(f"arm_{args.side}")
    config.ik.init()

    for path in args.trajectories:
        frames = json.loads(path.read_text())
        if isinstance(frames, dict):
            frames = frames["frames"]
        positions = []
        for frame in frames:
            motor = np.asarray(frame[args.side], dtype=np.float64)
            joints = config.q2urdf(motor.copy())[:7]
            xyz, _ = config.ik.fk(list(joints))
            positions.append(np.asarray(xyz, dtype=np.float64))
        xyz = np.stack(positions)
        reach = np.linalg.norm(xyz, axis=1)
        interesting = {
            "max_forward": int(np.argmax(xyz[:, 0])),
            "max_outward": int(np.argmax(xyz[:, 1]) if args.side == "left" else np.argmin(xyz[:, 1])),
            "max_height": int(np.argmax(xyz[:, 2])),
            "max_reach": int(np.argmax(reach)),
        }
        print(f"{path.name}: {len(frames)} frames")
        for label, index in interesting.items():
            stamp = float(frames[index]["t"])
            point = xyz[index]
            pose = np.asarray(frames[index][args.side], dtype=np.float64)
            print(
                f"  {label:11s} frame={index:4d} t={stamp:6.3f} "
                f"xyz={np.round(point, 3).tolist()} turns={np.round(pose, 4).tolist()}"
            )


if __name__ == "__main__":
    main()
