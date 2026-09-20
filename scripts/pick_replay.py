"""Replay the pick's perception offline, in seconds, with no robot.

Runs exactly the code the robot runs (``pick_object.measure_frame`` and
``combine_picks``) on depth frames saved by ``pick_object.py --record DIR``
(``scripts/pick_lab.sh pull`` fetches them), or on a synthetic tabletop:

    python3 scripts/pick_replay.py artifacts/pick/latest
    python3 scripts/pick_replay.py --synthetic --can 0.42 0.12
    python3 scripts/pick_replay.py DIR --near 0.40 0.15 --repeat 20

It stops where the robot's IK begins: it reports the target, the container,
the reach verdict and the grasp waypoints each pitch would ask the IK for.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick_object as po  # noqa: E402

CAN_RADIUS_METRES = 0.033
CAN_HEIGHT_METRES = 0.122


def synthetic_cloud(can=(0.42, 0.12), box=(0.47, -0.22), tilt_degrees=9.0, table_z=0.74,
                    noise=0.003, seed=0, table_points=150000):
    """A ``camera.points``-frame cloud: tilted table, a can, an open box, clutter."""

    rng = np.random.default_rng(seed)
    slope = math.tan(math.radians(tilt_degrees))

    def surface(x):
        return table_z + slope * (x - 0.2)

    x = rng.uniform(0.2, 1.0, table_points)
    y = rng.uniform(-0.6, 0.6, table_points)
    parts = [np.column_stack((x, y, surface(x) + rng.normal(0, noise, table_points)))]
    if can is not None:
        count = 4000
        angle = rng.uniform(0, 2 * math.pi, count)
        px = can[0] + CAN_RADIUS_METRES * np.cos(angle)
        py = can[1] + CAN_RADIUS_METRES * np.sin(angle)
        parts.append(np.column_stack(
            (px, py, surface(px) + rng.uniform(0, CAN_HEIGHT_METRES, count))))
    if box is not None:
        count, half = 8000, 0.125
        along = rng.uniform(-half, half, count)
        wall = rng.integers(0, 4, count)
        px = box[0] + np.where(wall < 2, along, np.where(wall == 2, -half, half))
        py = box[1] + np.where(wall < 2, np.where(wall == 0, -half, half), along)
        parts.append(np.column_stack((px, py, surface(px) + rng.uniform(0, 0.10, count))))
    parts.append(np.column_stack((rng.uniform(1.0, 3.0, 120000), rng.uniform(-2, 2, 120000),
                                  rng.uniform(0, 2, 120000))))
    arm = np.vstack(parts)
    arm += rng.normal(0, noise / 3, arm.shape)
    return np.column_stack((-arm[:, 1], arm[:, 0], arm[:, 2])).astype(np.float32)


def load_frames(directory):
    files = sorted(Path(directory).glob("frame_*.npz"))
    if not files:
        raise SystemExit(f"no frame_*.npz in {directory}; record some with "
                         "'scripts/pick_lab.sh plan --record /tmp/pick/frames' then 'pull'")
    return [np.load(file)["points"] for file in files]


def replay(frames, near=None, box_side=None, quiet=False):
    """Same lock-on loop as ``pick_object.scan``; returns (plane, item, box, seconds)."""

    report = (lambda *_: None) if quiet else po.log
    picks, box = [], None
    began = time.perf_counter()
    for cloud in frames:
        try:
            plane, item, found = po.measure_frame(cloud, near, None, box_side, report=report)
        except RuntimeError as exc:
            report("scan", f"frame rejected: {exc}")
            continue
        if item is None:
            continue
        if near is None:
            near = (item.center[0], item.center[1])
        picks.append((plane, item))
        box = found
    if not picks:
        raise RuntimeError("no graspable object on any frame")
    plane, item = po.combine_picks(picks, report=report)
    return plane, item, box, time.perf_counter() - began


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", nargs="?", type=Path, help="directory of frame_*.npz")
    parser.add_argument("--synthetic", action="store_true", help="use a generated tabletop")
    parser.add_argument("--can", type=float, nargs=2, default=(0.42, 0.12), metavar=("X", "Y"))
    parser.add_argument("--no-box", action="store_true")
    parser.add_argument("--near", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--box-side", choices=("left", "right"))
    parser.add_argument("--repeat", type=int, default=1, help="time this many replays")
    args = parser.parse_args()
    if args.synthetic or args.frames is None:
        frames = [synthetic_cloud(can=tuple(args.can), box=None if args.no_box else (0.47, -0.22),
                                  seed=seed) for seed in range(5)]
    else:
        frames = load_frames(args.frames)
    near = tuple(args.near) if args.near else None
    box_side = {"left": 1.0, "right": -1.0}.get(args.box_side)
    plane, item, box, seconds = replay(frames, near, box_side)
    for _ in range(args.repeat - 1):
        seconds = min(seconds, replay(frames, near, box_side, quiet=True)[3])

    reach = po.object_reach(item)
    verdict = ("GRASPABLE" if reach <= po.MAX_REACH_METRES
               else "needs lean" if reach <= po.HARD_REACH_METRES else "too far")
    side = "left" if item.center[1] >= 0.0 else "right"
    print(f"\n{len(frames)} frames, {sum(len(f) for f in frames) // len(frames)} points each: "
          f"{seconds:.2f}s total, {seconds / len(frames) * 1000:.0f} ms/frame")
    print(f"target {po.fmt(item.center)} top={item.top:.3f} width={item.width:.3f} "
          f"reach={reach:.3f} ({side} arm) {verdict}")
    print("box    " + ("none" if box is None else
                       f"{po.fmt(box.center)} rim={box.top:.3f} {box.length:.2f}x{box.width:.2f} m"))
    pitches = (po.NEAR_GRASP_PITCHES_DEGREES if reach < po.NEAR_OBJECT_METRES
               else po.GRASP_PITCHES_DEGREES)
    for pitch in pitches:
        pregrasp, grasp, lift, _ = po.grasp_waypoints(item, side, pitch, plane.height_at)
        print(f"pitch {pitch:>4.0f}: pregrasp={po.fmt(pregrasp)} grasp={po.fmt(grasp)} "
              f"tip_above_table={grasp[2] - plane.height_at(grasp[0], grasp[1]):.3f}")


if __name__ == "__main__":
    main()
