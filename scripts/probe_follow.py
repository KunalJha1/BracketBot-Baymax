"""Read-only probe for person-follow gate G0. Runs ON THE ROBOT and opens no writers.

    scp scripts/probe_follow.py bot:/tmp/
    ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/probe_follow.py --out /tmp/follow_probe'
    ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/probe_follow.py --out /tmp/follow_probe_left --person-left'
    scp -r bot:/tmp/follow_probe bot:/tmp/follow_probe_left artifacts/

First run: nobody in front of the robot (measures the robot's own body in the
depth cloud). --person-left: one person stands about 1 m ahead and 0.5 m to the
robot's LEFT (tells which way base +x points).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from bbos import Config, Reader

TOPICS = ("camera.depth", "camera.points", "imu.orientation", "drive.state", "drive.status")
CONFIGS = ("drive", "depth", "base")


def field_names(data):
    names = getattr(getattr(data, "dtype", None), "names", None)
    if names:
        return list(names)
    try:
        return list(data.keys())
    except AttributeError:
        return []


def describe(value):
    array = np.asarray(value)
    info = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size <= 8:
        info["value"] = array.tolist()
    return info


def sample_topic(topic, seconds):
    frames, stamps, first = 0, [], None
    try:
        with Reader(topic, keeptime=False) as reader:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if not reader.ready():
                    time.sleep(0.002)
                    continue
                data = reader.data
                frames += 1
                if first is None:
                    first = {name: np.array(data[name]).copy() for name in field_names(data)}
                if first and "timestamp" in first:
                    stamps.append(float(np.asarray(data["timestamp"]).item()))
    except Exception as exc:  # report and keep probing the other topics
        return {"error": repr(exc)}, None
    report = {"rate_hz": round(frames / seconds, 1),
              "fields": {name: describe(value) for name, value in (first or {}).items()}}
    if len(stamps) > 1:
        report["timestamp_first"] = stamps[0]
        report["timestamp_span_over_window"] = stamps[-1] - stamps[0]  # ~seconds, ms, or ns: tells the unit
    return report, first


def config_values(name):
    try:
        cfg = Config(name)
    except Exception as exc:
        return {"error": repr(exc)}
    values = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        try:
            value = getattr(cfg, key)
        except Exception:
            continue
        if isinstance(value, (bool, int, float, str)):
            values[key] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 16:
            values[key] = [v if isinstance(v, (bool, int, float, str)) else repr(v) for v in value]
    return values


def cloud_summary(points, person_left):
    p = np.asarray(points, dtype=float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    near = (y < 0.45) & (np.abs(x) < 0.4) & (z > 0.05) & (z < 1.7)
    summary = {"num_points": int(len(p)), "self_points": int(near.sum())}
    if near.any():
        summary["self_box_base_xyz_min"] = [round(float(v), 3) for v in p[near].min(axis=0)]
        summary["self_box_base_xyz_max"] = [round(float(v), 3) for v in p[near].max(axis=0)]
    if person_left:
        body = (y > 0.6) & (y < 1.6) & (z > 0.8) & (z < 1.6) & (np.abs(x) < 1.0)
        median_x = float(np.median(x[body])) if body.any() else None
        summary["person_points"] = int(body.sum())
        summary["person_median_base_x"] = median_x
        if median_x is not None:
            summary["verdict"] = (
                "+x is LEFT: set BASE_LEFT_SIGN = +1.0" if median_x > 0
                else "+x is RIGHT: keep BASE_LEFT_SIGN = -1.0"
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Read-only probe for person-follow gate G0")
    parser.add_argument("--out", type=Path, default=Path("/tmp/follow_probe"))
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--person-left", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"topics": {}, "configs": {name: config_values(name) for name in CONFIGS}}
    frames = {}
    for topic in TOPICS:
        report["topics"][topic], frames[topic] = sample_topic(topic, args.seconds)

    points = frames.get("camera.points")
    if points and "points" in points:
        n = int(np.asarray(points["num_points"]).item())
        report["cloud"] = cloud_summary(points["points"][:n], args.person_left)
        np.savez_compressed(args.out / "points.npz", points=points["points"][:n], mask=points["mask"][:n])
    depth = frames.get("camera.depth")
    if depth and "depth" in depth:
        np.save(args.out / "camera_depth.npy", depth["depth"])

    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=repr))
    print(json.dumps(report, indent=2, default=repr))


if __name__ == "__main__":
    main()
