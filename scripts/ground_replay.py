"""Replay a robot recording through the ground-safety pipeline, off the robot.

Record on the robot (read-only) with ``scripts/ground_record.py``, pull the
folder, then::

    python3 scripts/ground_replay.py RECORDING_DIR --model yolo11n-pose.onnx --sheet out.jpg

Every frame goes through the same code the robot runs: pose on camera.rect, the
raw-resolution floor crop, the merge, tracking, the depth-free floor test and
the alert hold. Recordings carry no depth cloud, so every person is judged by
the depth-free test, which is the path that matters beyond 1.7 m.
"""

from __future__ import annotations

import argparse
import collections
import glob
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bbapps" / "emotion_greeter"))

import main as greeter  # noqa: E402
from floor_roi import floor_crop  # noqa: E402
from ground_safety import GroundAlertTracker, assess_ground_pose_monocular  # noqa: E402

COLORS = {"alert": (0, 0, 255), "checking": (0, 165, 255), "clear": (0, 200, 0), "unknown": (160, 160, 160)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "yolo11n-pose.onnx")
    parser.add_argument("--no-floor-roi", action="store_true")
    parser.add_argument("--sheet", type=Path, help="write an annotated contact sheet here")
    parser.add_argument("--every", type=int, default=6, help="sheet: one tile per this many frames")
    args = parser.parse_args()

    detector = greeter.PersonDetector(args.model)
    tracker, alerts = greeter.PersonTracker(), GroundAlertTracker()
    files = sorted(glob.glob(str(args.recording / "f*.npz")))
    statuses, reasons, tiles, sources = collections.Counter(), collections.Counter(), [], collections.Counter()
    start = None
    rect_dets, floor_dets, rect_at, floor_at, focus, misses = [], [], 0.0, 0.0, None, 0
    for index, path in enumerate(files):
        data = np.load(path)
        now = float(data["t"])
        start = start if start is not None else now
        frame = cv2.cvtColor(data["rect"], cv2.COLOR_RGB2BGR)
        # Same one-pass-per-frame schedule as the robot (see main.py).
        has_raw = not args.no_floor_roi and "raw_left" in data.files
        if has_raw and (focus == "floor" or (focus is None and index % 3 == 2)):
            crop = cv2.cvtColor(np.ascontiguousarray(floor_crop(data["raw_left"])), cv2.COLOR_RGB2BGR)
            floor_dets = greeter.detections_from_floor_crop(detector.detect(crop), frame.shape[1], frame.shape[0])
            floor_at = now
            sources["floor crop passes"] += 1
        else:
            rect_dets, rect_at = detector.detect(frame), now
            sources["rect passes"] += 1
        if now - rect_at > greeter.VIEW_CARRY_S:
            rect_dets = []
        if now - floor_at > greeter.VIEW_CARRY_S:
            floor_dets = []
        detections = greeter.merge_detections(rect_dets, floor_dets)
        sources["merged"] += len(detections)
        tracked = tracker.update(detections)
        assessments = {
            item.track_id: assess_ground_pose_monocular(item.detection.keypoints, frame.shape[1], frame.shape[0])
            for item in tracked
        }
        latched = alerts.update(assessments, now)
        focus, misses = greeter.next_view_focus(focus, misses, latched, tracked, floor_dets)
        overall = "alert" if "alert" in latched.values() else "checking" if "checking" in latched.values() else "clear"
        statuses[overall] += 1
        for item in tracked:
            assessment, status = assessments[item.track_id], latched.get(item.track_id, "unknown")
            reasons[(assessment.state, assessment.reason.split(" would span")[0][:60])] += 1
            box = item.detection
            cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), COLORS[status], 2)
            for kp in box.keypoints:
                if kp.confidence >= 0.35:
                    cv2.circle(frame, (int(kp.x), int(kp.y)), 2, COLORS[status], -1)
            cv2.putText(frame, f"#{item.track_id} {status}", (box.x1, max(12, box.y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[status], 1)
            if status in ("checking", "alert"):
                print(f"{now - start:6.1f}s #{item.track_id} {status:8s} {assessment.reason}")
        if index % args.every == 0:
            cv2.putText(frame, f"{now - start:.0f}s {overall}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(frame)

    print(f"{len(files)} frames: {dict(statuses)}")
    print("per frame:", {key: round(value / max(1, len(files)), 2) for key, value in sources.items()})
    for (state, reason), count in reasons.most_common(6):
        print(f"  {count:4d}  {state:26s} {reason}")
    if args.sheet and tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.sheet), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
        print("sheet:", args.sheet)


if __name__ == "__main__":
    main()
