"""Just the "YOLO on real robot frames" window, live from the robot: no sim, for demos.

    uv run --locked --extra yolo python scripts/live_robot.py --robot bot [--run artifacts/yolo/v2]

The robot sends its head camera's table ROI (left eye, the crop YOLO trains on) as a small JPEG, one per
request, so the view stays current even on a slow link. Every new frame goes through the newest
checkpoint (reloaded when it changes), with the classical fallback finder when YOLO sees no sheet, and
bbsim/workbench/yolo/tracker.py keeps the corners steady: it only looks near the table's middle, keeps
corner labels fixed, and carries corners hidden under an arm along with the rest of the sheet. Crosses
are corners seen this frame, rings are carried-along ones. The title
line shows the camera frame rate reaching the laptop and the detection time per frame.

Keys: q / Esc quits, space freezes the view.
"""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_views import Model, RobotFrames, SheetTracker, draw_track, tracked_corners  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="bot", help="ssh host of the robot (bot over USB, botwifi over Wi-Fi)")
    p.add_argument("--run", type=Path, default=Path("artifacts/yolo/v2"))
    p.add_argument("--quality", type=int, default=90, help="JPEG quality the robot sends at")
    a = p.parse_args()
    model = Model(a.run)
    if model.get() is None:
        sys.exit(f"no checkpoint in {a.run}")
    robot = RobotFrames(a.robot, a.quality)
    cv2.namedWindow("YOLO on real robot frames", cv2.WINDOW_NORMAL)
    tracker = SheetTracker()
    shown, frozen, infer = 0, False, 0.
    while True:
        frame = robot.latest
        if frame and frame[0] != shown and not frozen:
            shown = frame[0]
            roi = cv2.imdecode(np.frombuffer(frame[1], np.uint8), cv2.IMREAD_COLOR)
            t0 = time.monotonic()
            track = tracked_corners(model, tracker, roi, t0)
            infer = .9 * infer + .1 * (time.monotonic() - t0) if infer else time.monotonic() - t0
            draw_track(roi, track)
            cv2.putText(roi, f"live from {a.robot}  {robot.fps:.1f} fps  detect {infer * 1000:.0f} ms", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("YOLO on real robot frames", roi)
        elif robot.error:
            sys.exit(robot.error)
        key = cv2.waitKey(5) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            frozen = not frozen
    robot.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
