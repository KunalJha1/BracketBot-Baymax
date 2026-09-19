"""Run the real fall-detection logic on real head-camera frames.

This is the bridge between ``scripts/fall_concept_check.py`` (synthetic
skeletons, proves the decision logic) and the robot (real bodies, real
keypoint noise).  It is read-only and never moves the robot.

Two ways to run it.

Off the robot, on frames captured earlier::

    python scripts/fall_check_frame.py --image head_frames/head_*.jpg --annotate out/

On the robot, against the live camera topic::

    uv run --no-sync --project ~/bbos python fall_check_frame.py --live --count 20

Unlike the concept check, this uses the **fisheye** camera model
(``cv2.fisheye.undistortPoints`` with k = [0.1287, -0.0281, 0, 0]) rather than a
pinhole approximation, because a body on the floor appears near the image edge
where distortion is worst.

The head topic publishes a 2560x960 side-by-side stereo frame.  Only one
1280x960 eye is ever used; inferring on the joined image would be meaningless.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "bbapps" / "emotion_greeter", Path(__file__).resolve().parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ground_safety import (  # noqa: E402
    GroundAssessment,
    GroundAlertTracker,
    Keypoint,
    assess_ground_pose,
    keypoints_in_base_frame,
)

# --- Camera model, from docs/robot-facts.md -------------------------------

EYE_W, EYE_H = 1280, 960
INTRINSICS = {
    "left": dict(fx=447.13, fy=447.13, cx=618.11, cy=497.87),
    "right": dict(fx=446.87, fy=446.87, cx=616.58, cy=498.80),
}
FISHEYE_D = np.array([0.1287, -0.0281, 0.0, 0.0], dtype=np.float64)
CAM_HEIGHT = 1.55
R_CAM_TO_BASE = np.array(
    [
        [1.000, 0.017, 0.000],
        [0.010, -0.545, 0.839],
        [0.015, -0.839, -0.545],
    ],
    dtype=np.float64,
)
CAM_ORIGIN = np.array([0.0, 0.0, CAM_HEIGHT])

SEGMENT_LIMITS = {
    (5, 11): 0.62, (6, 12): 0.62,
    (11, 13): 0.55, (12, 14): 0.55,
    (13, 15): 0.52, (14, 16): 0.52,
    (5, 7): 0.45, (6, 8): 0.45,
}
SKELETON_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def camera_matrix(eye):
    values = INTRINSICS[eye]
    return np.array(
        [[values["fx"], 0.0, values["cx"]], [0.0, values["fy"], values["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rays_in_base_frame(pixels, eye):
    """Undistort fisheye pixels and return unit ray directions in the base frame."""
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    normalised = cv2.fisheye.undistortPoints(points, camera_matrix(eye), FISHEYE_D)
    normalised = normalised.reshape(-1, 2)
    directions = np.column_stack([normalised, np.ones(len(normalised))])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions @ R_CAM_TO_BASE.T


def implausible_segment(recovered):
    for (first, second), limit in SEGMENT_LIMITS.items():
        if first in recovered and second in recovered:
            length = float(np.linalg.norm(recovered[first] - recovered[second]))
            if length > limit:
                return (first, second), length, limit
    return None


def monocular_assess(keypoints, eye, keypoint_confidence=0.35):
    """Floor-contact monocular estimate, with the anthropometric scale gate."""
    usable = [kp for kp in keypoints if kp.confidence >= keypoint_confidence]
    if len(usable) < 4:
        return GroundAssessment("unknown", 0.0, "fewer than four confident keypoints", len(usable))

    directions = rays_in_base_frame([(kp.x, kp.y) for kp in usable], eye)
    lowest = int(np.argmax([kp.y for kp in usable]))
    contact_ray = directions[lowest]
    if contact_ray[2] >= -1e-3:
        return GroundAssessment(
            "unknown", 0.0, "lowest keypoint does not look at the floor", len(usable)
        )
    contact = CAM_ORIGIN + (-CAM_HEIGHT / contact_ray[2]) * contact_ray
    plane_forward = float(contact[1])
    if plane_forward <= 0.2:
        return GroundAssessment("unknown", 0.0, "implied range under 0.2 m", len(usable))

    recovered = {}
    for keypoint, direction in zip(usable, directions):
        if abs(direction[1]) < 1e-6:
            continue
        scale = plane_forward / direction[1]
        if scale > 0:
            recovered[keypoint.index] = CAM_ORIGIN + scale * direction

    bad = implausible_segment(recovered)
    if bad is not None:
        (first, second), length, limit = bad
        return GroundAssessment(
            "unknown",
            0.0,
            f"monocular scale implausible: joints {first}-{second} at {length:.2f}m "
            f"> {limit:.2f}m, body is probably not on the floor",
            len(recovered),
        )
    return assess_ground_pose(recovered)


def split_eye(frame, eye):
    """Take one 1280x960 eye out of a 2560x960 side-by-side stereo frame."""
    height, width = frame.shape[:2]
    if width >= 2 * EYE_W:
        half = width // 2
        return frame[:, :half] if eye == "left" else frame[:, half:]
    return frame


def resolve_model(explicit):
    """Find the checkpoint, so the same command works in the repo and in /tmp."""
    if explicit is not None:
        if not Path(explicit).is_file():
            raise SystemExit(f"model not found: {explicit}")
        return Path(explicit)
    here = Path(__file__).resolve().parent
    for candidate in (here / "yolo11n-pose.pt", Path.cwd() / "yolo11n-pose.pt", REPO_ROOT / "yolo11n-pose.pt"):
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "yolo11n-pose.pt not found next to this script, in the working directory, "
        "or at the repo root; pass --model explicitly"
    )


def load_model(weights, device):
    from ultralytics import YOLO

    return YOLO(str(weights), task="pose")


def people_in_frame(model, frame, device, imgsz, confidence):
    result = model.predict(
        frame, classes=[0], imgsz=imgsz, device=device, conf=confidence, verbose=False
    )[0]
    if result.keypoints is None or len(result.boxes) == 0:
        return []
    coordinates = result.keypoints.xy.cpu().numpy()
    scores = (
        result.keypoints.conf.cpu().numpy()
        if result.keypoints.conf is not None
        else np.ones(coordinates.shape[:2], dtype=np.float32)
    )
    people = []
    for person_xy, person_conf in zip(coordinates, scores):
        people.append(
            [
                Keypoint(index, float(x), float(y), float(c))
                for index, ((x, y), c) in enumerate(zip(person_xy, person_conf))
            ]
        )
    return people


def annotate(frame, keypoints, assessment):
    canvas = frame.copy()
    colour = (0, 0, 255) if assessment.suspected else (0, 200, 0)
    lookup = {kp.index: kp for kp in keypoints if kp.confidence >= 0.35}
    for first, second in SKELETON_EDGES:
        if first in lookup and second in lookup:
            cv2.line(
                canvas,
                (int(lookup[first].x), int(lookup[first].y)),
                (int(lookup[second].x), int(lookup[second].y)),
                colour,
                2,
            )
    for keypoint in lookup.values():
        cv2.circle(canvas, (int(keypoint.x), int(keypoint.y)), 4, colour, -1)
    cv2.putText(
        canvas,
        f"{assessment.state} score={assessment.confidence:.2f}",
        (20, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        colour,
        2,
    )
    cv2.putText(canvas, assessment.reason[:110], (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
    return canvas


def report(source, index, assessment):
    flag = "ALERT" if assessment.suspected else assessment.state.upper()
    print(f"{source}  person {index}: {flag:<26} score={assessment.confidence:<8} {assessment.reason}")


def run_images(args, model):
    paths = []
    for pattern in args.image:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"no images matched: {args.image}")
    if args.annotate:
        args.annotate.mkdir(parents=True, exist_ok=True)

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: unreadable, skipped")
            continue
        eye_frame = split_eye(frame, args.eye)
        people = people_in_frame(model, eye_frame, args.device, args.imgsz, args.confidence)
        if not people:
            print(f"{Path(path).name}: no person detected")
            continue
        for index, keypoints in enumerate(people):
            assessment = monocular_assess(keypoints, args.eye, args.keypoint_confidence)
            report(Path(path).name, index, assessment)
            if args.annotate:
                out = args.annotate / f"{Path(path).stem}_p{index}.jpg"
                cv2.imwrite(str(out), annotate(eye_frame, keypoints, assessment))


def run_live(args, model):
    """Read the live camera topic on the robot.  Read-only; nothing moves."""
    from bbos import Reader

    tracker = GroundAlertTracker(hold_seconds=args.hold, clear_seconds=args.hold)
    with Reader("camera.head.jpeg", keeptime=False) as reader:
        for step in range(args.count):
            deadline = time.monotonic() + 3
            while not reader.ready():
                if time.monotonic() > deadline:
                    raise SystemExit("no camera frame: is the camera daemon running?")
                time.sleep(0.005)
            buffer = bytes(reader.data["jpeg"][: int(reader.data["jpeg_len"])])
            frame = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            eye_frame = split_eye(frame, args.eye)
            started = time.perf_counter()
            people = people_in_frame(model, eye_frame, args.device, args.imgsz, args.confidence)
            elapsed = (time.perf_counter() - started) * 1000

            assessments = {}
            for index, keypoints in enumerate(people):
                assessments[index] = monocular_assess(keypoints, args.eye, args.keypoint_confidence)
            statuses = tracker.update(assessments, time.monotonic())
            if not people:
                print(f"[{step:03d}] {elapsed:5.0f} ms  no person")
            for index, assessment in assessments.items():
                print(
                    f"[{step:03d}] {elapsed:5.0f} ms  person {index}: "
                    f"{statuses.get(index, '?'):<8} {assessment.state:<26} "
                    f"score={assessment.confidence:<8} {assessment.reason}"
                )
            if args.annotate and people:
                args.annotate.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(args.annotate / f"live_{step:03d}.jpg"),
                    annotate(eye_frame, people[0], assessments[0]),
                )
            time.sleep(args.every)


def main():
    parser = argparse.ArgumentParser(description="Run fall-detection logic on real head frames")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", nargs="+", help="image path(s) or glob(s)")
    source.add_argument("--live", action="store_true", help="read camera.head.jpeg on the robot")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO pose checkpoint; by default looked up next to this script, "
        "in the working directory, then at the repo root",
    )
    parser.add_argument("--eye", choices=("left", "right"), default="left")
    parser.add_argument("--device", default="cpu", help="cpu, 0, mps")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.4, help="person box confidence")
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--count", type=int, default=20, help="live frames to read")
    parser.add_argument("--every", type=float, default=0.2, help="live seconds between frames")
    parser.add_argument("--hold", type=float, default=2.0, help="live alert hold seconds")
    parser.add_argument("--annotate", type=Path, help="directory to write annotated frames into")
    args = parser.parse_args()

    weights = resolve_model(args.model)
    model = load_model(weights, args.device)
    if args.live:
        run_live(args, model)
    else:
        run_images(args, model)


if __name__ == "__main__":
    main()
