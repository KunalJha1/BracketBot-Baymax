"""See what the pick sees: the can and the box drawn on the head-camera image.

Two halves, so the drawing can be iterated with no robot:

    # on the robot (pick_lab.sh preview does this and copies the file back)
    python pick_preview.py --capture /tmp/pick/preview.npz
    # locally
    python3 scripts/pick_preview.py artifacts/pick/preview.npz --out preview.jpg

The capture holds one left-eye fisheye frame, one ``camera.points`` cloud and
the live calibration. Rendering runs ``pick_object.measure_frame`` -- the exact
detection the pick uses -- and projects its arm-frame results into the image.
Nothing here opens a writer: it cannot move the robot.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_geometry import (  # noqa: E402
    DEFAULT_D,
    DEFAULT_K,
    DEFAULT_T_POINTS_CAMERA,
    ray_to_fisheye_pixel,
)

CAN = (60, 220, 60)       # BGR
BOX = (40, 170, 255)
OTHER = (190, 190, 190)
EDGE = (255, 120, 60)
EYE_WIDTH = 1280


def capture(path):
    """Robot side: save one image, one cloud and the calibration. Readers only."""

    import table_rest as tr

    _, Config, Reader, _, _ = tr._load_bbos()
    K, D, rectify, T = DEFAULT_K, DEFAULT_D, np.eye(3), DEFAULT_T_POINTS_CAMERA
    source = "documented"
    try:
        calibration = Config("depth").camera_cal()
        K = np.asarray(calibration[0], dtype=np.float64)
        D = np.asarray(calibration[1], dtype=np.float64).reshape(-1)[:4]
        rectify = np.asarray(calibration[4], dtype=np.float64)
        source = "config"
    except Exception as exc:  # noqa: BLE001 - fall back to the documented model
        print(f"[preview] intrinsics fallback ({exc})", flush=True)
    for name in ("depth_b", "depth_custom", "depth"):
        try:
            T = np.asarray(Config(name).T_base_cam.mat(), dtype=np.float64)[:3]
            source += f" {name}.T_base_cam"
            break
        except Exception:  # noqa: BLE001
            continue
    else:
        try:
            T = np.asarray(Config("depth").camera_to_base_3x4, dtype=np.float64)
            source += " depth.camera_to_base_3x4"
        except Exception:  # noqa: BLE001
            pass

    def wait(reader, seconds=5.0):
        deadline = time.monotonic() + seconds
        while not reader.ready():
            if time.monotonic() > deadline:
                raise RuntimeError("topic not publishing")
            time.sleep(0.02)
        return reader.data

    with Reader("camera.points", keeptime=False) as points, \
            Reader("camera.head.jpeg", keeptime=False) as head:
        data = wait(points)
        cloud = np.asarray(data["points"])[:int(data["num_points"])].astype(np.float32).copy()
        image = wait(head)
        jpeg = np.frombuffer(bytes(image["jpeg"][:int(image["jpeg_len"])]), dtype=np.uint8)
    np.savez_compressed(path, points=cloud, jpeg=jpeg, K=K, D=D, rectify=rectify, T=T)
    print(f"[preview] saved {len(cloud)} points, {len(jpeg)} jpeg bytes, calibration={source}",
          flush=True)


class Camera:
    """Arm-frame point -> raw left-eye fisheye pixel."""

    def __init__(self, K, D, rectify, T):
        self.K, self.D = np.asarray(K, float), np.asarray(D, float)
        self.rectify = np.asarray(rectify, float)
        self.R, self.t = np.asarray(T, float)[:3, :3], np.asarray(T, float)[:3, 3]

    def pixel(self, arm_point):
        x, y, z = (float(v) for v in arm_point)
        in_points = np.array([-y, x, z])          # arm (fwd, left, up) -> points (right, fwd, up)
        ray = self.rectify.T @ (self.R.T @ (in_points - self.t))
        if ray[2] <= 0.02:                        # behind or beside the lens
            return None
        u, v = ray_to_fisheye_pixel(ray, self.K, self.D)
        return int(round(u)), int(round(v))


def polyline(cv2, image, camera, arm_points, colour, thickness, closed=True):
    pixels = [camera.pixel(p) for p in arm_points]
    for first, second in zip(pixels, pixels[1:] + (pixels[:1] if closed else [])):
        if first is not None and second is not None:
            cv2.line(image, first, second, colour, thickness, cv2.LINE_AA)
    return [p for p in pixels if p is not None]


def label(cv2, image, text, anchor, colour):
    x, y = anchor
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y, height + 8)
    cv2.rectangle(image, (x - 3, y - height - 6), (x + width + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


def ring(plane, centre_xy, radius, above, count=40):
    points = []
    for angle in np.linspace(0, 2 * math.pi, count, endpoint=False):
        x = centre_xy[0] + radius * math.cos(angle)
        y = centre_xy[1] + radius * math.sin(angle)
        points.append((x, y, plane.height_at(x, y) + above))
    return points


def rectangle(plane, item, above):
    major = np.array([math.cos(item.yaw), math.sin(item.yaw)])
    minor = np.array([-major[1], major[0]])
    centre = np.asarray(item.center[:2])
    corners = []
    for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        x, y = centre + a * major * item.length / 2 + b * minor * item.width / 2
        corners.append((x, y, plane.height_at(x, y) + above))
    return corners


def draw_can(cv2, image, camera, plane, item, colour, text):
    radius = max(item.width, 0.04) / 2
    base = ring(plane, item.center[:2], radius, 0.0)
    top = ring(plane, item.center[:2], radius, item.top)
    polyline(cv2, image, camera, base, colour, 2)
    pixels = polyline(cv2, image, camera, top, colour, 3)
    for index in range(0, len(base), len(base) // 4):
        polyline(cv2, image, camera, [base[index], top[index]], colour, 2, closed=False)
    if pixels:
        label(cv2, image, text, (min(p[0] for p in pixels), min(p[1] for p in pixels) - 8), colour)


def draw_box(cv2, image, camera, plane, item, colour, text):
    floor, rim = rectangle(plane, item, 0.0), rectangle(plane, item, item.top)
    polyline(cv2, image, camera, floor, colour, 2)
    pixels = polyline(cv2, image, camera, rim, colour, 3)
    for low, high in zip(floor, rim):
        polyline(cv2, image, camera, [low, high], colour, 2, closed=False)
    if pixels:
        label(cv2, image, text, (min(p[0] for p in pixels), min(p[1] for p in pixels) - 8), colour)


def top_down(cv2, arm, plane, objects, item, box, size=640, span=(0.0, 1.0, -0.6, 0.6)):
    """Bird's-eye map of the table in the arm frame: robot at the bottom."""

    x0, x1, y0, y1 = span
    canvas = np.full((size, size, 3), 24, np.uint8)

    def to_pixel(x, y):
        return (int((y1 - y) / (y1 - y0) * size), int((x1 - x) / (x1 - x0) * size))

    height = plane.height_above(arm)
    keep = (arm[:, 0] > x0) & (arm[:, 0] < x1) & (arm[:, 1] > y0) & (arm[:, 1] < y1)
    for mask, colour in (((np.abs(height) < 0.012), (70, 70, 70)),
                         ((height > 0.02) & (height < 0.45), (200, 200, 200))):
        chosen = arm[keep & mask][::3]
        cols = ((y1 - chosen[:, 1]) / (y1 - y0) * size).astype(int).clip(0, size - 1)
        rows = ((x1 - chosen[:, 0]) / (x1 - x0) * size).astype(int).clip(0, size - 1)
        canvas[rows, cols] = colour
    cv2.line(canvas, to_pixel(plane.near_edge, y0), to_pixel(plane.near_edge, y1), EDGE, 1)
    for shoulder in (0.0975, -0.0975):
        cv2.circle(canvas, to_pixel(0.0, shoulder), 6, (255, 255, 255), -1)
        for reach, shade in ((0.48, (90, 160, 90)), (0.68, (70, 90, 70))):
            cv2.ellipse(canvas, to_pixel(0.0, shoulder),
                        (int(reach / (y1 - y0) * size), int(reach / (x1 - x0) * size)),
                        0, 180, 360, shade, 1)
    for other in objects:
        if other is not item and other is not box:
            cv2.circle(canvas, to_pixel(*other.center[:2]), 5, OTHER, 1)
    if box is not None:
        corners = np.array([to_pixel(x, y) for x, y, _ in rectangle(plane, box, 0.0)])
        cv2.polylines(canvas, [corners], True, BOX, 2)
    if item is not None:
        cv2.circle(canvas, to_pixel(*item.center[:2]),
                   max(4, int(item.width / 2 / (y1 - y0) * size)), CAN, 2)
    cv2.putText(canvas, "top-down (arm frame): robot below, green arcs = reach 0.48/0.68 m",
                (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    return canvas


def render(capture_path, out_path, near=None, box_side=None):
    import cv2
    import pick_object as po
    from tabletop_scene import find_objects, points_to_arm

    data = np.load(capture_path)
    frame = cv2.imdecode(data["jpeg"], cv2.IMREAD_COLOR)
    image = frame[:, :EYE_WIDTH].copy()       # left eye: the one the calibration describes
    camera = Camera(data["K"], data["D"], data["rectify"], data["T"])
    arm = points_to_arm(data["points"].astype(np.float64))
    lines = []
    try:
        plane, item, box = po.measure_frame(data["points"], near, None, box_side,
                                            report=lambda *_: None)
    except RuntimeError as exc:
        label(cv2, image, f"NO TABLE: {exc}", (20, 40), (60, 60, 255))
        cv2.imwrite(str(out_path), image)
        return [f"no table: {exc}"]
    objects = find_objects(arm, plane)
    edge = [(plane.near_edge, y, plane.height_at(plane.near_edge, y))
            for y in np.linspace(-0.5, 0.5, 30)]
    polyline(cv2, image, camera, edge, EDGE, 2, closed=False)
    for other in objects:
        if other.center[0] > 1.0 or abs(other.center[1]) > 0.6:
            continue
        if (item is None or other.center != item.center) and \
                (box is None or other.center != box.center):
            draw_can(cv2, image, camera, plane, other, OTHER,
                     f"{other.center[0]:.2f},{other.center[1]:.2f}")
    if box is not None:
        draw_box(cv2, image, camera, plane, box, BOX,
                 f"BOX {box.length:.2f}x{box.width:.2f} m rim {box.top:.3f}")
        lines.append(f"box    ({box.center[0]:.3f}, {box.center[1]:.3f}) rim={box.top:.3f} "
                     f"{box.length:.2f}x{box.width:.2f} m")
    else:
        lines.append("box    NOT FOUND")
    if item is not None:
        reach = po.object_reach(item)
        verdict = ("GRASPABLE" if reach <= po.MAX_REACH_METRES
                   else "needs lean" if reach <= po.HARD_REACH_METRES else "TOO FAR")
        draw_can(cv2, image, camera, plane, item, CAN,
                 f"CAN top {item.top:.3f} w {item.width:.3f} reach {reach:.2f} {verdict}")
        lines.append(f"target ({item.center[0]:.3f}, {item.center[1]:.3f}) top={item.top:.3f} "
                     f"width={item.width:.3f} reach={reach:.3f} {verdict}")
    else:
        lines.append("target NOT FOUND")
    lines.append(f"table  tilt={plane.tilt_degrees:.1f}deg near_edge={plane.near_edge:.3f} "
                 f"objects={len(objects)}")
    for row, text in enumerate(lines):
        label(cv2, image, text, (16, 34 + 30 * row), (255, 255, 255))
    overhead = top_down(cv2, arm, plane, objects, item, box, size=image.shape[0])
    cv2.imwrite(str(out_path), np.hstack((image, overhead)))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture_file", nargs="?", type=Path)
    parser.add_argument("--capture", type=Path, help="robot side: write a capture here")
    parser.add_argument("--out", type=Path, default=Path("preview.jpg"))
    parser.add_argument("--near", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--box-side", choices=("left", "right"))
    args = parser.parse_args()
    if args.capture:
        capture(args.capture)
        return
    if args.capture_file is None:
        parser.error("give a capture .npz to render, or --capture on the robot")
    for line in render(args.capture_file, args.out, tuple(args.near) if args.near else None,
                       {"left": 1.0, "right": -1.0}.get(args.box_side)):
        print(line)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
