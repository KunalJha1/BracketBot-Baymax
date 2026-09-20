"""Feasibility harness for "person on the floor" detection from YOLO11 pose.

This script answers one question before anybody writes robot code: does pose
geometry actually separate a fallen person from the everyday postures that look
like one?  It is a concept check, not a detector.

It runs the *real* decision code from ``bbapps/emotion_greeter/ground_safety.py``
over a labelled set of synthetic COCO-17 skeletons placed in the robot base
frame, through two evidence paths:

* ``depth``      - synthesises a ``camera.points`` cloud and runs the shipped
                   ``keypoints_in_base_frame`` association, exactly as the robot
                   would.  This is the path the greeter already uses.
* ``monocular``  - uses no depth at all.  It assumes the lowest visible joint
                   touches the floor, intersects that pixel ray with the floor
                   plane to recover range, then lifts every other joint onto the
                   vertical plane at that range.  This matters because the probe
                   in docs/robot-facts.md found the depth daemon *not running*.

Both paths feed the same ``assess_ground_pose`` thresholds, so the comparison
isolates the evidence source rather than the policy.

Geometry comes from docs/robot-facts.md: head camera 1.55 m above the base
origin, pitched 33 deg down, per-eye fx = fy = 447.13, cx = 618.11, cy = 497.87
at 1280x960.  Fisheye distortion is ignored here; that is acceptable for
posture geometry but must be revisited on real frames.

Run:  python scripts/fall_concept_check.py [--verbose]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bbapps" / "emotion_greeter"))

from ground_safety import (  # noqa: E402
    GroundAssessment,
    GroundAlertTracker,
    Keypoint,
    assess_ground_pose,
    keypoints_in_base_frame,
)

# ---------------------------------------------------------------------------
# Camera model (docs/robot-facts.md)
# ---------------------------------------------------------------------------

DEPTH_W, DEPTH_H = 512, 384          # camera.rect.left / camera.points.idx_2d grid
_SCALE = DEPTH_W / 1280.0
FX = FY = 447.13 * _SCALE
CX, CY = 618.11 * _SCALE, 497.87 * _SCALE
CAM_HEIGHT = 1.55
# camera_to_base rotation, rows from Config("depth").camera_to_base_3x4
R_CAM_TO_BASE = np.array(
    [
        [1.000, 0.017, 0.000],
        [0.010, -0.545, 0.839],
        [0.015, -0.839, -0.545],
    ],
    dtype=np.float64,
)
CAM_ORIGIN = np.array([0.0, 0.0, CAM_HEIGHT])


def project(point_base):
    """Base-frame [right, forward, up] -> depth-grid pixel, or None if not seen."""
    cam = R_CAM_TO_BASE.T @ (np.asarray(point_base, dtype=np.float64) - CAM_ORIGIN)
    if cam[2] <= 0.05:
        return None
    u = FX * cam[0] / cam[2] + CX
    v = FY * cam[1] / cam[2] + CY
    if not (0 <= u < DEPTH_W and 0 <= v < DEPTH_H):
        return None
    return float(u), float(v)


def pixel_ray(u, v):
    """Unit ray direction in the base frame for a depth-grid pixel."""
    direction = R_CAM_TO_BASE @ np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
    return direction / np.linalg.norm(direction)


# ---------------------------------------------------------------------------
# Skeleton builders.  Profile values are (a, b, c) or (a, b, c, confidence).
# ---------------------------------------------------------------------------

def _entry(values):
    conf = values[3] if len(values) > 3 else 0.92
    return values[0], values[1], values[2], conf


def vertical_pose(profile, distance, lateral=0.0):
    """profile[i] = (lateral, height, forward_offset) for an upright-ish body."""
    out = {}
    for index, values in profile.items():
        lat, height, fwd_off, conf = _entry(values)
        out[index] = (np.array([lateral + lat, distance - fwd_off, height]), conf)
    return out


def horizontal_pose(profile, distance, lateral=0.0):
    """profile[i] = (depth_offset, along_right_axis, height) for a body lying down."""
    out = {}
    for index, values in profile.items():
        depth_off, along, height, conf = _entry(values)
        out[index] = (np.array([lateral + along, distance - depth_off, height]), conf)
    return out


# A 1.75 m adult.  Upright profiles: (lateral, height_above_floor, forward_offset)
STANDING = {
    0: (0.00, 1.63, 0.09), 1: (0.035, 1.66, 0.10), 2: (-0.035, 1.66, 0.10),
    3: (0.08, 1.64, 0.00), 4: (-0.08, 1.64, 0.00),
    5: (0.19, 1.43, 0.00), 6: (-0.19, 1.43, 0.00),
    7: (0.21, 1.15, 0.01), 8: (-0.21, 1.15, 0.01),
    9: (0.22, 0.88, 0.05), 10: (-0.22, 0.88, 0.05),
    11: (0.12, 0.95, 0.00), 12: (-0.12, 0.95, 0.00),
    13: (0.11, 0.50, 0.02), 14: (-0.11, 0.50, 0.02),
    15: (0.10, 0.08, 0.02), 16: (-0.10, 0.08, 0.02),
}

SITTING_CHAIR = {
    0: (0.00, 1.17, 0.09), 1: (0.035, 1.20, 0.10), 2: (-0.035, 1.20, 0.10),
    3: (0.08, 1.18, 0.00), 4: (-0.08, 1.18, 0.00),
    5: (0.19, 0.98, 0.00), 6: (-0.19, 0.98, 0.00),
    7: (0.21, 0.72, 0.02), 8: (-0.21, 0.72, 0.02),
    9: (0.20, 0.55, 0.22), 10: (-0.20, 0.55, 0.22),
    11: (0.12, 0.45, 0.00), 12: (-0.12, 0.45, 0.00),
    13: (0.12, 0.47, 0.40), 14: (-0.12, 0.47, 0.40),
    15: (0.11, 0.09, 0.42), 16: (-0.11, 0.09, 0.42),
}

CROUCHING = {
    0: (0.00, 1.02, 0.14), 1: (0.035, 1.05, 0.15), 2: (-0.035, 1.05, 0.15),
    3: (0.08, 1.03, 0.04), 4: (-0.08, 1.03, 0.04),
    5: (0.18, 0.85, 0.06), 6: (-0.18, 0.85, 0.06),
    7: (0.20, 0.62, 0.10), 8: (-0.20, 0.62, 0.10),
    9: (0.18, 0.40, 0.22), 10: (-0.18, 0.40, 0.22),
    11: (0.12, 0.30, 0.00), 12: (-0.12, 0.30, 0.00),
    13: (0.13, 0.42, 0.25), 14: (-0.13, 0.42, 0.25),
    15: (0.11, 0.07, 0.20), 16: (-0.11, 0.07, 0.20),
}

# Hard negative: sitting on the floor, cross-legged.  On the floor, not fallen.
SITTING_FLOOR = {
    0: (0.00, 0.88, 0.10), 1: (0.035, 0.91, 0.11), 2: (-0.035, 0.91, 0.11),
    3: (0.08, 0.89, 0.02), 4: (-0.08, 0.89, 0.02),
    5: (0.18, 0.68, 0.00), 6: (-0.18, 0.68, 0.00),
    7: (0.20, 0.45, 0.04), 8: (-0.20, 0.45, 0.04),
    9: (0.18, 0.25, 0.18), 10: (-0.18, 0.25, 0.18),
    11: (0.12, 0.12, 0.00), 12: (-0.12, 0.12, 0.00),
    13: (0.22, 0.18, 0.30), 14: (-0.22, 0.18, 0.30),
    15: (0.10, 0.14, 0.15), 16: (-0.10, 0.14, 0.15),
}

BENDING_OVER = {
    0: (0.00, 1.00, 0.60), 1: (0.035, 1.03, 0.60), 2: (-0.035, 1.03, 0.60),
    3: (0.08, 1.04, 0.52), 4: (-0.08, 1.04, 0.52),
    5: (0.19, 1.10, 0.42), 6: (-0.19, 1.10, 0.42),
    7: (0.21, 0.80, 0.50), 8: (-0.21, 0.80, 0.50),
    9: (0.20, 0.45, 0.55), 10: (-0.20, 0.45, 0.55),
    11: (0.12, 0.88, 0.00), 12: (-0.12, 0.88, 0.00),
    13: (0.11, 0.46, 0.04), 14: (-0.11, 0.46, 0.04),
    15: (0.10, 0.08, 0.02), 16: (-0.10, 0.08, 0.02),
}

# Horizontal profiles: (depth_offset, along_right_axis, height_above_floor)
FALLEN_SUPINE = {
    0: (0.00, 0.70, 0.20), 1: (0.03, 0.72, 0.22), 2: (-0.03, 0.72, 0.22),
    3: (0.07, 0.68, 0.14), 4: (-0.07, 0.68, 0.14),
    5: (0.18, 0.48, 0.13), 6: (-0.18, 0.48, 0.13),
    7: (0.24, 0.22, 0.11), 8: (-0.24, 0.22, 0.11),
    9: (0.26, -0.02, 0.10), 10: (-0.26, -0.02, 0.10),
    11: (0.11, 0.00, 0.13), 12: (-0.11, 0.00, 0.13),
    13: (0.11, -0.44, 0.13), 14: (-0.11, -0.44, 0.13),
    15: (0.10, -0.86, 0.10), 16: (-0.10, -0.86, 0.10),
}

FALLEN_SIDE = {  # on the left side, knees drawn up
    0: (0.10, 0.62, 0.22), 1: (0.12, 0.64, 0.24), 2: (0.06, 0.63, 0.20),
    3: (0.04, 0.60, 0.26), 4: (0.10, 0.60, 0.14),
    5: (0.06, 0.44, 0.28), 6: (0.10, 0.44, 0.12),
    7: (0.18, 0.22, 0.24), 8: (0.20, 0.24, 0.12),
    9: (0.26, 0.04, 0.18), 10: (0.26, 0.06, 0.12),
    11: (0.02, 0.00, 0.24), 12: (0.04, 0.00, 0.11),
    13: (0.22, -0.34, 0.22), 14: (0.24, -0.34, 0.12),
    15: (0.30, -0.66, 0.14), 16: (0.32, -0.66, 0.10),
}

# Positive, degraded: face down, legs behind furniture.  Low-confidence joints
# are dropped by the association step, so only the upper body survives.
FALLEN_PRONE_OCCLUDED = {
    0: (0.00, 0.68, 0.10, 0.55), 3: (0.07, 0.66, 0.16, 0.50), 4: (-0.07, 0.66, 0.16, 0.50),
    5: (0.18, 0.46, 0.16), 6: (-0.18, 0.46, 0.16),
    7: (0.26, 0.24, 0.12), 8: (-0.26, 0.24, 0.12),
    9: (0.30, 0.02, 0.09), 10: (-0.30, 0.02, 0.09),
    11: (0.11, 0.00, 0.17), 12: (-0.11, 0.00, 0.17),
    13: (0.11, -0.44, 0.15, 0.12), 14: (-0.11, -0.44, 0.15, 0.12),
    15: (0.10, -0.86, 0.12, 0.08), 16: (-0.10, -0.86, 0.12, 0.08),
}

# Hard negative: lying on a 0.45 m sofa.  Same posture, different height.
LYING_SOFA = {
    index: (values[0], values[1], values[2] + 0.45) + tuple(values[3:])
    for index, values in FALLEN_SUPINE.items()
}

# Hard negative: a plank / push-up.  Horizontal and near the floor on purpose.
PLANK = {
    0: (0.00, 0.74, 0.46), 1: (0.03, 0.76, 0.48), 2: (-0.03, 0.76, 0.48),
    3: (0.07, 0.70, 0.44), 4: (-0.07, 0.70, 0.44),
    5: (0.19, 0.52, 0.42), 6: (-0.19, 0.52, 0.42),
    7: (0.21, 0.30, 0.22), 8: (-0.21, 0.30, 0.22),
    9: (0.20, 0.34, 0.06), 10: (-0.20, 0.34, 0.06),
    11: (0.12, 0.00, 0.35), 12: (-0.12, 0.00, 0.35),
    13: (0.11, -0.44, 0.26), 14: (-0.11, -0.44, 0.26),
    15: (0.10, -0.84, 0.12), 16: (-0.10, -0.84, 0.12),
}


def build_scenarios():
    """(name, should_alert, joints) with joints = {index: (xyz, confidence)}."""
    return [
        ("fallen_supine_2m",      True,  horizontal_pose(FALLEN_SUPINE, 2.0)),
        ("fallen_supine_3m",      True,  horizontal_pose(FALLEN_SUPINE, 3.0)),
        ("fallen_side_2.5m",      True,  horizontal_pose(FALLEN_SIDE, 2.5)),
        ("fallen_prone_occluded", True,  horizontal_pose(FALLEN_PRONE_OCCLUDED, 2.4)),
        ("standing_2m",           False, vertical_pose(STANDING, 2.0)),
        ("standing_3.5m",         False, vertical_pose(STANDING, 3.5)),
        ("sitting_chair_2.5m",    False, vertical_pose(SITTING_CHAIR, 2.5)),
        ("crouching_2m",          False, vertical_pose(CROUCHING, 2.0)),
        ("bending_over_2.2m",     False, vertical_pose(BENDING_OVER, 2.2)),
        ("sitting_on_floor_2.5m", False, vertical_pose(SITTING_FLOOR, 2.5)),
        ("lying_on_sofa_2.5m",    False, horizontal_pose(LYING_SOFA, 2.5)),
        ("plank_exercise_2.5m",   False, horizontal_pose(PLANK, 2.5)),
    ]


# ---------------------------------------------------------------------------
# Evidence path 1: synthetic depth cloud -> shipped association code
# ---------------------------------------------------------------------------

def synthesise_depth_cloud(joints, rng, patch_radius=3, dropout=0.15):
    """Build camera.points-style (idx_2d, points) arrays around the skeleton."""
    indices = []
    points = []
    seen = set()
    for xyz, _conf in joints.values():
        pixel = project(xyz)
        if pixel is None:
            continue
        cu, cv = int(round(pixel[0])), int(round(pixel[1]))
        for dv in range(-patch_radius, patch_radius + 1):
            for du in range(-patch_radius, patch_radius + 1):
                u, v = cu + du, cv + dv
                if not (0 <= u < DEPTH_W and 0 <= v < DEPTH_H):
                    continue
                flat = v * DEPTH_W + u
                if flat in seen or rng.random() < dropout:
                    continue
                seen.add(flat)
                indices.append(flat)
                points.append(np.asarray(xyz) + rng.normal(0.0, 0.012, 3))
    if not indices:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(points, dtype=np.float32),
    )


def depth_path(joints, rng):
    keypoints = []
    for index, (xyz, conf) in joints.items():
        pixel = project(xyz)
        if pixel is None:
            continue
        keypoints.append(Keypoint(index, pixel[0], pixel[1], conf))
    idx_2d, points = synthesise_depth_cloud(joints, rng)
    recovered = keypoints_in_base_frame(
        keypoints, idx_2d, points, DEPTH_W, DEPTH_H, keypoint_confidence=0.35
    )
    return assess_ground_pose(recovered), len(keypoints)


# ---------------------------------------------------------------------------
# Evidence path 2: monocular, no depth at all
# ---------------------------------------------------------------------------

# Generous adult upper bounds, in metres.  These are not used to recognise a
# person; they only test whether the monocular scale assumption produced a body
# that could exist.  A body on a bed or sofa back-projects too far away and so
# comes out uniformly too large.
SEGMENT_LIMITS = {
    (5, 11): 0.62, (6, 12): 0.62,   # shoulder -> hip
    (11, 13): 0.55, (12, 14): 0.55,  # hip -> knee
    (13, 15): 0.52, (14, 16): 0.52,  # knee -> ankle
    (5, 7): 0.45, (6, 8): 0.45,      # shoulder -> elbow
}


def implausible_segment(recovered):
    """Return the first body segment that is too long to be a real adult."""
    for (first, second), limit in SEGMENT_LIMITS.items():
        if first in recovered and second in recovered:
            length = float(np.linalg.norm(recovered[first] - recovered[second]))
            if length > limit:
                return (first, second), length, limit
    return None


def monocular_path(joints, keypoint_confidence=0.35):
    """Recover pseudo-3D joints assuming the lowest joint touches the floor.

    The lowest image pixel of a person is nearly always a ground contact, for
    both standing and lying bodies.  Intersecting its ray with z=0 gives range;
    every other joint is then lifted onto the vertical plane at that range.
    The assumption breaks when a body is aligned with the viewing direction.
    """
    pixels = {}
    for index, (xyz, conf) in joints.items():
        if conf < keypoint_confidence:
            continue
        pixel = project(xyz)
        if pixel is not None:
            pixels[index] = pixel
    if len(pixels) < 4:
        return assess_ground_pose({}), pixels

    lowest = max(pixels.values(), key=lambda pixel: pixel[1])
    ray = pixel_ray(*lowest)
    if ray[2] >= -1e-3:  # not pointing at the floor
        return assess_ground_pose({}), pixels
    contact = CAM_ORIGIN + (-CAM_HEIGHT / ray[2]) * ray
    plane_forward = float(contact[1])
    if plane_forward <= 0.2:
        return assess_ground_pose({}), pixels

    recovered = {}
    for index, pixel in pixels.items():
        direction = pixel_ray(*pixel)
        if abs(direction[1]) < 1e-6:
            continue
        scale = plane_forward / direction[1]
        if scale <= 0:
            continue
        recovered[index] = CAM_ORIGIN + scale * direction

    # The floor-contact assumption is what makes this path work.  When the body
    # is not actually on the floor the assumption fails silently and produces a
    # confident wrong answer, so refuse to score an anatomically impossible body.
    bad_segment = implausible_segment(recovered)
    if bad_segment is not None:
        (first, second), length, limit = bad_segment
        return (
            GroundAssessment(
                "unknown",
                0.0,
                f"monocular scale implausible: joints {first}-{second} "
                f"recovered at {length:.2f}m > {limit:.2f}m, body is probably not on the floor",
                len(recovered),
            ),
            pixels,
        )
    return assess_ground_pose(recovered), pixels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def verdict(assessment, should_alert):
    if assessment.state == "unknown":
        return "MISS(unknown)" if should_alert else "ok(unknown)"
    if assessment.suspected:
        return "hit" if should_alert else "FALSE ALARM"
    return "MISS" if should_alert else "ok"


def run(verbose=False, seed=7):
    rng = np.random.default_rng(seed)
    scenarios = build_scenarios()
    rows = []
    for name, should_alert, joints in scenarios:
        depth_assessment, visible = depth_path(joints, rng)
        mono_assessment, _pixels = monocular_path(joints)
        rows.append((name, should_alert, visible, depth_assessment, mono_assessment))

    header = f"{'scenario':<24}{'truth':<10}{'kp':<4}{'depth path':<16}{'monocular path':<16}"
    print(header)
    print("-" * len(header))
    for name, should_alert, visible, depth_a, mono_a in rows:
        truth = "ON FLOOR" if should_alert else "normal"
        print(
            f"{name:<24}{truth:<10}{visible:<4}"
            f"{verdict(depth_a, should_alert):<16}"
            f"{verdict(mono_a, should_alert):<16}"
        )
        if verbose:
            print(f"{'':<24}depth : {depth_a.state} score={depth_a.confidence} {depth_a.reason}")
            print(f"{'':<24}mono  : {mono_a.state} score={mono_a.confidence} {mono_a.reason}")

    print()
    for label, getter in (("depth", 3), ("monocular", 4)):
        hits = sum(1 for row in rows if row[1] and row[getter].suspected)
        misses = sum(1 for row in rows if row[1] and not row[getter].suspected)
        false_alarms = sum(1 for row in rows if not row[1] and row[getter].suspected)
        clears = sum(1 for row in rows if not row[1] and not row[getter].suspected)
        print(
            f"{label:<10} detected {hits}/{hits + misses} on-floor cases, "
            f"{false_alarms} false alarms out of {false_alarms + clears} normal postures"
        )

    print()
    print("Temporal gate (GroundAlertTracker, 2.0 s hold) on a 10 Hz stream:")
    for name, should_alert, _visible, depth_a, _mono in rows:
        if not (should_alert or depth_a.suspected):
            continue
        tracker = GroundAlertTracker(hold_seconds=2.0, clear_seconds=2.0)
        status = "clear"
        for step in range(30):
            status = tracker.update({1: depth_a}, step * 0.1)[1]
        print(f"  {name:<24}after 3.0 s -> {status}")
    return rows


# ---------------------------------------------------------------------------
# Can the "in pain" half work at all?  Start with pixels on the face.
# ---------------------------------------------------------------------------

FULL_FX = 447.13            # per 1280x960 eye, before any downscale
FACE_WIDTH_M = 0.15         # adult bizygomatic width
YUNET_MIN_PX = 32           # optimistic floor for reliable YuNet detection
EXPRESSION_MIN_PX = 80      # below this an AffectNet 224x224 crop is interpolation


def face_feasibility_report():
    print()
    print("Face pixels available for expression, per 1280x960 eye (fx = 447.13):")
    print(f"{'range':<10}{'face px':<10}{'YuNet detect':<16}{'expression usable':<18}")
    print("-" * 54)
    for distance in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0):
        pixels = FULL_FX * FACE_WIDTH_M / distance
        detect = "likely" if pixels >= YUNET_MIN_PX else "unreliable"
        usable = "yes" if pixels >= EXPRESSION_MIN_PX else "no"
        print(f"{distance:<10.1f}{pixels:<10.0f}{detect:<16}{usable:<18}")
    usable_range = FULL_FX * FACE_WIDTH_M / EXPRESSION_MIN_PX
    print()
    print(
        f"  Expression input needs about {EXPRESSION_MIN_PX} px of face, which this camera "
        f"only delivers within ~{usable_range:.2f} m."
    )
    print(
        "  A person is typically first seen on the floor at 2-3 m, where the face is "
        "20-35 px: detectable at best, not classifiable."
    )
    print(
        "  The shipped expression model is AffectNet 8-class (anger, contempt, disgust,\n"
        "  fear, happiness, neutral, sadness, surprise). None of those classes is pain,\n"
        "  and a fallen person's face is usually turned away, downward, or occluded."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fall-detection feasibility harness")
    parser.add_argument("--verbose", action="store_true", help="print per-scenario geometry")
    args = parser.parse_args()
    run(verbose=args.verbose)
    face_feasibility_report()
