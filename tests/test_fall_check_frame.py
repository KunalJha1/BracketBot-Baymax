"""Geometry tests for the on-robot frame checker.

These cover the camera model only, so they run without ultralytics or a robot.
The fisheye model is the part that must be right: a body on the floor images
near the frame edge, where a pinhole approximation is badly wrong.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("cv2")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "bbapps" / "emotion_greeter"))

from fall_check_frame import (  # noqa: E402
    CAM_HEIGHT,
    CAM_ORIGIN,
    EYE_H,
    EYE_W,
    INTRINSICS,
    R_CAM_TO_BASE,
    Keypoint,
    implausible_segment,
    monocular_assess,
    rays_in_base_frame,
    split_eye,
)


def floor_hit(u, v, eye="left"):
    ray = rays_in_base_frame([(u, v)], eye)[0]
    if ray[2] >= 0:
        return None
    return CAM_ORIGIN + (-CAM_HEIGHT / ray[2]) * ray


def test_optical_centre_hits_the_documented_floor_distance():
    """docs/robot-facts.md: 33 deg down from 1.55 m puts the centre ~2.4 m ahead."""
    left = INTRINSICS["left"]
    hit = floor_hit(left["cx"], left["cy"])
    assert hit is not None
    assert hit[1] == pytest.approx(2.4, abs=0.1)
    assert hit[0] == pytest.approx(0.0, abs=0.05)


def test_lower_image_rows_map_to_nearer_floor():
    left = INTRINSICS["left"]
    distances = []
    for v in (500, 650, 800, 930):
        hit = floor_hit(left["cx"], v)
        assert hit is not None, f"row {v} should see the floor"
        distances.append(hit[1])
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < 1.0


def test_fisheye_and_pinhole_diverge_at_the_frame_edge():
    """Justifies undistorting: the edge is where a fallen body appears."""
    left = INTRINSICS["left"]

    def pinhole_ray(u, v):
        direction = np.array(
            [(u - left["cx"]) / left["fx"], (v - left["cy"]) / left["fy"], 1.0]
        )
        direction /= np.linalg.norm(direction)
        return R_CAM_TO_BASE @ direction

    def angle_between(u, v):
        fisheye = rays_in_base_frame([(u, v)], "left")[0]
        return np.degrees(np.arccos(np.clip(np.dot(fisheye, pinhole_ray(u, v)), -1, 1)))

    assert angle_between(left["cx"], left["cy"] + 2) < 2.0
    assert angle_between(1270, 800) > 10.0


def test_split_eye_takes_one_half_of_a_stereo_frame():
    stereo = np.zeros((EYE_H, 2 * EYE_W, 3), dtype=np.uint8)
    stereo[:, EYE_W:] = 255
    assert split_eye(stereo, "left").shape == (EYE_H, EYE_W, 3)
    assert split_eye(stereo, "left").max() == 0
    assert split_eye(stereo, "right").min() == 255


def test_split_eye_passes_through_a_single_eye_frame():
    single = np.zeros((EYE_H, EYE_W, 3), dtype=np.uint8)
    assert split_eye(single, "left").shape == single.shape


def test_too_few_confident_keypoints_is_unknown_not_an_alert():
    keypoints = [Keypoint(5, 600.0, 700.0, 0.9), Keypoint(6, 640.0, 700.0, 0.9)]
    assessment = monocular_assess(keypoints, "left")
    assert assessment.state == "unknown"
    assert not assessment.suspected


def test_low_confidence_keypoints_are_ignored():
    keypoints = [Keypoint(index, 600.0 + index, 700.0, 0.05) for index in range(5, 17)]
    assessment = monocular_assess(keypoints, "left")
    assert assessment.state == "unknown"


def test_scale_gate_rejects_an_impossible_body():
    inflated = {5: np.array([0.0, 2.0, 0.2]), 11: np.array([0.0, 2.9, 0.2])}
    bad = implausible_segment(inflated)
    assert bad is not None
    assert bad[0] == (5, 11)


def test_scale_gate_accepts_a_real_torso():
    ordinary = {5: np.array([0.0, 2.0, 0.2]), 11: np.array([0.0, 2.45, 0.2])}
    assert implausible_segment(ordinary) is None
