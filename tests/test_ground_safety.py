import math
from pathlib import Path
import sys

import numpy as np
import pytest


GREETER_DIR = Path(__file__).parents[1] / "bbapps" / "emotion_greeter"
sys.path.insert(0, str(GREETER_DIR))

from ground_safety import (  # noqa: E402
    GroundAlertTracker,
    Keypoint,
    assess_ground_pose,
    base_to_map,
    keypoints_in_base_frame,
)


def low_person_pose():
    return {
        5: np.array([-0.25, 1.8, 0.22]),
        6: np.array([0.25, 1.8, 0.23]),
        7: np.array([-0.40, 1.55, 0.18]),
        8: np.array([0.40, 1.55, 0.20]),
        11: np.array([-0.20, 1.25, 0.18]),
        12: np.array([0.20, 1.25, 0.20]),
        13: np.array([-0.18, 0.80, 0.14]),
        14: np.array([0.18, 0.80, 0.15]),
        15: np.array([-0.16, 0.35, 0.10]),
        16: np.array([0.16, 0.35, 0.11]),
    }


def test_low_extended_3d_pose_is_possible_person_on_ground():
    assessment = assess_ground_pose(
        low_person_pose(),
        robot_position=np.array([10.0, 20.0, 0.0]),
        robot_yaw=0.0,
    )

    assert assessment.state == "possible_person_on_ground"
    assert assessment.confidence >= 0.62
    assert assessment.torso_height_m == pytest.approx(0.21, abs=0.02)
    assert assessment.map_position is not None
    assert assessment.map_position[0] == pytest.approx(10.0, abs=0.1)
    assert assessment.map_position[1] > 21.0
    for joint in low_person_pose().values():
        assert np.linalg.norm(joint[:2] - np.asarray(assessment.base_position)[:2]) + 0.249 <= assessment.body_radius_m


def test_upright_torso_is_clear_even_with_feet_on_floor():
    pose = low_person_pose()
    pose[5] = np.array([-0.25, 1.4, 1.45])
    pose[6] = np.array([0.25, 1.4, 1.45])
    pose[11] = np.array([-0.20, 1.3, 0.85])
    pose[12] = np.array([0.20, 1.3, 0.85])

    assessment = assess_ground_pose(pose)

    assert assessment.state == "clear"
    assert assessment.torso_height_m > 0.55


def test_missing_torso_depth_is_unknown_not_clear():
    assessment = assess_ground_pose({5: np.zeros(3), 6: np.zeros(3)})

    assert assessment.state == "unknown"
    assert "three torso" in assessment.reason


def test_sparse_pixel_indices_are_associated_to_pose_keypoint():
    keypoints = [Keypoint(5, 5.0, 5.0, 0.9), Keypoint(6, 8.0, 8.0, 0.1)]
    indices = np.array([55, 56, 65])
    points = np.array(
        [[0.0, 1.0, 0.2], [0.02, 1.02, 0.22], [-0.02, 0.98, 0.18]],
        dtype=np.float32,
    )

    result = keypoints_in_base_frame(
        keypoints, indices, points, 10, 10, search_radius_px=1
    )

    np.testing.assert_allclose(result[5], [0.0, 1.0, 0.2], atol=0.021)
    assert 6 not in result


def test_base_to_map_matches_navigation_heading_convention():
    point = np.array([1.0, 2.0, 0.0])  # one metre right, two forward

    assert base_to_map(point, np.array([10.0, 20.0]), 0.0) == pytest.approx(
        (11.0, 22.0)
    )
    assert base_to_map(
        point, np.array([10.0, 20.0]), math.pi / 2
    ) == pytest.approx((8.0, 21.0))


def test_alert_requires_persistence_and_unknown_does_not_clear_it():
    tracker = GroundAlertTracker(hold_seconds=2.0, clear_seconds=1.0)
    suspected = assess_ground_pose(low_person_pose())
    clear_pose = low_person_pose()
    for index in (5, 6, 11, 12):
        clear_pose[index] = clear_pose[index] + np.array([0.0, 0.0, 1.0])
    clear = assess_ground_pose(clear_pose)
    unknown = assess_ground_pose({})

    assert tracker.update({7: suspected}, 0.0)[7] == "checking"
    assert tracker.update({7: suspected}, 1.9)[7] == "checking"
    assert tracker.update({7: suspected}, 2.0)[7] == "alert"
    assert tracker.update({7: unknown}, 20.0)[7] == "alert"
    assert tracker.update({7: clear}, 21.0)[7] == "alert"
    assert tracker.update({7: clear}, 22.0)[7] == "clear"


@pytest.mark.parametrize("missing", [True, False])
def test_detection_gap_restarts_confirmation_hold(missing):
    tracker = GroundAlertTracker(hold_seconds=2)
    low = assess_ground_pose(low_person_pose())
    tracker.update({7: low}, 0)
    tracker.update({} if missing else {7: assess_ground_pose({})}, 1)
    assert tracker.update({7: low}, 3)[7] == "checking"
    assert tracker.update({7: low}, 5)[7] == "alert"
