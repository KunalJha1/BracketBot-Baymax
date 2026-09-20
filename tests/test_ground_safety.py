import math
from pathlib import Path
import sys

import numpy as np
import pytest


GREETER_DIR = Path(__file__).parents[1] / "bbapps" / "emotion_greeter"
sys.path.insert(0, str(GREETER_DIR))

from ground_safety import (  # noqa: E402
    box_position_in_base_frame,
    GroundAlertTracker,
    Keypoint,
    RECT_PROJECTION,
    assess_ground_pose,
    assess_ground_pose_monocular,
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


def sitting_on_floor_pose():
    """Legs out along the floor, trunk upright: low hips and a long footprint."""
    pose = low_person_pose()
    pose[5] = np.array([-0.20, 1.30, 0.60])
    pose[6] = np.array([0.20, 1.30, 0.61])
    pose[7] = np.array([-0.30, 1.20, 0.35])
    pose[8] = np.array([0.30, 1.20, 0.36])
    pose[11] = np.array([-0.18, 1.25, 0.12])
    pose[12] = np.array([0.18, 1.25, 0.13])
    return pose


def test_sitting_on_the_floor_is_not_lying_down():
    assessment = assess_ground_pose(sitting_on_floor_pose())

    assert assessment.state == "clear"
    assert "not lying" in assessment.reason


def test_raised_head_keeps_a_low_pose_clear_and_a_low_head_does_not():
    pose = low_person_pose()
    pose[0] = np.array([0.0, 2.0, 0.85])
    assert assess_ground_pose(pose).state == "clear"
    pose[0] = np.array([0.0, 2.0, 0.20])
    assert assess_ground_pose(pose).state == "possible_person_on_ground"


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


def test_box_position_finds_a_person_from_behind_and_ignores_the_wall_behind_them():
    width, height = 64, 48
    ys, xs = np.mgrid[0:height, 0:width]
    indices = (ys * width + xs).reshape(-1)
    points = np.zeros((width * height, 3), dtype=np.float32)
    points[:, 1] = 3.0                                  # a wall fills the frame
    person = ((xs >= 24) & (xs <= 40) & (ys >= 8) & (ys <= 44)).reshape(-1)
    points[person] = (0.2, 1.4, 1.0)                    # no pose joints needed
    position = box_position_in_base_frame((20, 6, 44, 46), indices, points, width)
    assert position == pytest.approx((0.2, 1.4, 1.0), abs=1e-5)
    assert box_position_in_base_frame((20, 6, 44, 46), indices[:10], points[:10], width) is None


def project(pose):
    """Pose in the base frame -> camera.rect keypoints, through the fitted camera."""
    keypoints = []
    for index, point in pose.items():
        u, v, w = RECT_PROJECTION @ np.append(point, 1.0)
        keypoints.append(Keypoint(index, u / w, v / w, 0.9))
    return keypoints


def lying_pose(forward, heading_deg, side=0.0):
    """A 1.7 m adult flat on the floor, body axis turned ``heading_deg`` from the view."""
    along = np.array([math.sin(math.radians(heading_deg)), math.cos(math.radians(heading_deg)), 0.0])
    across = np.array([along[1], -along[0], 0.0])
    origin = np.array([side, forward, 0.12])
    layout = {5: (1.40, -0.2), 6: (1.40, 0.2), 7: (1.15, -0.32), 8: (1.15, 0.32),
              11: (0.90, -0.15), 12: (0.90, 0.15), 13: (0.45, -0.15), 14: (0.45, 0.15),
              15: (0.0, -0.13), 16: (0.0, 0.13), 0: (1.62, 0.0)}
    return {i: origin + a * along + c * across for i, (a, c) in layout.items()}


def upright_pose(forward, hip=0.92, shoulder=1.42, knee=0.48, knee_forward=0.0):
    pose = {}
    for side, x in ((0, -0.18), (1, 0.18)):
        pose[5 + side] = np.array([x, forward, shoulder])
        pose[7 + side] = np.array([x * 1.3, forward, shoulder - 0.3])
        pose[11 + side] = np.array([x * 0.8, forward, hip])
        pose[13 + side] = np.array([x * 0.8, forward - knee_forward, knee])
        pose[15 + side] = np.array([x * 0.8, forward - knee_forward, 0.08])
    pose[0] = np.array([0.0, forward, shoulder + 0.22])
    return pose


@pytest.mark.parametrize("heading", [0, 45, 90, 135])
@pytest.mark.parametrize("forward", [1.8, 2.5, 3.5])
def test_monocular_finds_a_person_lying_beyond_the_depth_cloud(forward, heading):
    assessment = assess_ground_pose_monocular(project(lying_pose(forward, heading)), 512, 384)

    assert assessment.state == "possible_person_on_ground", assessment.reason
    assert assessment.confidence >= 0.62
    assert assessment.reason.startswith("mono: ")
    # The approach steers at this, so it has to be where the body really is.
    centre = np.median([v for i, v in lying_pose(forward, heading).items() if i >= 5], axis=0)
    assert np.linalg.norm(np.asarray(assessment.base_position)[:2] - centre[:2]) < 0.15


@pytest.mark.parametrize("pose", [
    upright_pose(1.5), upright_pose(3.0), upright_pose(5.0),
    upright_pose(2.5, hip=0.50, shoulder=1.00, knee=0.50, knee_forward=0.4),   # on a chair
    upright_pose(2.5, hip=0.12, shoulder=0.62, knee=0.12, knee_forward=0.45),  # sitting on the floor
    upright_pose(2.0, hip=0.35, shoulder=0.75, knee=0.45, knee_forward=0.2),   # crouching
])
def test_monocular_never_calls_a_raised_body_lying(pose):
    assessment = assess_ground_pose_monocular(project(pose), 512, 384)

    assert assessment.state != "possible_person_on_ground", assessment.reason


def test_monocular_needs_the_fitted_camera_and_enough_joints():
    keypoints = project(lying_pose(2.5, 90))
    assert assess_ground_pose_monocular(keypoints, 640, 480).state == "unknown"
    assert assess_ground_pose_monocular(keypoints[:4], 512, 384).state == "unknown"


def test_confirmation_survives_a_dropped_frame_but_a_vanished_alert_expires():
    tracker = GroundAlertTracker(hold_seconds=2.0, gap_seconds=0.7, lost_seconds=15.0)
    low = assess_ground_pose(low_person_pose())

    tracker.update({7: low}, 0.0)
    tracker.update({}, 0.4)                              # detector dropped one frame
    tracker.update({7: assess_ground_pose({})}, 0.6)     # then saw too few joints
    assert tracker.update({7: low}, 1.0)[7] == "checking"
    assert tracker.update({7: low}, 2.0)[7] == "alert"   # hold ran from t=0
    assert tracker.update({}, 16.0) == {7: "alert"}
    assert tracker.update({}, 17.5) == {}
