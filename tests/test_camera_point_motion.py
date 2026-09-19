import numpy as np
import pytest

from scripts.camera_point_motion import (
    motor_command_path,
    pointing_goal,
    quaternion_from_z,
    quaternion_slerp,
)


def test_point_goal_raises_high_camera_target_into_pointing_band():
    side, position, _ = pointing_goal(
        x_offset=0.7,
        y_offset=-0.4,
        current_hand_height=0.43,
        shoulder_height=1.265,
    )

    assert side == "right"
    assert position[1] < 0
    assert position[2] == pytest.approx(0.995)
    assert np.linalg.norm(position[:2]) == pytest.approx(0.30)


def test_point_goal_uses_left_arm_for_image_left():
    side, position, _ = pointing_goal(-0.7, 0.5, 0.40, 1.265)

    assert side == "left"
    assert position[1] > 0


def test_point_goal_tracks_camera_height_and_caps_total_lift():
    _, high, _ = pointing_goal(0.2, -1.0, 0.43, 1.265)
    _, low, _ = pointing_goal(0.2, 1.0, 0.43, 1.265)
    _, capped, _ = pointing_goal(0.2, -1.0, 0.20, 1.265)

    assert high[2] > low[2]
    assert high[2] <= 1.265 - 0.20
    assert capped[2] == pytest.approx(0.90)


def test_motor_command_path_is_continuous_and_preserves_endpoints():
    start = np.array([0.88, -0.04, 0.01])
    goal = np.array([0.0, 0.06, -0.20])

    path = np.asarray(motor_command_path(start, goal, samples=20))

    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    expected_step = np.tile((goal - start) / 20, (20, 1))
    np.testing.assert_allclose(np.diff(path, axis=0), expected_step)


def test_point_quaternion_interpolation_stays_normalized():
    start = quaternion_from_z([1.0, 0.0, 0.0])
    end = quaternion_from_z([0.8, 0.4, -0.2])

    middle = quaternion_slerp(start, end, 0.5)

    np.testing.assert_allclose(np.linalg.norm(middle), 1.0)
