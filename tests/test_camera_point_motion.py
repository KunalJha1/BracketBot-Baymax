import numpy as np
import pytest

from scripts.camera_point_motion import (
    aim_error_degrees,
    motor_command_path,
    pointing_goal,
    quaternion_from_z,
    quaternion_slerp,
)


def test_point_goal_raises_high_target_into_pointing_band_and_preserves_aim():
    target = np.array([2.0, -0.7, 1.6])
    side, position, direction, quaternion = pointing_goal(
        target=target,
        current_hand_height=0.43,
        shoulder_height=1.265,
    )

    assert side == "right"
    assert position[1] < 0
    assert 1.265 - 0.60 <= position[2] <= 1.265 - 0.20
    expected = target - position
    np.testing.assert_allclose(direction, expected / np.linalg.norm(expected))
    assert aim_error_degrees(quaternion, direction) < 1e-5


def test_point_goal_uses_left_arm_for_target_on_robot_left():
    side, position, _, _ = pointing_goal([2.0, 0.7, 1.0], 0.40, 1.265)

    assert side == "left"
    assert position[1] > 0


def test_point_goal_tracks_camera_height_and_caps_total_lift():
    _, high, _, _ = pointing_goal([2.0, -0.2, 1.8], 0.43, 1.265)
    _, low, _, _ = pointing_goal([2.0, -0.2, -5.0], 0.43, 1.265)
    _, capped, _, _ = pointing_goal([2.0, -0.2, 1.8], 0.20, 1.265)

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
