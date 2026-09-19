import numpy as np
import pytest

from bbapps.greeter.pointing import (
    PersonTargetTracker,
    pointing_goal,
    quaternion_forward_z,
    quaternion_from_z,
    quaternion_slerp,
)


DETECTIONS = [
    (20, 20, 180, 460, 0.91),
    (430, 40, 610, 470, 0.88),
    (245, 160, 385, 460, 0.96),
]


def tracker(now=10.0):
    result = PersonTargetTracker(max_age=1.0)
    result.update(DETECTIONS, 640, 480, observed_at=now)
    return result


def test_primary_target_is_largest_person_not_arbitrary_detection_order():
    target = tracker().select("primary", now=10.5)

    assert (target.x1, target.x2) == (430, 610)
    assert target.arm == "right"


def test_left_and_right_commands_choose_opposite_people():
    targets = tracker()

    assert targets.select("left", now=10.5).x1 == 20
    assert targets.select("right", now=10.5).x1 == 430


def test_stale_or_empty_camera_snapshot_cannot_start_a_point():
    targets = tracker()

    assert targets.select("primary", now=11.01) is None
    targets.update([], 640, 480, observed_at=12.0)
    assert targets.select("primary", now=12.0) is None


def test_pointing_goal_uses_matching_arm_side_and_bounded_reach():
    targets = tracker()
    left_position, left_direction = pointing_goal(
        targets.select("left", now=10.2), shoulder_height=0.9
    )
    right_position, right_direction = pointing_goal(
        targets.select("right", now=10.2), shoulder_height=0.9
    )

    assert left_position[1] > 0
    assert right_position[1] < 0
    assert left_direction[1] > 0
    assert right_direction[1] < 0
    assert np.linalg.norm(left_position[:2]) == pytest.approx(0.34)
    assert np.linalg.norm(right_position[:2]) == pytest.approx(0.34)
    assert 0.55 <= left_position[2] <= 1.30


@pytest.mark.parametrize(
    "direction",
    ([1.0, 0.0, 0.0], [0.8, 0.4, -0.2], [0.8, -0.4, 0.2]),
)
def test_pointing_quaternion_aligns_gripper_axis(direction):
    expected = np.asarray(direction, dtype=float)
    expected /= np.linalg.norm(expected)

    actual = quaternion_forward_z(quaternion_from_z(direction))

    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_invalid_detection_dimensions_and_preferences_are_rejected():
    targets = PersonTargetTracker()

    with pytest.raises(ValueError):
        targets.update([], 0, 480)
    with pytest.raises(ValueError):
        targets.select("nearest")


def test_quaternion_slerp_preserves_endpoints_and_unit_length():
    start = quaternion_from_z([1.0, 0.0, 0.0])
    end = quaternion_from_z([0.8, 0.5, -0.1])

    np.testing.assert_allclose(quaternion_slerp(start, end, 0.0), start)
    np.testing.assert_allclose(quaternion_slerp(start, end, 1.0), end)
    assert np.linalg.norm(quaternion_slerp(start, end, 0.4)) == pytest.approx(1.0)
