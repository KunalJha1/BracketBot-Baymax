import numpy as np
import pytest

from bbapps.greeter import fist_target as ft


def to_camera(base):
    """Inverse of the camera.points -> base conversion."""
    base = np.asarray(base, dtype=np.float64)
    return np.column_stack((-base[:, 1], base[:, 0], base[:, 2]))


def blob(centre, size, count, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(centre) + rng.uniform(-0.5, 0.5, (count, 3)) * np.asarray(size)


def person_with_fist(fist=(0.50, -0.05, 1.05)):
    fist = np.asarray(fist)
    hand = blob(fist + [0.04, 0, 0], (0.08, 0.09, 0.09), 300, 1)
    forearm = blob(fist + [0.20, 0, 0], (0.24, 0.07, 0.07), 300, 2)
    torso = blob((fist[0] + 0.40, 0.0, 1.10), (0.10, 0.45, 0.40), 3000, 3)
    floor = blob((0.6, 0.0, 0.0), (1.0, 1.0, 0.02), 500, 4)
    return to_camera(np.vstack((hand, forearm, torso, floor)))


def test_finds_the_front_of_a_fist_held_out_in_front_of_a_body():
    fist = ft.find_offered_fist(person_with_fist())
    assert fist is not None
    assert np.allclose(fist, (0.50, -0.05, 1.05), atol=0.02)


def test_a_body_with_no_fist_held_out_is_not_a_target():
    torso = blob((0.55, 0.0, 1.10), (0.10, 0.45, 0.40), 3000, 3)
    assert ft.find_offered_fist(to_camera(torso)) is None


def test_two_hands_at_the_same_depth_are_ambiguous():
    cloud = np.vstack((person_with_fist(), person_with_fist((0.50, -0.35, 1.05))))
    assert ft.find_offered_fist(cloud) is None


def test_an_empty_scene_has_no_fist():
    assert ft.find_offered_fist(np.zeros((0, 3))) is None


def test_fist_must_hold_still_between_two_looks():
    seen = np.array([0.5, -0.05, 1.05])
    assert ft.stable_fist(seen, None) is None
    assert ft.stable_fist(seen, seen + [0.0, 0.10, 0.0]) is None
    assert np.allclose(ft.stable_fist(seen, seen + [0.0, 0.02, 0.0]), seen + [0.0, 0.01, 0.0])


def test_only_the_fist_is_removed_from_the_obstacle_cloud():
    cloud = person_with_fist()
    target = np.array([0.50, -0.05, 1.05])
    kept = ft.camera_to_base(ft.without_target(cloud, target))
    assert np.linalg.norm(kept - target, axis=1).min() > ft.TARGET_EXCLUSION_METRES
    # The forearm beyond the fist and the torso are still obstacles.
    assert np.count_nonzero(np.abs(kept[:, 0] - 0.75) < 0.05) > 50
    assert len(kept) > 3500


# A six-joint toy arm: enough freedom to move the hand anywhere nearby.
LINKS = (0.30, 0.28, 0.12)


def hand_position(motor):
    yaw, shoulder, elbow, wrist = (2 * np.pi * motor[i] for i in (1, 2, 3, 4))
    reach = (
        LINKS[0] * np.cos(shoulder)
        + LINKS[1] * np.cos(shoulder + elbow)
        + LINKS[2] * np.cos(shoulder + elbow + wrist)
    )
    height = (
        LINKS[0] * np.sin(shoulder)
        + LINKS[1] * np.sin(shoulder + elbow)
        + LINKS[2] * np.sin(shoulder + elbow + wrist)
    )
    return np.array([reach * np.cos(yaw), reach * np.sin(yaw) - 0.12, 0.9 + height])


def recording():
    times = np.linspace(0.0, 3.0, 121)
    out = np.sin(np.pi * times / 3.0) ** 2
    poses = np.zeros((len(times), 8))
    poses[:, 0] = 0.4                       # lift
    poses[:, 2] = -0.20 + 0.22 * out        # shoulder swings up and forward
    poses[:, 3] = 0.30 - 0.22 * out         # elbow straightens
    poses[:, 7] = 0.01                      # gripper
    return times, poses


def recorded_apex(poses, hand_position=hand_position):
    hands = np.stack([hand_position(pose) for pose in poses])
    index = int(np.argmax(hands[:, 0]))
    return index, hands[index]


def test_aimed_bump_peaks_at_the_fist_and_keeps_both_ends():
    times, poses = recording()
    index, apex = recorded_apex(poses)
    fist = apex + [0.05 + ft.STANDOFF_METRES, 0.10, -0.08]

    aimed = ft.retarget_trajectory(hand_position, times, poses, fist)

    assert aimed.apex_index == index
    assert np.allclose(aimed.offset, (0.05, 0.10, -0.08), atol=1e-6)
    assert np.linalg.norm(aimed.apex - (fist - [ft.STANDOFF_METRES, 0, 0])) < 0.005
    assert np.allclose(aimed.trajectory[0], poses[0], atol=1e-6)
    assert np.allclose(aimed.trajectory[-1], poses[-1], atol=1e-6)
    # Lift and gripper are never touched.
    assert np.allclose(aimed.trajectory[:, 0], 0.4)
    assert np.allclose(aimed.trajectory[:, 7], 0.01)
    # No jumps: the bent motion is about as smooth as the recording.
    recorded_step = np.abs(np.diff(poses, axis=0)).max()
    assert np.abs(np.diff(aimed.trajectory, axis=0)).max() < 2.0 * recorded_step + 0.005


def test_a_fist_far_outside_reach_only_moves_the_apex_by_the_allowed_amount():
    times, poses = recording()
    index, apex = recorded_apex(poses)
    aimed = ft.retarget_trajectory(hand_position, times, poses, apex + [ft.STANDOFF_METRES, 1.0, 0.0])
    assert np.all(aimed.offset <= ft.MAX_OFFSET + 1e-9)
    assert aimed.max_joint_delta_turns <= ft.MAX_JOINT_DELTA_TURNS


def lifting_hand_position(motor):
    """The toy arm on a lift: 0.29 m of height per motor turn, like the robot's."""
    return hand_position(motor) + np.array([0.0, 0.0, 0.29 * motor[0]])


def test_a_high_fist_is_reached_with_the_lift_before_the_arm_bends():
    times, poses = recording()
    index, apex = recorded_apex(poses, lifting_hand_position)
    fist = apex + [ft.STANDOFF_METRES, 0.0, 0.05]

    aimed = ft.retarget_trajectory(
        lifting_hand_position, times, poses, fist, lift_range=(0.0, 1.0)
    )

    assert aimed.lift_metres == pytest.approx(0.05, abs=1e-6)
    assert np.allclose(aimed.offset, 0.0, atol=1e-6)
    assert aimed.max_joint_delta_turns < 1e-3
    assert np.linalg.norm(aimed.apex - (fist - [ft.STANDOFF_METRES, 0, 0])) < 0.005
    # The lift rises with the reach and is back where it was parked at both ends.
    assert aimed.trajectory[index, 0] == pytest.approx(0.4 + 0.05 / 0.29, abs=1e-4)
    assert aimed.trajectory[0, 0] == pytest.approx(0.4)
    assert aimed.trajectory[-1, 0] == pytest.approx(0.4)


def test_the_arm_takes_the_height_the_lift_cannot():
    times, poses = recording()
    index, apex = recorded_apex(poses, lifting_hand_position)
    fist = apex + [ft.STANDOFF_METRES, 0.0, 0.10]

    # Only 0.1 turn of travel left above the parked height.
    aimed = ft.retarget_trajectory(
        lifting_hand_position, times, poses, fist, lift_range=(0.0, 0.5)
    )

    assert aimed.lift_turns == pytest.approx(0.1)
    # What is left is the arm's to bend for, as far as it is willing to.
    assert 0.03 < aimed.offset[2] <= 0.10 - 0.029 + 1e-9
    assert aimed.trajectory[:, 0].max() <= 0.5 + 1e-6


def test_the_lift_never_has_to_move_faster_than_its_limit():
    times, poses = recording()
    index, apex = recorded_apex(poses, lifting_hand_position)
    fist = apex + [ft.STANDOFF_METRES, 0.0, 0.28]

    aimed = ft.retarget_trajectory(
        lifting_hand_position, times, poses, fist, lift_range=(-5.0, 5.0), playback_speed=0.6
    )

    lift_speed = np.abs(np.diff(aimed.trajectory[:, 0]) / np.diff(times / 0.6))
    assert lift_speed.max() <= ft.LIFT_PEAK_TURNS_PER_SECOND * 1.01
    assert aimed.lift_turns > 0.3


def test_a_lift_parked_outside_its_range_is_not_dragged_back_in():
    times, poses = recording()
    index, apex = recorded_apex(poses, lifting_hand_position)
    aimed = ft.retarget_trajectory(
        lifting_hand_position, times, poses, apex + [ft.STANDOFF_METRES, 0.0, -0.10],
        lift_range=(0.5, 1.0),
    )
    assert aimed.lift_turns == 0.0
    assert np.allclose(aimed.trajectory[:, 0], 0.4)


def test_the_body_turns_part_of_the_way_and_leaves_the_rest_to_the_arm():
    apex = np.array([0.43, -0.125, 0.84])
    fist = np.array([0.55, 0.12, 1.0])
    full = np.degrees(np.arctan2(0.12, 0.55) - np.arctan2(-0.125, 0.43))

    turn = ft.body_turn_deg(fist, apex)

    assert turn == pytest.approx(ft.BODY_TURN_SHARE * full)
    assert 0 < turn < full                      # left, toward the fist, but not all the way


def test_body_turn_is_skipped_when_small_and_capped_when_large():
    apex = np.array([0.43, -0.125, 0.84])
    assert ft.body_turn_deg(apex + [0.10, 0.02, 0.2], apex) == 0.0
    assert ft.body_turn_deg(np.array([0.10, -0.80, 1.0]), apex) == -ft.MAX_BODY_TURN_DEG


def test_an_untrustworthy_bend_is_refused(monkeypatch):
    times, poses = recording()
    index, apex = recorded_apex(poses)
    monkeypatch.setattr(ft, "MAX_JOINT_DELTA_TURNS", 0.001)
    with pytest.raises(RuntimeError, match="bends a joint"):
        ft.retarget_trajectory(hand_position, times, poses, apex + [0.10, 0.10, 0.0])


def test_the_log_says_why_a_scene_has_no_fist():
    torso = blob((0.55, 0.0, 1.10), (0.10, 0.45, 0.40), 3000, 3)
    assert "too big for a fist" in ft.examine_offered_fist(to_camera(torso))[1] or \
        "fist-sized blob" in ft.examine_offered_fist(to_camera(torso))[1]
    assert "nothing in reach" in ft.examine_offered_fist(np.zeros((0, 3)))[1]
    assert ft.examine_offered_fist(person_with_fist())[1].startswith("fist at")
