import numpy as np
import pytest

from bbapps.greeter.gesture_safety import (
    HARD_COLLISION_RADIUS_METRES,
    PROFILES,
    ClearanceProfile,
    active_profile,
    depth_clearance,
    plan_recorded_gesture,
    spoken_safety_refusal,
)


def frames():
    still = [0.0] * 8
    moved = [0.0, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        {"t": 0.0, "left": still, "right": still},
        {"t": 1.0, "left": moved, "right": still},
    ]


def clear_depth(seed=3):
    rng = np.random.default_rng(seed)
    # Camera frame (right, forward, height), safely beyond arm reach.
    return np.column_stack(
        (
            rng.uniform(-0.5, 0.5, 400),
            rng.uniform(1.2, 1.8, 400),
            rng.uniform(0.5, 1.5, 400),
        )
    )


def test_plan_preserves_lift_and_ignores_inactive_arm():
    recording = frames()
    recording[0]["left"][0] = 0.7
    recording[1]["left"][0] = 0.7
    start = np.array([0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    plan = plan_recorded_gesture(
        recording,
        {"left": start},
        [0.0, 0.0, 0.0],
        clear_depth(),
    )

    assert plan.sides == ("left",)
    assert np.all(plan.poses["left"][:, 0] == pytest.approx(0.21))
    assert "right" not in plan.poses


def test_plan_rejects_robot_that_is_not_upright():
    with pytest.raises(RuntimeError, match="not upright"):
        plan_recorded_gesture(
            frames(),
            {"left": np.zeros(8)},
            [26.0, 0.0, 0.0],
            clear_depth(),
        )


def test_plan_rejects_large_entry_move():
    recording = frames()
    recording[0]["left"][2] = 0.7

    with pytest.raises(RuntimeError, match="entry move"):
        plan_recorded_gesture(
            recording,
            {"left": np.zeros(8)},
            [0.0, 0.0, 0.0],
            clear_depth(),
        )


def test_plan_rejects_obstacle_in_active_arm_clearance_zone():
    rng = np.random.default_rng(9)
    # Base frame target: forward=.45, left=.35, height=1.0. Convert to
    # camera.points frame: right=-left, forward, height.
    obstacle = np.column_stack(
        (
            rng.normal(-0.35, 0.01, 80),
            rng.normal(0.45, 0.01, 80),
            rng.normal(1.0, 0.01, 80),
        )
    )

    with pytest.raises(RuntimeError, match="inside the arm clearance zone"):
        plan_recorded_gesture(
            frames(),
            {"left": np.zeros(8)},
            [0.0, 0.0, 0.0],
            np.vstack((clear_depth(), obstacle)),
        )


def test_plan_fails_closed_when_depth_is_unavailable():
    with pytest.raises(RuntimeError, match="surroundings check unavailable"):
        plan_recorded_gesture(
            frames(),
            {"left": np.zeros(8)},
            [0.0, 0.0, 0.0],
            np.zeros((10, 3)),
        )


def test_exact_hand_path_ignores_geometry_elsewhere_in_broad_workspace():
    rng = np.random.default_rng(12)
    table = np.column_stack(
        (
            rng.normal(-0.35, 0.01, 100),
            rng.normal(0.45, 0.01, 100),
            rng.normal(0.75, 0.01, 100),
        )
    )
    path = {"left": np.array([[0.2, -0.3, 1.2], [0.3, -0.3, 1.25]])}

    result = depth_clearance(
        np.vstack((clear_depth(), table)), ("left",), path
    )

    assert result["left"] == 0


def to_camera(points):
    """Base frame (forward, left, height) into camera frame (right, forward, height)."""
    return np.column_stack((-points[:, 1], points[:, 0], points[:, 2]))


def test_exact_hand_path_rejects_dense_nearby_geometry():
    rng = np.random.default_rng(13)
    path = {"left": np.array([[0.2, -0.3, 1.2], [0.3, -0.3, 1.25]])}
    obstacle = to_camera(rng.normal([0.25, -0.3, 1.22], 0.015, size=(80, 3)))

    # This obstacle's centroid sits ~54 mm off the nearest waypoint: inside the
    # cautious 60 mm bubble, outside the balanced 45 mm one. It is the exact
    # case the looser profile is meant to stop refusing.
    with pytest.raises(RuntimeError, match="inside the arm clearance zone"):
        depth_clearance(
            np.vstack((clear_depth(), obstacle)),
            ("left",),
            path,
            profile=PROFILES["cautious"],
        )

    counts = depth_clearance(
        np.vstack((clear_depth(), obstacle)),
        ("left",),
        path,
        profile=PROFILES["balanced"],
    )
    assert counts["left"] < PROFILES["balanced"].min_blocking_points


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_obstacle_on_the_hand_path_blocks_under_every_profile(name):
    # Whatever the padding, geometry the hand actually passes through is a
    # collision, and the hard gate must catch it even at the boldest setting.
    rng = np.random.default_rng(21)
    path = {"left": np.array([[0.2, -0.3, 1.2], [0.3, -0.3, 1.25]])}
    obstacle = to_camera(rng.normal([0.2, -0.3, 1.2], 0.01, size=(120, 3)))

    with pytest.raises(RuntimeError, match="inside the arm clearance zone"):
        depth_clearance(
            np.vstack((clear_depth(), obstacle)),
            ("left",),
            path,
            profile=PROFILES[name],
        )


def test_no_profile_may_reach_inside_the_hard_collision_radius():
    with pytest.raises(ValueError, match="hard collision radius"):
        ClearanceProfile("reckless", HARD_COLLISION_RADIUS_METRES - 0.001, 200, 40)

    for profile in PROFILES.values():
        assert profile.hand_path_clearance_m >= HARD_COLLISION_RADIUS_METRES


def test_broad_workspace_fallback_keeps_the_conservative_point_budget():
    # With no hand path there is no hard gate to backstop a looser budget, so
    # the fallback must not follow the profile down.
    rng = np.random.default_rng(9)
    obstacle = np.column_stack(
        (
            rng.normal(-0.35, 0.01, 70),
            rng.normal(0.45, 0.01, 70),
            rng.normal(1.0, 0.01, 70),
        )
    )

    with pytest.raises(RuntimeError, match="inside the arm clearance zone"):
        depth_clearance(
            np.vstack((clear_depth(), obstacle)),
            ("left",),
            None,
            profile=PROFILES["bold"],
        )


def test_risk_profile_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("BAYMAX_GESTURE_RISK", "bold")
    assert active_profile() is PROFILES["bold"]

    monkeypatch.setenv("BAYMAX_GESTURE_RISK", " Cautious ")
    assert active_profile() is PROFILES["cautious"]

    monkeypatch.setenv("BAYMAX_GESTURE_RISK", "yolo")
    assert active_profile() is PROFILES["balanced"]

    monkeypatch.delenv("BAYMAX_GESTURE_RISK")
    assert active_profile() is PROFILES["balanced"]


def test_spoken_safety_refusal_hides_depth_diagnostics():
    diagnostic = (
        "gesture blocked because something is inside the arm clearance zone: "
        "left arm (4093 depth points; path_min=[0.1, -0.4, 0.5])"
    )

    reply = spoken_safety_refusal("wave", diagnostic)

    assert reply == (
        "I can't wave; there isn't enough clearance. Please step back a little."
    )
    assert "depth" not in reply
    assert "4093" not in reply


def test_goodbye_safety_refusal_describes_the_wave():
    assert spoken_safety_refusal("goodbye", "robot is not upright") == (
        "I can't wave goodbye while I'm not upright."
    )


def _recording(name):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    folder = "bbapps/mimic/recordings" if name == "dance" else "bbapps/greeter/movements"
    return json.loads((root / folder / f"{name}.json").read_text())


def test_trim_idle_drops_dance_dead_time_but_keeps_the_motion():
    from bbapps.greeter.gesture_safety import trajectory_arrays, trim_idle

    times, poses = trajectory_arrays(_recording("dance"))
    trimmed_times, trimmed = trim_idle(times, poses)

    assert times[-1] - trimmed_times[-1] > 3.5
    assert trimmed_times[0] == 0.0
    for side in ("left", "right"):
        assert np.allclose(
            np.ptp(trimmed[side][:, 1:7], axis=0),
            np.ptp(poses[side][:, 1:7], axis=0),
            atol=0.01,
        )


def test_trim_idle_keeps_the_namaste_hold():
    from bbapps.greeter.gesture_safety import trajectory_arrays, trim_idle

    times, poses = trajectory_arrays(_recording("namaste"))
    trimmed_times, trimmed = trim_idle(times, poses)

    assert times[-1] - trimmed_times[-1] < 0.2
    assert np.array_equal(trimmed["left"][-1], poses["left"][-1])


def test_ease_seconds_scales_with_distance_within_bounds():
    from bbapps.greeter.gesture_safety import MAX_EASE_SECONDS, MIN_EASE_SECONDS, ease_seconds

    start = {"left": np.zeros(8, dtype=np.float32)}
    assert ease_seconds(start, {"left": np.full(8, 0.02)}) == MIN_EASE_SECONDS
    assert ease_seconds(start, {"left": np.full(8, 0.2)}) == pytest.approx(1.0)
    assert ease_seconds(start, {"left": np.full(8, 0.9)}) == MAX_EASE_SECONDS


def test_contact_gestures_keep_the_slow_playback_speed():
    from bbapps.greeter.gesture_safety import playback_speed

    for name in ("handshake", "fist bump", "hug", "dance"):
        assert playback_speed(name) == 0.6
    assert playback_speed("wave") == playback_speed("goodbye") == 0.75


def test_pose_at_blends_between_recorded_frames():
    from bbapps.greeter.gesture_safety import pose_at

    times = np.asarray([0.0, 1.0, 3.0])
    trajectory = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0]])

    assert pose_at(times, trajectory, -1.0) == pytest.approx([0.0, 0.0])
    assert pose_at(times, trajectory, 0.25) == pytest.approx([0.25, 0.5])
    assert pose_at(times, trajectory, 1.0) == pytest.approx([1.0, 2.0])
    assert pose_at(times, trajectory, 2.0) == pytest.approx([2.0, 2.0])
    assert pose_at(times, trajectory, 9.0) == pytest.approx([3.0, 2.0])


def test_hug_is_smooth_and_returns_to_its_resting_pose():
    import json
    from pathlib import Path

    from bbapps.greeter.gesture_safety import trajectory_arrays

    movements = Path(__file__).resolve().parents[1] / "bbapps" / "greeter" / "movements"
    times, poses = trajectory_arrays(json.loads((movements / "hug.json").read_text()))
    for side in ("left", "right"):
        velocity = np.diff(poses[side], axis=0) / np.diff(times)[:, None]
        assert np.abs(velocity).max() < 0.3
        # No frame-to-frame jerk: the old recording stepped by whole encoder ticks.
        assert np.abs(np.diff(velocity, axis=0)).max() < 0.03
        assert np.abs(poses[side]).max() < 0.30
        assert poses[side][-1] == pytest.approx(poses[side][0], abs=0.005)
