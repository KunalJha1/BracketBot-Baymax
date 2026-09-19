import numpy as np
import pytest

from bbapps.greeter.gesture_safety import (
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


def test_exact_hand_path_rejects_dense_nearby_geometry():
    rng = np.random.default_rng(13)
    path = {"left": np.array([[0.2, -0.3, 1.2], [0.3, -0.3, 1.25]])}
    obstacle_base = rng.normal([0.25, -0.3, 1.22], 0.015, size=(80, 3))
    obstacle_camera = np.column_stack(
        (-obstacle_base[:, 1], obstacle_base[:, 0], obstacle_base[:, 2])
    )

    with pytest.raises(RuntimeError, match="inside the arm clearance zone"):
        depth_clearance(
            np.vstack((clear_depth(), obstacle_camera)), ("left",), path
        )


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
