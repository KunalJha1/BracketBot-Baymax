import numpy as np
import pytest

from scripts.table_rest import (
    MAX_LIFT_COMMAND_STEP_TURNS,
    MAX_ROTARY_COMMAND_STEP_TURNS,
    densify_synchronized_paths,
    detect_table_plane,
    nonsuppressing,
    validate_playback_clearance,
)


def tabletop(seed=7, height=0.74, count=2500):
    rng = np.random.default_rng(seed)
    arm_frame = np.column_stack(
        (
            rng.uniform(0.24, 0.72, count),
            rng.uniform(-0.48, 0.48, count),
            rng.normal(height, 0.003, count),
        )
    )
    # camera.points is lateral/right, forward, height. Arm IK is
    # forward, lateral/left, height.
    return np.column_stack((-arm_frame[:, 1], arm_frame[:, 0], arm_frame[:, 2]))


def test_detect_table_plane_adapts_height_and_hand_distance():
    noise = np.array([[0.1, 0.0, 0.0], [2.0, 2.0, 2.0], [np.nan, 0.0, 0.7]])
    result = detect_table_plane(np.vstack((tabletop(height=0.78), noise)))

    assert result["height"] == pytest.approx(0.78, abs=0.006)
    assert 0.30 <= result["hand_x"] <= 0.48
    assert min(result["hand_support"]) >= 10
    assert result["flatness"] < 0.006


def test_detect_table_plane_rejects_surface_too_narrow_for_two_hands():
    points = tabletop()
    points[:, 0] = np.linspace(-0.08, 0.08, len(points))

    with pytest.raises(RuntimeError, match="supports both hand positions"):
        detect_table_plane(points)


def test_detect_table_plane_ignores_floor_and_out_of_reach_surfaces():
    rng = np.random.default_rng(2)
    floor = np.column_stack(
        (
            rng.uniform(-0.6, 0.6, 2000),
            rng.uniform(0.2, 0.8, 2000),
            rng.normal(0.02, 0.003, 2000),
        )
    )

    with pytest.raises(RuntimeError, match="not enough depth points"):
        detect_table_plane(floor)


def test_rejected_plane_logs_numeric_gate_evidence():
    points = tabletop()
    points[:, 0] = np.linspace(-0.08, 0.08, len(points))
    lines = []

    with pytest.raises(RuntimeError, match="supports both hand positions"):
        detect_table_plane(
            points,
            logger=lambda stage, message: lines.append(f"{stage}: {message}"),
        )

    assert any(line.startswith("depth: cloud shape=") for line in lines)
    assert any(line.startswith("transform: arm_xyz=") for line in lines)
    assert any(line.startswith("filter: workspace") for line in lines)
    assert any(line.startswith("bins: top=") for line in lines)
    assert any("reject=small-span" in line for line in lines)


def test_bbos_context_cannot_suppress_runner_failure():
    class SuppressingManager:
        def __enter__(self):
            return "reader"

        def __exit__(self, *_):
            return True

    with pytest.raises(RuntimeError, match="visible failure"):
        with nonsuppressing(SuppressingManager()) as value:
            assert value == "reader"
            raise RuntimeError("visible failure")


def test_densify_keeps_both_arms_synchronized_and_bounds_motor_steps():
    left_start = np.zeros(8)
    right_start = np.zeros(8)
    left_end = np.array([0.13, 0.05, 0, 0, 0, 0, 0, 0])
    right_end = np.array([-0.02, 0, 0, 0.07, 0, 0, 0, 0])

    dense = densify_synchronized_paths(
        {
            "left": np.stack((left_start, left_end)),
            "right": np.stack((right_start, right_end)),
        },
        logger=lambda *_: None,
    )

    assert len(dense["left"]) == len(dense["right"])
    assert len(dense["left"]) > 2
    for path in dense.values():
        delta = np.abs(np.diff(path, axis=0))
        assert np.max(delta[:, 0]) <= MAX_LIFT_COMMAND_STEP_TURNS + 1e-12
        assert np.max(delta[:, 1:7]) <= MAX_ROTARY_COMMAND_STEP_TURNS + 1e-12


def test_final_playback_clearance_checks_inserted_motor_poses():
    class FakeIK:
        @staticmethod
        def fk(values):
            return [values[0], 0.0, values[1]], [0.0, 0.0, 0.0, 1.0]

    class FakeConfig:
        ik = FakeIK()

        @staticmethod
        def q2urdf(pose):
            return pose

    path = np.zeros((3, 8))
    path[:, 0] = [0.10, 0.25, 0.30]
    path[:, 1] = [0.50, 0.70, 0.80]
    observation = {"near_edge": 0.24, "height": 0.70}

    with pytest.raises(RuntimeError, match="interpolated playback"):
        validate_playback_clearance(
            path, FakeConfig(), observation, "left", logger=lambda *_: None
        )
