import math
import sys

import numpy as np
import pytest

import robot_follow


def test_defaults_are_the_bring_up_limits():
    args = robot_follow.parse_args([])
    assert (args.gap, args.v_max, args.dry_run, args.rotate_only) == (1.0, 0.15, False, False)
    assert robot_follow.loop_config(args).v_max == 0.15


def test_rotate_only_holds_forward_speed_at_zero():
    assert robot_follow.loop_config(robot_follow.parse_args(["--rotate-only"])).v_max == 0.0


@pytest.mark.parametrize("argv", [
    ["--gap", "2.0"], ["--gap", "0.3"], ["--v-max", "0.5"], ["--v-max", "0"], ["--no-heartbeat"],
])
def test_unsafe_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        robot_follow.parse_args(argv)


def test_only_a_dry_run_may_skip_the_heartbeat():
    assert robot_follow.parse_args(["--dry-run", "--no-heartbeat"]).no_heartbeat is True


def test_perceive_finds_a_person_in_a_base_frame_cloud():
    rng = np.random.default_rng(3)
    phi = rng.uniform(-math.pi / 2, math.pi / 2, 900)
    forward = 1.4 - 0.18 * np.cos(phi)  # front half of a person standing 1.4 m ahead
    left = 0.2 + 0.18 * np.sin(phi)  # ... and 0.2 m to the left
    z = rng.uniform(0.05, 1.75, 900)
    points_base = np.column_stack([-left, forward, z])  # base +x points right (BASE_LEFT_SIGN = -1)

    perception = robot_follow.perceive(points_base, 5.0)

    assert perception.t == 5.0
    assert perception.points.shape == (900, 3)
    [person] = perception.people
    assert person.forward == pytest.approx(1.4 - 0.7 * 0.18, abs=0.05)
    assert person.left == pytest.approx(0.2, abs=0.05)
    assert person.hist is None


def test_perceive_with_nobody_there():
    assert robot_follow.perceive(np.empty((0, 3)), 1.0).people == ()


def test_runner_imports_without_bbos():
    assert "bbos" not in sys.modules
