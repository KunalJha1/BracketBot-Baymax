import sys

import numpy as np
import pytest

import robot_follow
from follow_perception import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER, PoseDetection


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


def test_left_eye_splits_side_by_side_stereo_only():
    assert robot_follow.left_eye(np.zeros((960, 2560, 3))).shape == (960, 1280, 3)
    assert robot_follow.left_eye(np.zeros((384, 640, 3))).shape == (384, 640, 3)


def test_perceive_turns_detections_into_located_people():
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[NOSE] = (150, 110, 0.9)
    kp[L_SHOULDER], kp[R_SHOULDER] = (180, 150, 0.9), (120, 150, 0.9)
    kp[L_HIP], kp[R_HIP] = (175, 250, 0.9), (125, 250, 0.9)

    class FakeEngine:
        def infer(self, image):
            return [PoseDetection(np.array([100, 90, 200, 380], dtype=np.float32), 0.8, kp)]

    rows, cols = np.meshgrid(np.arange(160, 240, 5), np.arange(130, 170, 5))
    mask = (rows * 640 + cols).ravel()
    points_base = np.tile([-0.1, 1.2, 1.1], (len(mask), 1))  # x=-0.1 is 0.1 m to the left
    image = np.zeros((384, 640, 3), dtype=np.uint8)

    pixels = robot_follow.mask_to_image_pixels(mask, (384, 640), image.shape)
    perception = robot_follow.perceive(FakeEngine(), image, points_base, pixels, 5.0)

    assert perception.t == 5.0
    assert perception.points.shape == (len(mask), 3)
    [person] = perception.people
    assert (person.forward, person.left) == pytest.approx((1.2, 0.1))
    assert person.hand_raised is False
    assert person.hist.shape == (64,)


def test_runner_imports_without_bbos():
    assert "bbos" not in sys.modules
