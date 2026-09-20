from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

GREETER_DIR = Path(__file__).parents[1] / "bbapps" / "emotion_greeter"
sys.path.insert(0, str(GREETER_DIR))

from floor_roi import (  # noqa: E402
    FLOOR_ROI, RAW_CAMERA_MATRIX, RAW_FISHEYE_D, RAW_TO_RECT,
    crop_pixels_to_rect, floor_crop, left_eye, raw_pixels_to_rect,
)
from ground_safety import Keypoint  # noqa: E402
from main import (  # noqa: E402
    Detection, TrackedDetection, detections_from_floor_crop, merge_detections, next_view_focus,
)


def test_raw_pixels_follow_the_fisheye_model_into_the_rect_image():
    normalised = np.array([[[0.0, 0.0]], [[0.3, -0.4]], [[-0.5, 0.2]]])
    raw = cv2.fisheye.distortPoints(normalised, RAW_CAMERA_MATRIX, RAW_FISHEYE_D).reshape(-1, 2)
    expected = np.column_stack([normalised.reshape(-1, 2), np.ones(3)]) @ RAW_TO_RECT.T
    expected = expected[:, :2] / expected[:, 2:3]

    assert raw_pixels_to_rect(raw) == pytest.approx(expected, abs=0.05)
    assert raw_pixels_to_rect(np.empty((0, 2))).shape == (0, 2)


def test_the_floor_crop_lands_inside_the_rect_image():
    x0, y0, x1, y1 = FLOOR_ROI
    eye = left_eye(np.zeros((960, 2560, 3), dtype=np.uint8))
    assert eye.shape == (960, 1280, 3)
    assert floor_crop(eye).shape == (y1 - y0, x1 - x0, 3)
    corners = crop_pixels_to_rect([(0, 0), (x1 - x0, 0), (0, y1 - y0), (x1 - x0, y1 - y0)])
    assert (corners[:, 0] > 0).all() and (corners[:, 0] < 512).all()
    assert (corners[:, 1] > 0).all() and (corners[:, 1] < 384).all()
    assert crop_pixels_to_rect([(0, 0)]) == pytest.approx(raw_pixels_to_rect([(x0, y0)]))


def test_crop_detections_move_into_rect_pixels_with_their_keypoints():
    detection = Detection(100, 100, 300, 200, 0.8, (Keypoint(5, 150.0, 120.0, 0.9), Keypoint(11, 250.0, 180.0, 0.7)))

    (moved,) = detections_from_floor_crop([detection], 512, 384)

    points = crop_pixels_to_rect([(150, 120), (250, 180)])
    assert [(kp.index, kp.confidence) for kp in moved.keypoints] == [(5, 0.9), (11, 0.7)]
    assert [(kp.x, kp.y) for kp in moved.keypoints] == pytest.approx([tuple(p) for p in points])
    assert moved.x1 <= min(points[:, 0]) and moved.x2 >= max(points[:, 0])
    assert moved.y1 <= min(points[:, 1]) and moved.y2 >= max(points[:, 1])
    assert moved.confidence == 0.8


def person(x1, joints):
    keypoints = tuple(Keypoint(5 + i, float(x1 + 5), 50.0 + i, 0.9) for i in range(joints))
    return Detection(x1, 40, x1 + 60, 140, 0.8, keypoints)


def test_merge_adds_people_only_the_crop_saw_and_keeps_the_fuller_pose():
    seen_by_both_poorly, only_rect = person(100, 2), person(300, 8)
    same_person_full_pose, only_crop = person(104, 9), person(420, 7)

    merged = merge_detections([seen_by_both_poorly, only_rect], [same_person_full_pose, only_crop])

    assert merged == [same_person_full_pose, only_rect, only_crop]
    assert merge_detections([only_rect], [person(302, 3)]) == [only_rect]



def test_view_focus_follows_whichever_view_sees_the_low_person():
    near, far = person(100, 8), person(300, 8)
    tracked = [TrackedDetection(1, near), TrackedDetection(2, far)]

    assert next_view_focus(None, 0, {1: "clear", 2: "clear"}, tracked, [far]) == (None, 0)
    assert next_view_focus(None, 0, {2: "checking"}, tracked, [far]) == ("floor", 0)
    assert next_view_focus("floor", 0, {1: "alert"}, tracked, [far]) == ("rect", 0)
    # Robot drew near: the body left the crop. One miss waits, the second switches view.
    assert next_view_focus("floor", 0, {2: "alert"}, [], []) == ("floor", 1)
    assert next_view_focus("floor", 1, {2: "alert"}, [], []) == ("rect", 0)
    assert next_view_focus("rect", 1, {2: "alert"}, [], []) == ("floor", 0)
