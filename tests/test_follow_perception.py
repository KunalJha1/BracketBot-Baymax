from pathlib import Path

import numpy as np
import pytest

from follow_core import hist_distance
from follow_perception import (
    L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST, PAD_VALUE,
    Letterbox, PoseDetection, base_to_local, decode_pose, hand_raised, letterbox,
    mask_to_image_pixels, person_position, torso_histogram, torso_rect,
)

ROOT = Path(__file__).resolve().parents[1]


def detection(box=(100, 50, 300, 450), points=None):
    """points: keypoint index -> (x, y, conf); every other keypoint has confidence 0."""
    kp = np.zeros((17, 3), dtype=np.float32)
    for index, value in (points or {}).items():
        kp[index] = value
    return PoseDetection(np.array(box, dtype=np.float32), 0.9, kp)


def upright_person(overrides=None):
    points = {NOSE: (200, 100, 0.9), L_SHOULDER: (240, 150, 0.9), R_SHOULDER: (160, 150, 0.9),
              L_HIP: (230, 280, 0.9), R_HIP: (170, 280, 0.9), L_WRIST: (250, 300, 0.9),
              R_WRIST: (150, 300, 0.9)}
    points.update(overrides or {})
    return detection(points=points)


def test_letterbox_scales_and_pads_the_head_eye():
    image = np.zeros((960, 1280, 3), dtype=np.uint8)
    tensor, lb = letterbox(image, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert lb == Letterbox(0.5, 0, 80)
    assert tensor[0, 0, 0, 0] == pytest.approx(PAD_VALUE / 255)
    assert tensor[0, 0, 320, 320] == 0.0


def test_decode_filters_suppresses_and_maps_back_to_image_pixels():
    raw = np.zeros((1, 56, 8400), dtype=np.float32)
    raw[0, :5, 0] = (320, 320, 100, 200, 0.9)  # kept
    raw[0, :5, 1] = (322, 318, 100, 200, 0.8)  # overlaps the first: suppressed
    raw[0, :5, 2] = (100, 400, 50, 100, 0.3)  # below the confidence threshold
    raw[0, 5:8, 0] = (320, 250, 0.95)  # nose
    dets = decode_pose(raw, Letterbox(0.5, 0, 80))
    assert len(dets) == 1
    assert dets[0].score == pytest.approx(0.9)
    assert dets[0].box == pytest.approx([540, 280, 740, 680])
    assert dets[0].keypoints[NOSE] == pytest.approx([640, 340, 0.95])


def test_decode_with_no_confident_boxes_is_empty():
    assert decode_pose(np.zeros((1, 56, 8400), dtype=np.float32), Letterbox(1.0, 0, 0)) == []


def test_hand_raised_needs_a_confident_wrist_well_above_the_nose():
    assert hand_raised(upright_person({L_WRIST: (250, 40, 0.9)})) is True
    assert hand_raised(upright_person({R_WRIST: (150, 80, 0.9)})) is False  # only 20 px, margin is 40
    assert hand_raised(upright_person({L_WRIST: (250, 40, 0.3)})) is False
    assert hand_raised(upright_person({L_WRIST: (250, 40, 0.9), NOSE: (200, 100, 0.2)})) is False
    assert hand_raised(upright_person()) is False


def test_torso_rect_from_keypoints_with_fallback_and_minimum_width():
    assert torso_rect(upright_person()) == pytest.approx((160, 150, 240, 280))
    assert torso_rect(detection()) == pytest.approx((100 + 200 / 3, 50 + 400 / 3, 300 - 200 / 3, 450 - 400 / 3))
    side_on = upright_person({L_SHOULDER: (202, 150, 0.9), R_SHOULDER: (198, 150, 0.9),
                              L_HIP: (201, 280, 0.9), R_HIP: (199, 280, 0.9)})
    x1, _, x2, _ = torso_rect(side_on)
    assert x2 - x1 == pytest.approx(50)


def test_torso_histogram_separates_clothing_colours():
    image = np.zeros((100, 300, 3), dtype=np.uint8)
    image[:, :100] = (200, 30, 30)  # red
    image[:, 100:200] = (30, 30, 200)  # blue
    image[:, 200:] = (245, 245, 245)  # white; black is the zero default elsewhere
    red = torso_histogram(image, (0, 0, 100, 100))
    blue = torso_histogram(image, (100, 0, 200, 100))
    white = torso_histogram(image, (200, 0, 300, 100))
    black = torso_histogram(np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 10, 10))
    assert red.shape == (64,)
    assert red.sum() == pytest.approx(1.0)
    assert hist_distance(red, blue) > 0.9
    assert hist_distance(white, black) > 0.9
    assert hist_distance(red, torso_histogram(image, (10, 10, 90, 90))) == pytest.approx(0.0)
    assert torso_histogram(image, (400, 0, 500, 50)) is None


def test_base_frame_points_become_forward_left_up():
    local = base_to_local(np.array([[0.2, 1.0, 0.5]]))
    assert local[0] == pytest.approx([1.0, -0.2, 0.5])


def test_mask_indices_map_to_detection_image_pixels():
    pixels = mask_to_image_pixels(np.array([10 * 640 + 20]), (384, 640), (768, 1280))
    assert pixels[0] == pytest.approx([40, 20])


def test_person_position_is_the_torso_median_and_ignores_background_and_floor():
    rng = np.random.default_rng(0)
    torso = np.column_stack([rng.normal(1.1, 0.02, 100), rng.normal(0.1, 0.02, 100), np.full(100, 1.2)])
    background = np.column_stack([np.full(30, 3.0), np.zeros(30), np.full(30, 1.2)])
    floor = np.column_stack([np.full(80, 0.9), np.zeros(80), np.zeros(80)])
    points = np.vstack([torso, background, floor])
    pixels = np.full((len(points), 2), 50.0)
    position = person_position(points, pixels, (0, 0, 100, 100))
    assert position == pytest.approx((1.1, 0.1), abs=0.02)
    assert person_position(points[:39], pixels[:39], (0, 0, 100, 100)) is None
    assert person_position(points, pixels, (60, 60, 100, 100)) is None


def test_decode_matches_ultralytics_on_the_committed_checkpoint():
    pytest.importorskip("ultralytics")
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    model = YOLO(str(ROOT / "yolo11n-pose.pt"))
    bgr = cv2.imread(str(ASSETS / "bus.jpg"))
    expected = model.predict(bgr, imgsz=640, conf=0.5, iou=0.5, classes=[0], verbose=False)[0]
    tensor, lb = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 640)
    with torch.no_grad():
        raw = model.model.float().eval()(torch.from_numpy(tensor))
    raw = raw[0] if isinstance(raw, (list, tuple)) else raw
    ours = decode_pose(raw.numpy(), lb, conf=0.40, iou=0.5)

    theirs = expected.boxes.xyxy.numpy()
    their_kp = expected.keypoints.xy.numpy()
    assert len(theirs) >= 3
    for box, kp in zip(theirs, their_kp):
        ious = [_iou(box, d.box) for d in ours]
        best = int(np.argmax(ious))
        assert ious[best] >= 0.85
        assert np.mean(np.linalg.norm(ours[best].keypoints[:, :2] - kp, axis=1)) <= 8.0


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)
