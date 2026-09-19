from types import SimpleNamespace

import numpy as np

from people_detector import Detection, detections_from_result, parse_source, smooth_scores


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10.2, 20.7, 99.6, 200.1], [1, 2, 3, 4]])
        self.conf = FakeTensor([0.9123, 0.8])
        self.cls = FakeTensor([0, 2])

    def __len__(self):
        return 2


def test_numeric_source_is_camera_index():
    assert parse_source("0") == 0
    assert parse_source("12") == 12


def test_non_numeric_source_is_left_as_string():
    assert parse_source("clip.mp4") == "clip.mp4"
    assert parse_source("rtsp://robot/camera") == "rtsp://robot/camera"


def test_only_person_boxes_are_returned():
    result = SimpleNamespace(boxes=FakeBoxes())

    assert detections_from_result(result) == [Detection(10, 21, 100, 200, 0.9123)]


def test_empty_boxes_are_returned_as_empty_list():
    assert detections_from_result(SimpleNamespace(boxes=None)) == []


def test_expression_scores_are_smoothed():
    previous = np.array([0.8, 0.2])
    current = np.array([0.0, 1.0])

    np.testing.assert_allclose(smooth_scores(previous, current, 0.25), [0.6, 0.4])


def test_first_expression_scores_are_copied():
    current = np.array([0.2, 0.8])
    smoothed = smooth_scores(None, current, 0.25)

    np.testing.assert_array_equal(smoothed, current)
    assert smoothed is not current
