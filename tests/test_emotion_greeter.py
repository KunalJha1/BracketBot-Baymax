from pathlib import Path
import sys

import pytest
import numpy as np


pytest.importorskip("cv2")


GREETER_DIR = Path(__file__).parents[1] / "bbapps" / "emotion_greeter"
sys.path.insert(0, str(GREETER_DIR))

from main import (  # noqa: E402
    CameraActionController,
    DashboardState,
    Detection,
    Expression,
    PersonTracker,
    SadVoiceTrigger,
    face_belongs_to_person,
    yolo_detections,
)


def expression(label="sadness", confidence=0.9):
    return Expression(20, 20, 40, 40, label, confidence)


def test_face_must_be_inside_a_yolo_person_box():
    people = [Detection(10, 10, 100, 200, 0.9)]

    assert face_belongs_to_person(expression(), people)
    assert not face_belongs_to_person(Expression(200, 20, 240, 60, "sadness", 0.9), people)


def test_robot_sadness_cue_is_debounced():
    trigger = SadVoiceTrigger(
        hold_seconds=1.5,
        cooldown_seconds=30,
        reset_seconds=2,
        confidence=0.6,
    )

    assert not trigger.update(expression(), True, 0)
    assert not trigger.update(expression(), True, 1.49)
    assert trigger.update(expression(), True, 1.5)
    assert not trigger.update(expression(), True, 60)


def test_yolo_pose_output_decodes_person_box():
    output = np.zeros((1, 56, 2), dtype=np.float32)
    output[0, :5, 0] = [160, 160, 100, 200, 0.9]
    output[0, :5, 1] = [10, 10, 5, 5, 0.1]

    detections = yolo_detections(
        output,
        scale=0.5,
        pad_x=0,
        pad_y=40,
        frame_width=640,
        frame_height=480,
        confidence=0.4,
        nms_threshold=0.45,
    )

    assert detections == [Detection(220, 40, 420, 440, 0.8999999761581421)]


def test_person_tracker_keeps_id_for_overlapping_detection():
    tracker = PersonTracker()

    first = tracker.update([Detection(10, 10, 110, 210, 0.9)])
    second = tracker.update([Detection(20, 10, 120, 210, 0.8)])

    assert first[0].track_id == 1
    assert second[0].track_id == 1


def test_person_tracker_assigns_new_id_to_separate_person():
    tracker = PersonTracker()
    tracker.update([Detection(10, 10, 110, 210, 0.9)])

    tracked = tracker.update([Detection(400, 10, 500, 210, 0.9)])

    assert tracked[0].track_id == 2


def test_person_tracker_keeps_id_when_box_moves_without_overlap():
    tracker = PersonTracker()
    tracker.update([Detection(100, 100, 300, 500, 0.9)])

    tracked = tracker.update([Detection(310, 100, 510, 500, 0.9)])

    assert tracked[0].track_id == 1


def test_dashboard_selects_primary_left_and_right_camera_targets():
    state = DashboardState()
    state.update_targets(
        [
            Detection(10, 20, 210, 460, 0.9),
            Detection(450, 30, 620, 450, 0.95),
        ],
        640,
        480,
        observed_at=0.0,
    )
    state.targets_at = __import__("time").monotonic()

    primary = state.select_target("primary")
    left = state.select_target("left")
    right = state.select_target("right")

    assert primary == left
    assert left[0] < 0
    assert right[0] > 0


def test_dashboard_rejects_stale_camera_target():
    state = DashboardState()
    state.update_targets(
        [Detection(10, 20, 210, 460, 0.9)],
        640,
        480,
        observed_at=0.0,
    )

    assert state.select_target("primary") is None


def test_camera_action_controller_preserves_runner_failure_log(tmp_path):
    runner = tmp_path / "failing_runner.py"
    runner.write_text(
        "import sys\n"
        "print('[point] stage=planning diagnostic')\n"
        "print('[point] NOT SAFE TO RUN: test failure')\n"
        "sys.exit(1)\n"
    )
    controller = CameraActionController(runner)

    assert controller.start((0.2, 0.3))[0]
    deadline = __import__("time").monotonic() + 2
    while controller.running() and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)

    status = controller.status()
    assert status["movement_running"] is False
    assert status["movement_stage"] == "failed"
    assert status["movement_error"] == "[point] NOT SAFE TO RUN: test failure"
    assert status["movement_log"][-1] == status["movement_error"]
