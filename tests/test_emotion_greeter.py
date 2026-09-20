from pathlib import Path
import sys

import pytest
import numpy as np


pytest.importorskip("cv2")


GREETER_DIR = Path(__file__).parents[1] / "bbapps" / "emotion_greeter"
sys.path.insert(0, str(GREETER_DIR))

from main import (  # noqa: E402
    EMOTION_LABELS,
    CameraActionController,
    DashboardState,
    Detection,
    Expression,
    ExpressionAnalyzer,
    PersonTracker,
    SadVoiceTrigger,
    expression_probabilities,
    face_belongs_to_person,
    square_face_box,
    yolo_detections,
)
from check_in import (  # noqa: E402
    FALLBACK_REPLY,
    NO_ANSWER_REPLY,
    OPENING_LINE,
    SadCheckIn,
)


def expression(label="sadness", confidence=0.9, distress=0.9):
    return Expression(20, 20, 40, 40, label, confidence, distress)


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


def test_trigger_uses_summed_negative_affect_not_the_sadness_label():
    """A plain frown scores disgust=60%/sadness=20%, so the single "sadness"
    class cannot gate the check-in; the four negative classes together can."""

    trigger = SadVoiceTrigger(hold_seconds=1.0, confidence=0.6)
    frown = expression(label="disgust", confidence=0.6, distress=0.87)

    assert not trigger.update(frown, True, 0)
    assert trigger.update(frown, True, 1.0)


def test_trigger_ignores_a_confident_but_untroubled_face():
    trigger = SadVoiceTrigger(hold_seconds=1.0, confidence=0.6)
    calm = expression(label="surprise", confidence=0.66, distress=0.19)

    assert not trigger.update(calm, True, 0)
    assert not trigger.update(calm, True, 1.0)
    assert not trigger.update(calm, True, 5.0)


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


def test_yolo_pose_output_decodes_keypoints_into_source_pixels():
    output = np.zeros((1, 56, 1), dtype=np.float32)
    output[0, :5, 0] = [160, 160, 100, 200, 0.9]
    output[0, 5:8, 0] = [100, 140, 0.8]

    detection = yolo_detections(
        output,
        scale=0.5,
        pad_x=0,
        pad_y=40,
        frame_width=640,
        frame_height=480,
        confidence=0.4,
        nms_threshold=0.45,
    )[0]

    assert len(detection.keypoints) == 1
    assert detection.keypoints[0].index == 0
    assert detection.keypoints[0].x == pytest.approx(200)
    assert detection.keypoints[0].y == pytest.approx(200)
    assert detection.keypoints[0].confidence == pytest.approx(0.8)


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


def test_square_face_box_is_centered_and_clamped():
    assert square_face_box(40, 20, 20, 40, 200, 200) == (30, 20, 70, 60)
    assert square_face_box(0, 0, 20, 40, 200, 200) == (0, 0, 30, 40)


def test_expression_probabilities_drop_contempt_and_valence_arousal():
    logits = np.zeros(10)
    logits[EMOTION_LABELS.index("contempt")] = 9.0
    logits[8:] = 50.0  # valence/arousal from a multi-task head
    probabilities = expression_probabilities(logits)

    assert probabilities.shape == (len(EMOTION_LABELS),)
    assert probabilities[EMOTION_LABELS.index("contempt")] == 0
    assert probabilities.sum() == pytest.approx(1.0)


class FakeNet:
    """Stands in for one entry of ExpressionAnalyzer.expression_nets, which
    holds callables so the ONNX backend (onnxruntime or cv2.dnn) can vary."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)[None, :]

    def __call__(self, _blob):
        return self.logits


class FakeFaceDetector:
    def __init__(self, face):
        self.face = face

    def setInputSize(self, _size):
        pass

    def detect(self, _frame):
        return 1, np.asarray([self.face], dtype=np.float32)


def fake_analyzer(face, nets, min_face_size=40):
    analyzer = ExpressionAnalyzer.__new__(ExpressionAnalyzer)
    analyzer.face_detector = FakeFaceDetector(face)
    analyzer.expression_nets = nets
    analyzer.smoothing = 1.0
    analyzer.min_face_size = min_face_size
    analyzer.scores = None
    analyzer.missing_frames = 0
    return analyzer


def one_hot(label, size=8):
    logits = np.zeros(size)
    logits[EMOTION_LABELS.index(label)] = 5.0
    return logits


def test_expression_analyzer_averages_model_ensemble():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    analyzer = fake_analyzer(
        [50, 50, 80, 100],
        [FakeNet(one_hot("sadness")), FakeNet(one_hot("sadness", size=10))],
    )
    result = analyzer.analyze(frame)

    assert result.label == "sadness"
    assert result.distress == pytest.approx(result.confidence)
    assert (result.x2 - result.x1) == (result.y2 - result.y1) == 100


def test_expression_analyzer_skips_faces_too_small_to_read():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    analyzer = fake_analyzer([50, 50, 20, 25], [FakeNet(one_hot("sadness"))])

    assert analyzer.analyze(frame) is None
    assert analyzer.missing_frames == 1


class FakeSpeakerWriter:
    def __init__(self):
        self.frames = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def buf(self):
        writer = self

        class Buffer(dict):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                writer.frames.append(self["audio"])

        return Buffer()


class FakeMic:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.data = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def ready(self):
        if not self.chunks:
            return False
        self.data = {"audio": self.chunks.pop(0)}
        return True


class FakeTranscriber:
    def __init__(self, answers):
        self.answers = list(answers)

    def transcribe(self, _audio, _rate):
        return self.answers.pop(0)


class RecordingSynthesizer:
    def __init__(self):
        self.spoken = []

    def synthesize(self, text, _rate):
        self.spoken.append(text)
        return np.zeros(16, dtype=np.int16)


class FakeLlm:
    def __init__(self, fail=False):
        self.fail = fail
        self.heard = []

    def complete(self, utterance):
        if self.fail:
            raise RuntimeError("offline")
        self.heard.append(utterance)
        return type("Reply", (), {"text": f"reply to {utterance}"})()


def check_in_for(mic_sessions, answers, llm, **kwargs):
    config = type("Config", (), {"sample_rate": 1000, "chunk_size": 100, "channels": 1})()
    synthesizer = RecordingSynthesizer()
    sessions = iter(mic_sessions)
    check_in = SadCheckIn(
        synthesizer=synthesizer,
        transcriber=FakeTranscriber(answers),
        llm_factory=lambda: llm,
        speaker_config=config,
        mic_config=config,
        open_speaker=FakeSpeakerWriter,
        open_mic=lambda: FakeMic(next(sessions)),
        trailing_silence=0.2,
        log=lambda _line: None,
        **kwargs,
    )
    return check_in, synthesizer


def spoken_answer():
    loud = np.full(100, 3000, dtype=np.int16)
    quiet = np.zeros(100, dtype=np.int16)
    return [loud] * 6 + [quiet] * 4


def test_check_in_asks_listens_and_replies_with_llm(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    llm = FakeLlm()
    check_in, synthesizer = check_in_for(
        [spoken_answer(), []],
        ["my exam went badly"],
        llm,
        max_turns=2,
        answer_timeout=0.05,
    )
    check_in.converse()

    assert synthesizer.spoken == [OPENING_LINE, "reply to my exam went badly"]
    assert llm.heard == ["my exam went badly"]
    assert check_in.status == "idle"


def test_check_in_is_gentle_when_nobody_answers(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    check_in, synthesizer = check_in_for([[]], [], FakeLlm(), answer_timeout=0.01)
    check_in.converse()

    assert synthesizer.spoken == [OPENING_LINE, NO_ANSWER_REPLY]


def test_check_in_falls_back_when_llm_is_offline(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    check_in, synthesizer = check_in_for(
        [spoken_answer()], ["not great"], FakeLlm(fail=True), answer_timeout=0.05
    )
    check_in.converse()

    assert synthesizer.spoken == [OPENING_LINE, FALLBACK_REPLY]


def test_dashboard_only_encodes_frames_while_someone_is_watching():
    state = DashboardState()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    state.publish(frame, {"frame": 1})
    assert state.jpeg is None
    assert state.status()["frame"] == 1

    state.add_viewer(1)
    state.publish(frame, {"frame": 2})
    sequence, jpeg = state.wait_for_frame(-1, timeout=0.1)
    assert sequence == 2
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"

    state.add_viewer(-1)
    state.publish(frame, {"frame": 3})
    assert state.jpeg is None
