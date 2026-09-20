from pathlib import Path
import sys
import json
import time

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
    leans_sad,
    square_face_box,
    yolo_detections,
    publish_ground_safety_file,
)


def test_ground_publisher_supports_motion_without_reusing_latched_positions(tmp_path):
    from ground_approach import target_from_payload

    now = time.time()
    path = tmp_path / "ground.json"
    observation = {"track_id": 4, "state": "possible_person_on_ground", "latch_status": "alert",
                   "base_position": [0, 3, 0.2], "body_radius_m": 0.8, "confidence": 0.9}
    publish_ground_safety_file(path, "alert", [{"track_id": 4}], camera_timestamp_ns=int(now * 1e9),
                               map_epoch=1, observations=[observation], depth_aligned=True)
    data = json.loads(path.read_text())
    assert data["approach_schema_version"] == 1
    assert data["possible_person_on_ground"] is True
    assert target_from_payload(data, time.time()).track_id == 4
    publish_ground_safety_file(path, "alert", [{"track_id": 4}], camera_timestamp_ns=int(now * 1e9),
                               map_epoch=1, observations=[], depth_aligned=False)
    data = json.loads(path.read_text())
    assert data["possible_person_on_ground"] is True
    assert target_from_payload(data, time.time()) is None
from check_in import (  # noqa: E402
    DISMISSED_REPLY,
    FALLBACK_REPLY,
    LAST_TURN_NOTE,
    NO_ANSWER_REPLY,
    OPENING_LINE,
    OPENING_LINES,
    LocalVoiceError,
    SadCheckIn,
    addresses_assistant,
    is_dismissal,
    speech_segments,
)


def expression(label="sadness", confidence=0.9, distress=0.9):
    return Expression(20, 20, 40, 40, label, confidence, distress)


def test_face_must_be_inside_a_yolo_person_box():
    people = [Detection(10, 10, 100, 200, 0.9)]

    assert face_belongs_to_person(expression(), people)
    assert not face_belongs_to_person(Expression(200, 20, 240, 60, "sadness", 0.9), people)


def borderline():
    """A cue above the trigger threshold but below the instant one."""
    return expression(confidence=0.7, distress=0.7)


def test_robot_sadness_cue_is_debounced():
    trigger = SadVoiceTrigger(
        hold_seconds=1.5,
        cooldown_seconds=30,
        reset_seconds=2,
        confidence=0.6,
        instant_confidence=0.85,
    )

    assert not trigger.update(borderline(), True, 0)
    assert not trigger.update(borderline(), True, 1.49)
    assert trigger.update(borderline(), True, 1.5)
    assert not trigger.update(borderline(), True, 60)


def test_unmistakable_sadness_cue_skips_the_hold():
    trigger = SadVoiceTrigger(
        hold_seconds=1.5,
        cooldown_seconds=30,
        reset_seconds=2,
        confidence=0.6,
        instant_confidence=0.85,
    )

    assert trigger.update(expression(confidence=0.9), True, 0)


def test_instant_cue_still_respects_the_cooldown():
    trigger = SadVoiceTrigger(
        hold_seconds=1.5,
        cooldown_seconds=30,
        reset_seconds=2,
        confidence=0.6,
        instant_confidence=0.85,
    )
    assert trigger.update(expression(confidence=0.9), True, 0)

    # Stay clear long enough to re-arm, then frown again inside the cooldown.
    assert not trigger.update(None, False, 3)
    assert not trigger.update(None, False, 5.1)
    assert trigger.armed
    assert not trigger.update(expression(confidence=0.9), True, 6)
    assert trigger.update(expression(confidence=0.9), True, 31)


def test_sad_leaning_readings_are_recognised_at_any_confidence():
    assert leans_sad(expression("sadness", 0.2))
    assert leans_sad(expression("sad", 0.99))
    assert not leans_sad(expression("neutral", 0.99))
    assert not leans_sad(None)


def test_trigger_reports_when_sad_evidence_is_building():
    trigger = SadVoiceTrigger(hold_seconds=1.5, confidence=0.6, instant_confidence=0.85)

    assert not trigger.warming
    trigger.update(borderline(), True, 0)
    assert trigger.warming
    trigger.update(None, False, 1)
    assert not trigger.warming


def test_trigger_uses_summed_negative_affect_not_the_sadness_label():
    """A plain frown scores disgust=60%/sadness=20%, so the single "sadness"
    class cannot gate the check-in; the four negative classes together can."""

    trigger = SadVoiceTrigger(hold_seconds=1.0, confidence=0.6, instant_confidence=0.95)
    frown = expression(label="disgust", confidence=0.6, distress=0.87)

    assert not trigger.update(frown, True, 0)
    assert trigger.update(frown, True, 1.0)


def test_trigger_ignores_diffuse_distress_on_a_resting_face():
    """Seen on the robot: "neutral 25%" with the four negative classes summing
    past the threshold started check-ins on people who were not frowning."""

    trigger = SadVoiceTrigger(hold_seconds=1.0, confidence=0.6)
    resting = expression(label="neutral", confidence=0.25, distress=0.65)

    assert not trigger.update(resting, True, 0)
    assert not trigger.update(resting, True, 5.0)
    assert not trigger.warming


def test_default_trigger_sits_above_the_resting_face_noise_band():
    trigger = SadVoiceTrigger()
    noisy = expression(label="sadness", confidence=0.5, distress=0.68)
    frown = expression(label="disgust", confidence=0.6, distress=0.87)

    assert not trigger.update(noisy, True, 0)
    assert not trigger.update(noisy, True, 5.0)
    assert not trigger.update(frown, True, 6.0)
    assert trigger.update(frown, True, 7.0)


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


def fake_analyzer(face, nets, min_face_size=40, smoothing=1.0, attack=None):
    analyzer = ExpressionAnalyzer.__new__(ExpressionAnalyzer)
    analyzer.face_detector = FakeFaceDetector(face)
    analyzer.expression_nets = nets
    analyzer.smoothing = smoothing
    analyzer.attack = smoothing if attack is None else attack
    analyzer.min_face_size = min_face_size
    analyzer.scores = None
    analyzer.missing_frames = 0
    return analyzer


class SwitchableNet:
    """One fake model whose predicted expression can change between frames."""

    def __init__(self, label):
        self.label = label

    def __call__(self, _blob):
        return np.asarray(one_hot(self.label), dtype=np.float32)[None, :]


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
    # distress sums sadness with anger, disgust and fear, so it carries the
    # softmax tail the sadness class alone leaves behind.
    assert result.confidence <= result.distress <= 1.0
    assert (result.x2 - result.x1) == (result.y2 - result.y1) == 100


def sadness_after_frames(frames, **smoothing):
    """Read the smoothed sadness score after a run of predicted labels."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    net = SwitchableNet(frames[0])
    analyzer = fake_analyzer([50, 50, 80, 100], [net], **smoothing)
    for label in frames:
        net.label = label
        analyzer.analyze(frame)
    return analyzer.scores[EMOTION_LABELS.index("sadness")]


def test_rising_sadness_is_followed_faster_than_a_symmetric_filter():
    frames = ["neutral", "sadness", "sadness"]

    fast = sadness_after_frames(frames, smoothing=0.25, attack=0.55)
    symmetric = sadness_after_frames(frames, smoothing=0.25)

    assert fast > symmetric
    assert fast >= 0.6  # crosses --sad-confidence two readings sooner


def test_one_contrary_reading_does_not_wipe_sad_evidence():
    sustained = sadness_after_frames(
        ["neutral", "sadness", "sadness"], smoothing=0.25, attack=0.55
    )
    interrupted = sadness_after_frames(
        ["neutral", "sadness", "sadness", "neutral"], smoothing=0.25, attack=0.55
    )

    assert interrupted > sustained * 0.5


def test_smoothed_expression_scores_stay_a_probability_distribution():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    net = SwitchableNet("neutral")
    analyzer = fake_analyzer([50, 50, 80, 100], [net], smoothing=0.25, attack=0.55)
    analyzer.analyze(frame)
    net.label = "sadness"
    analyzer.analyze(frame)

    assert analyzer.scores.sum() == pytest.approx(1.0)
    assert 0.0 <= analyzer.analyze(frame).confidence <= 1.0


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


class BusySpeakerWriter:
    """speaker.audio already has a writer process, as when the voice app runs."""

    def __init__(self):
        raise RuntimeError("Writer for speaker.audio already exists (pid=4242)")


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


def check_in_for(mic_sessions, answers, llm, openings=(OPENING_LINE,), **kwargs):
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
        openings=openings,
        log=lambda _line: None,
        **kwargs,
    )
    # Replies are synthesized sentence by sentence, so record whole utterances
    # separately from the individual synthesizer calls.
    spoken = []
    say = check_in.speak

    def record(text):
        spoken.append(text)
        say(text)

    check_in.speak = record
    return check_in, synthesizer, spoken


def spoken_answer():
    loud = np.full(100, 3000, dtype=np.int16)
    quiet = np.zeros(100, dtype=np.int16)
    return [loud] * 6 + [quiet] * 4


def test_check_in_asks_listens_and_replies_with_llm(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    llm = FakeLlm()
    check_in, _synthesizer, spoken = check_in_for(
        [spoken_answer(), []],
        ["my exam went badly"],
        llm,
        max_turns=2,
        answer_timeout=0.05,
    )
    check_in.converse()

    assert spoken == [OPENING_LINE, "reply to my exam went badly"]
    assert llm.heard == ["my exam went badly"]
    assert check_in.status == "idle"


def test_check_in_shows_listening_leds_while_waiting_for_an_answer(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    shown = []
    check_in, _synthesizer, _spoken = check_in_for(
        [spoken_answer()],
        ["my exam went badly"],
        FakeLlm(),
        max_turns=1,
        answer_timeout=0.05,
        show_led=shown.append,
    )
    check_in.converse()

    assert shown == ["speaking", "listening", "processing", "speaking", None]


def test_check_in_is_gentle_when_nobody_answers(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    check_in, _synthesizer, spoken = check_in_for([[]], [], FakeLlm(), answer_timeout=0.01)
    check_in.converse()

    assert spoken == [OPENING_LINE, NO_ANSWER_REPLY]


def test_check_in_falls_back_when_llm_is_offline(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    check_in, _synthesizer, spoken = check_in_for(
        [spoken_answer()], ["not great"], FakeLlm(fail=True), answer_timeout=0.05
    )
    check_in.converse()

    assert spoken == [OPENING_LINE, FALLBACK_REPLY]


def test_stop_ends_the_check_in_and_snoozes_the_next_one(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    llm = FakeLlm()
    check_in, _synthesizer, spoken = check_in_for(
        [spoken_answer(), spoken_answer()], ["Stop."], llm, answer_timeout=0.05
    )
    check_in.converse()

    assert spoken == [OPENING_LINE, DISMISSED_REPLY]
    assert llm.heard == []
    assert not check_in.start_async()


@pytest.mark.parametrize(
    "answer",
    [
        "No.",
        "Not now.",
        "Maybe later.",
        "I don't wanna talk.",
        "I don't feel like talking.",
        "I'd rather not.",
        "Give me some space.",
        "I need quiet.",
    ],
)
def test_natural_no_talk_answers_are_respected(answer):
    assert is_dismissal(answer)


def test_silence_also_starts_the_quiet_period(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    check_in, _synthesizer, spoken = check_in_for(
        [[]], [], FakeLlm(), answer_timeout=0.01, dismiss_snooze=1800
    )

    check_in.converse()

    assert spoken == [OPENING_LINE, NO_ANSWER_REPLY]
    assert check_in.quiet_until >= time.monotonic() + 1799
    assert not check_in.start_async()


def test_wake_phrase_hands_the_turn_to_the_voice_assistant(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    llm = FakeLlm()
    check_in, _synthesizer, spoken = check_in_for(
        [spoken_answer(), spoken_answer()],
        ["Hey BracketBot. Do a hug."],
        llm,
        answer_timeout=0.05,
    )
    check_in.converse()

    assert spoken == [OPENING_LINE]
    assert llm.heard == []
    assert not check_in.start_async()


def test_only_a_whole_answer_dismisses_the_check_in():
    assert is_dismissal("Stop.")
    assert is_dismissal("No, I'm fine, thanks.")
    assert is_dismissal("Okay, stop talking please")
    assert is_dismissal("Nope")
    assert is_dismissal("I don't want to chat")
    assert not is_dismissal("I can't stop crying")
    assert not is_dismissal("I'm fine I guess, but my exam went badly")
    assert addresses_assistant("hey, Bracket Bot do a wave")
    assert not addresses_assistant("the bracket broke")


def test_last_reply_is_told_not_to_ask_a_question(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    llm = FakeLlm()
    check_in, _synthesizer, _spoken = check_in_for(
        [spoken_answer(), spoken_answer()],
        ["rough day", "my exam"],
        llm,
        max_turns=2,
        answer_timeout=0.05,
    )
    check_in.converse()

    assert llm.heard == ["rough day", "my exam" + LAST_TURN_NOTE]


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


def test_speech_segments_split_sentences_and_merge_short_fragments():
    assert speech_segments("Oh no. That sounds really hard, I'm sorry.") == [
        "Oh no. That sounds really hard, I'm sorry."
    ]
    assert speech_segments(
        "That sounds really hard to sit with. Do you want to tell me more?"
    ) == [
        "That sounds really hard to sit with.",
        "Do you want to tell me more?",
    ]
    assert speech_segments("   ") == [""]


def test_reply_is_synthesized_sentence_by_sentence(monkeypatch):
    """The first sentence can start playing before the rest is rendered."""
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    llm = FakeLlm()
    llm.complete = lambda _u: type(
        "Reply",
        (),
        {"text": "That sounds really hard to sit with. Do you want to tell me more?"},
    )()
    check_in, synthesizer, _spoken = check_in_for(
        [spoken_answer(), []], ["work was rough"], llm, max_turns=1, answer_timeout=0.05
    )
    check_in.converse()

    assert synthesizer.spoken == [
        OPENING_LINE,
        "That sounds really hard to sit with.",
        "Do you want to tell me more?",
    ]


def test_openers_are_rendered_once_and_reused(monkeypatch):
    monkeypatch.setattr("check_in.time.sleep", lambda _s: None)
    check_in, synthesizer, _spoken = check_in_for(
        [[], []], [], FakeLlm(), answer_timeout=0.01
    )

    assert check_in.prewarm() == 1
    check_in.converse()
    check_in.converse()

    # One pre-render, then the cached PCM for both conversations; only the
    # no-answer reply is synthesized on demand.
    assert synthesizer.spoken == [OPENING_LINE, NO_ANSWER_REPLY, NO_ANSWER_REPLY]


def test_openers_rotate_so_the_robot_does_not_repeat_itself():
    check_in, _synthesizer, _spoken = check_in_for(
        [[]], [], FakeLlm(), openings=OPENING_LINES
    )

    openings = [check_in.next_opening() for _ in range(6)]

    assert set(openings) <= set(OPENING_LINES)
    assert all(first != second for first, second in zip(openings, openings[1:]))
    assert len(set(openings)) > 1


def test_prewarm_survives_a_synthesizer_that_is_not_ready():
    class BrokenSynthesizer:
        def synthesize(self, _text, _rate):
            raise RuntimeError("voice service down")

    check_in, _synthesizer, _spoken = check_in_for(
        [[]], [], FakeLlm(), openings=OPENING_LINES
    )
    check_in.synthesizer = BrokenSynthesizer()

    assert check_in.prewarm() == 0


def test_check_in_relays_speech_when_another_process_owns_the_speaker(monkeypatch):
    """The always-on assistant holds the single speaker.audio writer, so the
    check-in must hand its line over rather than fail the conversation."""

    import speech_relay

    relayed = []
    monkeypatch.setattr(
        speech_relay, "request", lambda **kwargs: relayed.append(kwargs) or True
    )

    check_in, synthesizer, _spoken = check_in_for([], [], FakeLlm([]))
    check_in.open_speaker = BusySpeakerWriter
    check_in.speak("Hey, why are you sad?")

    assert relayed == [{"text": "Hey, why are you sad?"}]
    # Nothing was synthesized locally: the owner renders the audio itself.
    assert synthesizer.spoken == []


def test_check_in_reports_when_nobody_can_play_the_relayed_line(monkeypatch):
    import speech_relay

    monkeypatch.setattr(speech_relay, "request", lambda **_kwargs: False)

    check_in, _synthesizer, _spoken = check_in_for([], [], FakeLlm([]))
    check_in.open_speaker = BusySpeakerWriter

    with pytest.raises(LocalVoiceError):
        check_in.speak("Hey, why are you sad?")


def test_analyzer_survives_a_nan_face_box():
    """YuNet can return NaN coordinates. NaN loses every comparison, so it used
    to slip past the size check and crash square_face_box, killing the app."""

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    analyzer = fake_analyzer(
        [float("nan"), float("nan"), float("nan"), float("nan")],
        [FakeNet(one_hot("sadness"))],
    )

    assert analyzer.analyze(frame) is None


def test_analyzer_ignores_a_nan_box_but_still_reads_a_good_face():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    analyzer = fake_analyzer([50, 50, 80, 100], [FakeNet(one_hot("sadness"))])
    nan_row = np.asarray([float("nan")] * 15, dtype=np.float32)
    good_row = np.asarray([50, 50, 80, 100] + [0.0] * 10 + [0.9], dtype=np.float32)
    analyzer.face_detector.detect = lambda _frame: (
        2,
        np.stack([nan_row, good_row]),
    )

    result = analyzer.analyze(frame)

    assert result is not None
    assert result.label == "sadness"
