from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import threading
import time
import wave

import numpy as np

from bbapps.greeter import voice_actions


class FakeGestureController:
    def __init__(self):
        self.actions = []
        self.stops = 0
        self.active = False

    def start(self, action):
        self.actions.append(action)
        self.active = True
        return True, f"Started {action}"

    def stop(self):
        self.stops += 1
        self.active = False
        return True, "stopped"

    def running(self):
        return self.active


class FakeLeds:
    def __init__(self):
        self.effects = []
        self.clears = 0

    def start_effect(self, rgb, pattern, duration):
        self.effects.append((rgb, pattern, duration))

    def clear_effect(self):
        self.clears += 1


class FakeSpeaker:
    def __init__(self):
        self.frames = []

    @contextmanager
    def buf(self):
        frame = {}
        yield frame
        self.frames.append(np.asarray(frame["audio"]).copy())


def wait_for_controller(controller, timeout=1.0):
    deadline = time.monotonic() + timeout
    while controller._operation_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not controller._operation_lock.locked()


def test_unknown_voice_action_is_rejected(tmp_path):
    controller = voice_actions.VoiceActionController(FakeGestureController(), tmp_path)

    assert controller.start("drive-away") == (
        False,
        "Voice action 'drive-away' is not installed.",
    )


def test_gesture_uses_existing_safety_controller_and_stop(tmp_path):
    gestures = FakeGestureController()
    controller = voice_actions.VoiceActionController(gestures, tmp_path)

    assert controller.start("namaste") == (True, "Started namaste")
    assert gestures.actions == ["namaste"]
    assert controller.stop() == (True, "Okay. Stopping safely.")
    wait_for_controller(controller)
    assert gestures.stops >= 1


def test_routine_runs_declared_steps_in_order(tmp_path, monkeypatch):
    controller = voice_actions.VoiceActionController(FakeGestureController(), tmp_path)
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds())
    steps = []
    monkeypatch.setattr(controller, "_run_step", steps.append)

    assert controller.start("welcome") == (True, "Started welcome")
    wait_for_controller(controller)

    assert steps == ["light-ready", "wave"]


def test_sound_reuses_bound_speaker_writer(tmp_path):
    path = tmp_path / "happy_birthday.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.array([1000, -1000, 500, -500], dtype=np.int16).tobytes())

    speaker = FakeSpeaker()
    controller = voice_actions.VoiceActionController(FakeGestureController(), tmp_path)
    controller.bind(
        speaker,
        SimpleNamespace(sample_rate=16000, chunk_size=4, channels=1),
        FakeLeds(),
    )

    assert controller.start("sound-birthday") == (True, "Started sound-birthday")
    wait_for_controller(controller)

    assert len(speaker.frames) == 1
    assert speaker.frames[0].reshape(-1).tolist() == [650, -650, 325, -325]


def test_stop_clears_active_led_effect(tmp_path, monkeypatch):
    leds = FakeLeds()
    controller = voice_actions.VoiceActionController(FakeGestureController(), tmp_path)
    controller.bind(FakeSpeaker(), SimpleNamespace(), leds)
    monkeypatch.setitem(
        voice_actions.LED_EFFECTS,
        "light-ready",
        ((70, 220, 120), "solid", 5.0),
    )

    assert controller.start("light-ready") == (True, "Started light-ready")
    deadline = time.monotonic() + 1.0
    while not leds.effects and time.monotonic() < deadline:
        time.sleep(0.005)
    assert controller.stop() == (True, "Okay. Stopping safely.")
    wait_for_controller(controller)

    assert leds.effects == [((70, 220, 120), "solid", 5.0)]
    assert leds.clears >= 1


class FakeScanner:
    installed = True

    def __init__(self, result=None, error=None, block=False, ticks=(), guidance=()):
        self.result = result
        self.error = error
        self.block = block
        self.ticks = list(ticks)
        self.guidance = list(guidance)
        self.calls = 0

    def scan(self, cancel, on_tick=None, on_guidance=None):
        self.calls += 1
        if self.block:
            cancel.wait(2.0)
            return None
        if on_tick is not None:
            for index, bpm in enumerate(self.ticks):
                on_tick(bpm, (index + 1) / (len(self.ticks) or 1))
        if on_guidance is not None:
            for kind in self.guidance:
                on_guidance(kind)
        if self.error:
            raise self.error
        return self.result


def test_heart_rate_scan_announces_result_and_frees_the_controller(tmp_path):
    announced = []
    leds = FakeLeds()
    scanner = FakeScanner({"bpm": 71.6, "confident": True})
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, heart_rate_scanner=scanner
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), leds, announce=announced.append)

    assert controller.start("heart-rate") == (True, "Started heart-rate scan")
    wait_for_controller(controller)

    assert scanner.calls == 1
    assert announced == [
        "Your heart rate looks like about 72 beats per minute. "
        "This is a camera estimate, not a medical measurement."
    ]
    assert leds.effects == [((70, 220, 120), "pulse", 600.0)]
    assert leds.clears >= 1


def test_heart_rate_scan_announces_forehead_guidance_only_once(tmp_path):
    announced = []
    scanner = FakeScanner(
        {"bpm": 72.0, "confident": True},
        guidance=["clear-forehead", "clear-forehead"],
    )
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, heart_rate_scanner=scanner
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)

    assert controller.start("heart-rate")[0] is True
    wait_for_controller(controller)

    assert announced[0] == voice_actions.CLEAR_FOREHEAD_MESSAGE
    assert announced.count(voice_actions.CLEAR_FOREHEAD_MESSAGE) == 1
    assert announced[-1].startswith("Your heart rate looks like about 72")


def test_checkup_reports_range_and_scan_failure_is_spoken_safely(tmp_path):
    announced = []
    controller = voice_actions.VoiceActionController(
        FakeGestureController(),
        tmp_path,
        heart_rate_scanner=FakeScanner(error=voice_actions.HeartRateScanError("boom")),
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)

    assert controller.start("checkup")[0] is True
    wait_for_controller(controller)
    assert announced == ["Sorry, I couldn't run the heart-rate scan right now."]

    in_range = voice_actions.heart_rate_message({"bpm": 64, "confident": True}, checkup=True)
    assert in_range.startswith("Your checkup is done.")
    assert "within the typical adult resting range" in in_range
    high = voice_actions.heart_rate_message({"bpm": 120, "confident": True}, checkup=True)
    assert "outside the typical adult resting range" in high
    weak = voice_actions.heart_rate_message({"bpm": 80, "confident": False})
    assert "don't rely on it" in weak
    assert "couldn't get a clear" in voice_actions.heart_rate_message(None)


def test_stop_cancels_heart_rate_scan_without_announcing(tmp_path):
    announced = []
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, heart_rate_scanner=FakeScanner(block=True)
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)

    assert controller.start("heart-rate")[0] is True
    assert controller.start("wave") == (False, "Another voice action is already running.")
    assert controller.stop() == (True, "Okay. Stopping safely.")
    wait_for_controller(controller)

    assert announced == []


def test_heart_rate_is_rejected_when_scanner_is_not_installed(tmp_path):
    controller = voice_actions.VoiceActionController(FakeGestureController(), tmp_path)

    assert controller.start("heart-rate") == (
        False,
        "Heart-rate scanning is not installed on this robot yet.",
    )
    assert not controller._operation_lock.locked()


def test_rppg_scanner_parses_script_json(tmp_path, python_instead_of_uv):
    script = tmp_path / "robot_rppg.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'result': {'bpm': 66.0, 'confident': True}}))\n"
    )
    scanner = voice_actions.RppgScanner(script, uv_bin=python_instead_of_uv, duration_s=1)

    assert scanner.scan(threading.Event()) == {"bpm": 66.0, "confident": True}


def test_rppg_scanner_forwards_forehead_guidance(tmp_path, python_instead_of_uv):
    script = tmp_path / "robot_rppg.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({'guidance': 'clear-forehead'}))\n"
        "print(json.dumps({'result': {'bpm': 66.0, 'confident': True}}))\n"
    )
    scanner = voice_actions.RppgScanner(script, uv_bin=python_instead_of_uv, duration_s=1)
    guidance = []

    scanner.scan(threading.Event(), on_guidance=guidance.append)

    assert guidance == ["clear-forehead"]


def test_reminder_fires_while_a_gesture_owns_the_action_lock(tmp_path):
    gestures = FakeGestureController()
    leds = FakeLeds()
    announcements = []
    controller = voice_actions.VoiceActionController(
        gestures,
        tmp_path,
        reminder_db_path=tmp_path / "reminders.sqlite3",
        reminder_timezone="America/Toronto",
    )
    controller.bind(
        FakeSpeaker(),
        SimpleNamespace(),
        leds,
        announce=announcements.append,
    )

    assert controller.start("wave") == (True, "Started wave")
    reminder = SimpleNamespace(delay_seconds=0.02, message="take my meds")
    started, reply = controller.schedule_reminder(reminder)

    assert started is True
    assert reply == "Okay. I'll remind you in 0.02 seconds to take your meds."
    deadline = time.monotonic() + 1.0
    while not announcements and time.monotonic() < deadline:
        time.sleep(0.005)
    assert announcements == ["Reminder: take your meds."]
    assert leds.effects == [voice_actions.REMINDER_SET_LED, voice_actions.REMINDER_LED]
    controller.stop()
    controller.close()


def test_cancelled_reminder_does_not_announce(tmp_path):
    announcements = []
    controller = voice_actions.VoiceActionController(
        FakeGestureController(),
        tmp_path,
        reminder_db_path=tmp_path / "reminders.sqlite3",
        reminder_timezone="America/Toronto",
    )
    controller.bind(
        FakeSpeaker(),
        SimpleNamespace(),
        FakeLeds(),
        announce=announcements.append,
    )
    reminder = SimpleNamespace(delay_seconds=0.05, message=None)

    assert controller.schedule_reminder(reminder)[0] is True
    assert controller.cancel_reminders() == (True, "Okay. I cancelled 1 reminder.")
    time.sleep(0.08)

    assert announcements == []
    controller.close()


def test_internal_reminder_actions_set_list_cancel_and_audit(tmp_path):
    controller = voice_actions.VoiceActionController(
        FakeGestureController(),
        tmp_path,
        reminder_db_path=tmp_path / "reminders.sqlite3",
        reminder_timezone="America/Toronto",
    )
    try:
        created = controller.set_reminder(
            "2099-01-15T09:30:00",
            "call home",
            source="internal-test",
        )

        assert created["timezone"] == "America/Toronto"
        assert controller.reminder_records() == [created]
        assert controller.list_reminders()[0] is True
        assert controller.cancel_reminder(created["id"]) == (
            True,
            f"Okay. I cancelled reminder {created['id']}.",
        )
        assert controller.reminder_records() == []
        assert [
            item["event"] for item in controller.reminder_audit(created["id"])
        ] == ["cancelled", "scheduled"]
    finally:
        controller.close()


class FakeFinder:
    def __init__(self, result):
        self.result = result
        self.purposes = []

    def acquire(self, purpose, cancel, hint_deg=None):
        self.purposes.append(purpose)
        return dict(self.result)


def test_scan_finds_the_person_first_and_skips_scan_when_nobody_is_there(tmp_path):
    announced = []
    scanner = FakeScanner({"bpm": 70, "confident": True})
    finder = FakeFinder({"found": False, "reason": "nobody"})
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, heart_rate_scanner=scanner, person_finder=finder
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)

    assert controller.start("heart-rate")[0] is True
    wait_for_controller(controller)

    assert finder.purposes == ["scan"]
    assert scanner.calls == 0
    assert announced == [voice_actions.NOT_FOUND_MESSAGE]


def test_scan_runs_after_turning_to_the_person(tmp_path):
    announced = []
    scanner = FakeScanner({"bpm": 70, "confident": True})
    finder = FakeFinder({"found": True, "distance": "ok", "turned_deg": 120})
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, heart_rate_scanner=scanner, person_finder=finder
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)

    controller.start("checkup")
    wait_for_controller(controller)

    assert scanner.calls == 1
    assert announced[0] == "There you are."
    assert announced[1].startswith("Your checkup is done.")


def test_unavailable_tracker_keeps_old_behaviour(tmp_path):
    scanner = FakeScanner({"bpm": 70, "confident": True})
    finder = FakeFinder({"found": False, "unavailable": True})
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, heart_rate_scanner=scanner, person_finder=finder
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=lambda _text: None)

    controller.start("heart-rate")
    wait_for_controller(controller)

    assert scanner.calls == 1


def test_person_gesture_faces_person_then_runs(tmp_path):
    gestures = FakeGestureController()
    gestures.start = lambda action: (gestures.actions.append(action), True, "ok")[1:]
    finder = FakeFinder({"found": True, "distance": "far", "turned_deg": 0})
    announced = []
    controller = voice_actions.VoiceActionController(gestures, tmp_path, person_finder=finder)
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)
    controller._cancel.wait = lambda timeout=None: False     # skip the 3 s pause

    assert controller.start("handshake")[0] is True
    wait_for_controller(controller)

    assert finder.purposes == ["gesture"]
    assert gestures.actions == ["handshake"]
    assert announced == ["Please come a little closer."]


def test_person_gesture_still_runs_when_the_base_is_too_busy_to_turn(tmp_path):
    for result in (
        {"found": False, "refused": True, "reason": "another app is already driving the base"},
        {"found": False, "error": True, "reason": "Writer for drive.ctrl already exists"},
    ):
        gestures = FakeGestureController()
        gestures.start = lambda action, gestures=gestures: (
            gestures.actions.append(action), True, "ok")[1:]
        announced = []
        controller = voice_actions.VoiceActionController(
            gestures, tmp_path, person_finder=FakeFinder(result)
        )
        controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)

        assert controller.start("fist bump")[0] is True
        wait_for_controller(controller)

        assert gestures.actions == ["fist bump"]
        assert announced == []


def test_fist_bump_gets_a_body_turn_only_when_the_base_was_free_to_face_the_person(tmp_path):
    class TurningFinder(FakeFinder):
        def __init__(self, result):
            super().__init__(result)
            self.turned = []

        def turn(self, delta_deg, cancel):
            self.turned.append(delta_deg)
            return {"ok": True}

    found = {"found": True, "centered": True, "distance": "ok", "turned_deg": 0}
    for result, may_turn in ((found, True), ({**found, "centered": False}, False)):
        offered = []
        gestures = FakeGestureController()

        def start(action, turn_body=None, gestures=gestures, offered=offered):
            offered.append(turn_body)
            gestures.actions.append(action)
            return True, "ok"

        gestures.start = start
        finder = TurningFinder(result)
        controller = voice_actions.VoiceActionController(gestures, tmp_path, person_finder=finder)
        controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=lambda _: None)

        assert controller.start("fist bump")[0] is True
        wait_for_controller(controller)

        assert gestures.actions == ["fist bump"]
        assert (offered[0] is not None) is may_turn
        if may_turn:
            assert offered[0](12.0) is True
            assert finder.turned == [12.0]


def test_non_person_gestures_do_not_search(tmp_path):
    finder = FakeFinder({"found": False})
    gestures = FakeGestureController()
    controller = voice_actions.VoiceActionController(gestures, tmp_path, person_finder=finder)

    assert controller.start("wave") == (True, "Started wave")
    assert finder.purposes == []
    controller.stop()
    wait_for_controller(controller)


def test_scan_speaks_spaced_ticks_and_caps_how_many(tmp_path, monkeypatch):
    # Every estimate reaches the controller; only a few become speech.
    monkeypatch.setattr(voice_actions, "TICK_SPACING_S", 0.0)
    announced = []
    scanner = FakeScanner(
        {"bpm": 74.0, "confident": True},
        ticks=[None, 70.4, 71.6, 72.0, 73.2, 74.4, 75.0],
    )
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, heart_rate_scanner=scanner
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)
    assert controller.start("heart-rate")[0] is True
    wait_for_controller(controller, timeout=2.0)

    ticks = announced[:-1]
    assert len(ticks) == voice_actions.SPOKEN_TICKS
    # The None estimate is skipped, so the first spoken tick is the first lock.
    assert ticks[0] == "I'm reading about 70 beats per minute."
    assert ticks[1:] == ["About 72.", "About 72.", "About 73."]
    assert announced[-1].startswith("Your heart rate looks like about 74")


def test_ticks_are_throttled_so_speech_cannot_pile_up(tmp_path):
    announced = []
    scanner = FakeScanner(
        {"bpm": 80.0, "confident": True}, ticks=[80.0, 80.5, 81.0]
    )
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, heart_rate_scanner=scanner
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)
    assert controller.start("heart-rate")[0] is True
    wait_for_controller(controller, timeout=2.0)

    # The ticks arrive back to back, so the spacing gate allows only the first.
    assert announced == [
        "I'm reading about 80 beats per minute.",
        "Your heart rate looks like about 80 beats per minute. "
        "This is a camera estimate, not a medical measurement.",
    ]


def look_controller(tmp_path, finder, announced):
    gestures = FakeGestureController()
    controller = voice_actions.VoiceActionController(gestures, tmp_path, person_finder=finder)
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)
    return controller, gestures


def test_look_at_me_turns_to_the_person_without_moving_the_arms(tmp_path):
    announced = []
    finder = FakeFinder({"found": True, "distance": "far", "turned_deg": 140})
    controller, gestures = look_controller(tmp_path, finder, announced)

    assert controller.start("look-at-me") == (True, "Okay. Looking for you.")
    wait_for_controller(controller)

    assert finder.purposes == ["look"]
    assert gestures.actions == []
    # Distance does not matter for a look, so no "come closer".
    assert announced == ["There you are."]


def test_look_at_me_confirms_when_already_facing_the_person(tmp_path):
    announced = []
    finder = FakeFinder({"found": True, "distance": "ok", "turned_deg": 4})
    controller, _ = look_controller(tmp_path, finder, announced)

    controller.start("look-at-me")
    wait_for_controller(controller)

    assert announced == ["I see you."]


def test_look_at_me_says_so_when_nobody_or_no_tracker(tmp_path):
    announced = []
    controller, _ = look_controller(
        tmp_path, FakeFinder({"found": False, "reason": "nobody"}), announced
    )
    controller.start("look-at-me")
    wait_for_controller(controller)
    assert announced == [voice_actions.NOT_FOUND_MESSAGE]

    announced.clear()
    controller, _ = look_controller(
        tmp_path, FakeFinder({"found": False, "unavailable": True}), announced
    )
    controller.start("look-at-me")
    wait_for_controller(controller)
    assert announced == [voice_actions.TRACKER_MISSING_MESSAGE]


def test_fist_bump_plays_its_sound_cue_during_the_gesture(tmp_path, monkeypatch):
    with wave.open(str(tmp_path / "fist_bump_balalala.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.array([1000, -1000, 500, -500], dtype=np.int16).tobytes())
    monkeypatch.setitem(voice_actions.GESTURE_SOUNDS, "fist bump", ("fist_bump_balalala.wav", 0.01))

    speaker = FakeSpeaker()
    gestures = FakeGestureController()
    controller = voice_actions.VoiceActionController(gestures, tmp_path)
    controller.bind(speaker, SimpleNamespace(sample_rate=16000, chunk_size=4, channels=1), FakeLeds())

    assert controller.start("fist bump")[0] is True
    deadline = time.monotonic() + 1.0
    while not speaker.frames and time.monotonic() < deadline:
        time.sleep(0.005)
    gestures.active = False
    wait_for_controller(controller)

    assert len(speaker.frames) == 1


def test_gesture_sound_is_skipped_when_the_gesture_already_ended(tmp_path, monkeypatch):
    monkeypatch.setitem(voice_actions.GESTURE_SOUNDS, "fist bump", ("fist_bump_balalala.wav", 0.05))
    speaker = FakeSpeaker()
    gestures = FakeGestureController()
    controller = voice_actions.VoiceActionController(gestures, tmp_path)
    controller.bind(speaker, SimpleNamespace(sample_rate=16000, chunk_size=4, channels=1), FakeLeds())

    assert controller.start("fist bump")[0] is True
    gestures.active = False           # e.g. the safety check refused the motion
    wait_for_controller(controller)
    time.sleep(0.15)

    assert speaker.frames == []


# --- follow me -------------------------------------------------------------

class FakeFollowRunner:
    installed = True

    def __init__(self, states=(), last="stopped; zero twist sent"):
        self.states = states
        self.last = last
        self.started = threading.Event()

    def follow(self, cancel, on_state=None):
        self.started.set()
        for state in self.states:
            on_state(state)
        if self.last.startswith("refusing"):
            return self.last
        cancel.wait(2.0)
        return self.last


def test_follow_me_faces_the_person_then_follows_until_stopped(tmp_path):
    announced = []
    leds = FakeLeds()
    finder = FakeFinder({"found": True, "distance": "far", "turned_deg": 3})
    runner = FakeFollowRunner(states=("SEARCHING", "FOLLOWING", "LOST", "FOLLOWING", "LOST"))
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, person_finder=finder, follow_runner=runner
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), leds, announce=announced.append)

    assert controller.start("follow-me")[0] is True
    assert runner.started.wait(1.0)
    assert controller.start("wave")[0] is False  # nothing else may move the robot meanwhile
    assert controller.stop()[0] is True
    wait_for_controller(controller)

    assert finder.purposes == ["look"]
    # Each state is spoken once, and a stop the person asked for needs no "I've stopped".
    assert announced == [
        voice_actions.FOLLOW_STATE_MESSAGES["FOLLOWING"],
        voice_actions.FOLLOW_STATE_MESSAGES["LOST"],
    ]
    assert leds.effects[1][:2] == voice_actions.FOLLOW_STATE_LED["FOLLOWING"]


def test_follow_me_chirps_only_while_following_and_not_while_listening(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_actions, "FOLLOW_CHIRP_PERIOD_S", 0.02)
    finder = FakeFinder({"found": True, "distance": "far", "turned_deg": 3})
    played = []

    def run(states, listening=False):
        controller = voice_actions.VoiceActionController(
            FakeGestureController(), tmp_path, person_finder=finder,
            follow_runner=FakeFollowRunner(states=states),
        )
        controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=lambda _: None)
        monkeypatch.setattr(controller, "_play_sound", lambda path, **_: played.append(path.name))
        if listening:
            controller.listening.set()
        assert controller.start("follow-me")[0] is True
        time.sleep(0.2)
        controller.stop()
        wait_for_controller(controller)
        count = len(played)
        played.clear()
        return count

    assert run(("SEARCHING", "FOLLOWING")) >= 2
    assert run(("SEARCHING", "FOLLOWING", "LOST")) == 0
    assert run(("SEARCHING", "FOLLOWING"), listening=True) == 0


def test_follow_me_does_not_start_when_nobody_is_found(tmp_path):
    announced = []
    runner = FakeFollowRunner()
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path,
        person_finder=FakeFinder({"found": False, "reason": "nobody"}), follow_runner=runner,
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)

    assert controller.start("follow-me")[0] is True
    wait_for_controller(controller)

    assert not runner.started.is_set()
    assert announced == [voice_actions.NOT_FOUND_MESSAGE]


def test_follow_me_says_why_the_runner_refused(tmp_path):
    announced = []
    runner = FakeFollowRunner(last="refusing to start: robot is not upright")
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, follow_runner=runner
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)

    assert controller.start("follow-me")[0] is True
    wait_for_controller(controller)

    assert announced == ["I can't follow you right now: robot is not upright."]


def test_follow_me_is_refused_when_the_runner_is_not_installed(tmp_path):
    controller = voice_actions.VoiceActionController(FakeGestureController(), tmp_path)
    assert controller.start("follow-me") == (False, voice_actions.FOLLOW_MISSING_MESSAGE)
    assert not controller._operation_lock.locked()


FAKE_RUNNER = '''
import json, sys
assert sys.argv[1:] == ["--v-max", "0.3", "--relock", "--no-odom-check", "--human-gate", "--ignore-writer", "person_tracker.py", "--no-led"], sys.argv
print("[follow] state FOLLOWING", flush=True)
print("FOLLOW_STATUS {}", flush=True)
beats = 0
for line in sys.stdin:
    kind = json.loads(line)["type"]
    beats += kind == "heartbeat"
    if kind == "stop":
        print(f"[follow] stopped after {beats} heartbeats", flush=True)
        break
'''


def test_follow_runner_heartbeats_then_asks_the_process_to_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_actions, "FOLLOW_HEARTBEAT_S", 0.02)
    script = tmp_path / "robot_follow.py"
    script.write_text(FAKE_RUNNER)
    runner = voice_actions.FollowRunner(script, python_bin=sys.executable)
    cancel = threading.Event()
    states = []
    threading.Timer(0.4, cancel.set).start()

    last = runner.follow(cancel, on_state=states.append)

    assert states == ["FOLLOWING"]
    assert last.startswith("stopped after ")
    assert int(last.split()[2]) >= 3


class FakeGroundRunner:
    installed = True

    def __init__(self, lines):
        self.lines = lines
        self.runs = 0

    def check_on_person(self, cancel, on_state=None):
        self.runs += 1
        return list(self.lines)


def ground_controller(tmp_path, runner, announced):
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, follow_runner=runner
    )
    controller.bind(FakeSpeaker(), SimpleNamespace(), FakeLeds(), announce=announced.append)
    return controller


def test_ground_check_announces_itself_then_speaks_the_runners_line_on_arrival(tmp_path, monkeypatch):
    announced = []
    arrival = tmp_path / "arrived.json"
    monkeypatch.setattr(voice_actions.note_ground_arrival, "__defaults__", (arrival,))
    runner = FakeGroundRunner(["state APPROACHING", "say are you okay", "ground approach complete"])
    controller = ground_controller(tmp_path, runner, announced)

    assert controller.start("check-on-person")[0] is True
    wait_for_controller(controller)

    assert announced == [voice_actions.GROUND_START_MESSAGE, "are you okay"]
    assert not controller._operation_lock.locked()
    # The vision app is told to start listening for the answer.
    assert time.time() - json.loads(arrival.read_text())["arrived_at"] < 5


def test_ground_check_says_why_it_could_not_move(tmp_path):
    announced = []
    runner = FakeGroundRunner(["refusing to start: drive.ctrl already has a writer"])
    controller = ground_controller(tmp_path, runner, announced)

    controller.start("check-on-person")
    wait_for_controller(controller)

    assert announced[-1] == "I can't come over right now: drive.ctrl already has a writer."


def test_ground_check_needs_the_speaker_and_an_installed_runner(tmp_path):
    controller = voice_actions.VoiceActionController(
        FakeGestureController(), tmp_path, follow_runner=FakeGroundRunner([])
    )
    assert controller.start("check-on-person")[0] is False
    assert not controller._operation_lock.locked()


GROUND_FAKE_RUNNER = '''
import sys
assert sys.argv[1:] == ["--ground-approach", "--no-speech", "--v-max", "0.05", "--ignore-writer", "person_tracker.py", "--no-led"], sys.argv
print("[follow] say hello there", flush=True)
print("[follow] ground approach complete", flush=True)
'''


def test_follow_runner_ground_mode_creeps_and_leaves_speech_to_the_assistant(tmp_path):
    script = tmp_path / "robot_follow.py"
    script.write_text(GROUND_FAKE_RUNNER)
    runner = voice_actions.FollowRunner(script, python_bin=sys.executable)

    lines = runner.check_on_person(threading.Event())

    assert lines == ["say hello there", "ground approach complete"]


class FakeStarter:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.listening = threading.Event()

    def start(self, action):
        assert action == "check-on-person"
        self.calls += 1
        return self.results.pop(0), ""


def write_alert(path, status, published_at, alerts=1):
    path.write_text(json.dumps({
        "status": status, "published_at": published_at,
        "alerts": [{"track_id": i} for i in range(alerts)],
    }))


def test_ground_watcher_starts_once_per_alert_episode(tmp_path):
    path = tmp_path / "alert.json"
    clock = [100.0]
    starter = FakeStarter([False, True, True])
    watcher = voice_actions.GroundAlertWatcher(
        starter, path, rearm_clear_s=10.0, clock=lambda: clock[0], wall=lambda: 500.0
    )

    assert watcher.poll() is False                      # no file yet
    write_alert(path, "alert", 499.8)
    assert watcher.poll() is False                      # assistant busy: stays pending
    assert watcher.poll() is True
    assert watcher.poll() is False                      # same episode, still lying there
    assert starter.calls == 2

    write_alert(path, "clear", 499.9)
    watcher.poll()
    clock[0] += 5.0
    write_alert(path, "alert", 499.9)
    assert watcher.poll() is False                      # cleared for too short a time
    write_alert(path, "clear", 499.9)
    watcher.poll()
    clock[0] += 10.0
    watcher.poll()
    write_alert(path, "alert", 499.9)
    assert watcher.poll() is True


def test_ground_watcher_ignores_stale_ambiguous_and_mid_conversation_alerts(tmp_path):
    path = tmp_path / "alert.json"
    starter = FakeStarter([True])
    watcher = voice_actions.GroundAlertWatcher(starter, path, wall=lambda: 500.0)

    write_alert(path, "alert", 490.0)                   # vision app stopped publishing
    assert watcher.poll() is False
    write_alert(path, "alert", 499.9, alerts=2)         # two people down
    assert watcher.poll() is False
    path.write_text("{not json")
    assert watcher.poll() is False
    write_alert(path, "alert", 499.9)
    starter.listening.set()                             # recording a wake-word turn
    assert watcher.poll() is False
    assert starter.calls == 0
    starter.listening.clear()
    assert watcher.poll() is True


def test_reminder_replies_are_speakable(tmp_path):
    controller = voice_actions.VoiceActionController(
        SimpleNamespace(running=lambda: False, stop=lambda: None),
        tmp_path,
        reminder_db_path=tmp_path / "reminders.sqlite3",
        reminder_timezone="America/Toronto",
    )
    try:
        assert controller._duration_text(5400) == "1 hour 30 minutes"
        assert controller._duration_text(90) == "1 minute 30 seconds"
        assert controller._second_person("call my mom and tell her i'm fine") == (
            "call your mom and tell her you're fine"
        )
        _, reply = controller.schedule_reminder(
            SimpleNamespace(
                delay_seconds=7200.0,
                message="the meeting",
                connector="about",
                due_text="5 PM",
            )
        )
        assert reply == "Okay. I'll remind you at 5 PM about the meeting."
        _, reply = controller.schedule_reminder(
            SimpleNamespace(delay_seconds=600.0, message=None)
        )
        assert reply == "Okay. Your timer is set for 10 minutes."
        listed, reply = controller.list_reminders()
        assert listed is True
        assert "the meeting, at " in reply
        assert "a timer, in 10 minutes" in reply
        # ISO timestamps are unreadable through text to speech.
        assert "T" not in reply.split("You have", 1)[1].replace("PM", "").replace("AM", "")
    finally:
        controller.close()


def test_setting_a_reminder_plays_a_confirmation_ding(tmp_path):
    frames = []

    class RecordingSpeaker:
        def buf(self):
            speaker = self

            class Frame(dict):
                def __enter__(self):
                    return self

                def __exit__(self, *_exc):
                    frames.append(self["audio"])

            return Frame()

    controller = voice_actions.VoiceActionController(
        SimpleNamespace(running=lambda: False, stop=lambda: None),
        tmp_path,
        reminder_db_path=tmp_path / "reminders.sqlite3",
        reminder_timezone="America/Toronto",
    )
    try:
        controller.bind(
            RecordingSpeaker(),
            SimpleNamespace(sample_rate=16000, chunk_size=320, channels=1),
            None,
        )
        started, _reply = controller.schedule_reminder(
            SimpleNamespace(kind="timer", delay_seconds=600.0, message=None)
        )
        assert started is True
        assert frames and all(frame.shape == (320, 1) for frame in frames)
        assert max(int(abs(frame).max()) for frame in frames) > 3000
    finally:
        controller.close()
