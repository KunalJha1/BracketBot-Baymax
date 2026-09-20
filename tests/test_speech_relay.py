"""The speech relay hands one utterance to whichever process owns the speaker.

``speaker.audio`` takes a single writer, so the emotion greeter cannot speak
while the always-on assistant runs. These cover the handoff, and the failure
modes that would otherwise leave the greeter blocked forever.
"""

from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "bbapps" / "greeter"))

import speech_relay  # noqa: E402


def serve_until(spool, spoken, played, deadline_s=5.0, **kwargs):
    """Run the owner's side until it plays something or the test times out."""

    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        count = speech_relay.serve_pending(
            spoken.append, play_wav=played.append, spool=spool, **kwargs
        )
        if count:
            return count
        time.sleep(0.01)
    return 0


def test_request_is_spoken_by_the_owner_and_unblocks_the_caller(tmp_path):
    spoken: list[str] = []
    played: list[Path] = []
    result: list[bool] = []

    caller = threading.Thread(
        target=lambda: result.append(
            speech_relay.request(text="Hey, why are you sad?", spool=tmp_path)
        )
    )
    caller.start()
    assert serve_until(tmp_path, spoken, played) == 1
    caller.join(timeout=5)

    assert not caller.is_alive()
    assert result == [True]
    assert spoken == ["Hey, why are you sad?"]
    assert played == []


def test_wav_requests_are_played_not_synthesized(tmp_path):
    spoken: list[str] = []
    played: list[Path] = []
    result: list[bool] = []

    caller = threading.Thread(
        target=lambda: result.append(
            speech_relay.request(wav="/tmp/sad_prompt.wav", spool=tmp_path)
        )
    )
    caller.start()
    assert serve_until(tmp_path, spoken, played) == 1
    caller.join(timeout=5)

    assert result == [True]
    assert played == [Path("/tmp/sad_prompt.wav")]
    assert spoken == []


def test_request_reports_failure_when_nothing_serves_the_spool(tmp_path):
    """The greeter must learn the person heard nothing, not block forever."""

    assert speech_relay.request(text="hello", timeout=0.2, spool=tmp_path) is False
    # The abandoned request is cleaned up rather than spoken later out of context.
    assert list(tmp_path.glob("*.json")) == []


def test_stale_requests_are_dropped_instead_of_spoken_late(tmp_path):
    spoken: list[str] = []
    speech_relay._write_atomic(
        tmp_path / "old.json",
        {"id": "old", "created": 1000.0, "text": "ancient"},
    )

    served = speech_relay.serve_pending(
        spoken.append,
        spool=tmp_path,
        now=lambda: 1000.0 + speech_relay.REQUEST_TTL_S + 1,
    )

    assert served == 0
    assert spoken == []
    assert list(tmp_path.glob("*.json")) == []


def test_a_failing_utterance_still_releases_the_caller(tmp_path):
    """The owner's loop must survive a bad request, and not strand the caller."""

    def explode(_text):
        raise RuntimeError("tts is down")

    speech_relay._write_atomic(
        tmp_path / "boom.json", {"id": "boom", "created": time.time(), "text": "hi"}
    )

    served = speech_relay.serve_pending(explode, spool=tmp_path)

    assert served == 0
    assert (tmp_path / "boom.done").exists()
    assert list(tmp_path.glob("*.json")) == []


def test_malformed_requests_are_discarded(tmp_path):
    spoken: list[str] = []
    (tmp_path / "junk.json").write_text("{not json")

    assert speech_relay.serve_pending(spoken.append, spool=tmp_path) == 0
    assert spoken == []
    assert list(tmp_path.glob("*.json")) == []


def test_requests_are_served_in_the_order_they_were_posted(tmp_path):
    spoken: list[str] = []
    for index, line in enumerate(["first", "second", "third"]):
        speech_relay._write_atomic(
            tmp_path / f"{index}.json",
            {"id": str(index), "created": time.time(), "text": line},
        )

    assert speech_relay.serve_pending(spoken.append, spool=tmp_path) == 3
    assert spoken == ["first", "second", "third"]


def test_request_requires_something_to_say(tmp_path):
    with pytest.raises(ValueError):
        speech_relay.request(spool=tmp_path)


@pytest.mark.parametrize("outcome", ["success", "failed", "cancelled"])
def test_strict_cancellable_request_reports_actual_playback(tmp_path, outcome):
    cancel, playing = threading.Event(), threading.Event()
    results = []

    def speak(text, cancelled):
        assert text == "hello specimen, are you in trouble"
        playing.set()
        if outcome == "failed":
            raise RuntimeError("speaker stopped")
        if outcome == "cancelled":
            cancel.set()
            deadline = time.monotonic() + 2
            while not cancelled() and time.monotonic() < deadline:
                time.sleep(.005)
            assert cancelled()

    caller = threading.Thread(target=lambda: results.append(speech_relay.request(
        text="hello specimen, are you in trouble", spool=tmp_path,
        cancel=cancel, require_success=True, timeout=3)))
    caller.start()
    deadline = time.monotonic() + 2
    while not list(tmp_path.glob("*.json")) and time.monotonic() < deadline:
        time.sleep(.005)
    speech_relay.serve_pending(lambda _: pytest.fail("must use cancellation callback"),
                               spool=tmp_path, speak_cancellable=speak)
    caller.join(2)
    assert not caller.is_alive() and playing.is_set()
    assert results == [outcome == "success"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("updated_owner", [True, False])
def test_probe_checks_cancellation_support_without_speaking(tmp_path, updated_owner):
    results = []

    def unexpected(*_):
        pytest.fail("probe must never speak")

    caller = threading.Thread(target=lambda: results.append(speech_relay.request(
        probe=True, spool=tmp_path, cancel=threading.Event(), require_success=True, timeout=3)))
    caller.start()
    deadline = time.monotonic() + 2
    while not list(tmp_path.glob("*.json")) and time.monotonic() < deadline:
        time.sleep(.005)
    assert speech_relay.serve_pending(unexpected, spool=tmp_path,
        speak_cancellable=unexpected if updated_owner else None) == 0
    caller.join(2)
    assert results == [updated_owner]


def test_strict_request_does_not_accept_an_old_empty_receipt(tmp_path):
    results = []
    caller = threading.Thread(target=lambda: results.append(speech_relay.request(
        probe=True, spool=tmp_path, require_success=True, timeout=2)))
    caller.start()
    deadline = time.monotonic() + 1
    paths = []
    while not paths and time.monotonic() < deadline:
        paths = list(tmp_path.glob("*.json"))
        time.sleep(.005)
    assert paths
    paths[0].with_suffix(".done").touch()
    caller.join(2)
    assert results == [False]


def test_led_status_round_trips_and_expires(tmp_path):
    speech_relay.post_led_status("listening", ttl=10.0, spool=tmp_path)
    assert speech_relay.read_led_status(spool=tmp_path) == "listening"
    later = lambda: time.time() + 11.0
    assert speech_relay.read_led_status(spool=tmp_path, now=later) is None

    speech_relay.post_led_status("idle", spool=tmp_path)
    assert speech_relay.read_led_status(spool=tmp_path) is None


def test_led_emergency_round_trips_expires_and_ignores_status(tmp_path):
    speech_relay.post_led_emergency(True, ttl=3.0, spool=tmp_path)
    # A check-in changing or releasing its color must not end the emergency.
    speech_relay.post_led_status("listening", spool=tmp_path)
    speech_relay.post_led_status("idle", spool=tmp_path)
    assert speech_relay.read_led_emergency(spool=tmp_path) is True
    later = lambda: time.time() + 60.0
    assert speech_relay.read_led_emergency(spool=tmp_path, now=later) is False

    speech_relay.post_led_emergency(True, spool=tmp_path)
    speech_relay.post_led_emergency(False, spool=tmp_path)
    assert speech_relay.read_led_emergency(spool=tmp_path) is False
