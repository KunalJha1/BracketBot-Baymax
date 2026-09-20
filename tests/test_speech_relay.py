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
