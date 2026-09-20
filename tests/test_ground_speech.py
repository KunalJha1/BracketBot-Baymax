from contextlib import contextmanager
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from ground_speech import GroundSpeech


def test_busy_speaker_uses_relay_and_requires_success_before_completing():
    calls = []

    def open_writer():
        raise RuntimeError("Writer for speaker.audio already exists")

    def request(**kwargs):
        calls.append(kwargs)
        return kwargs.get("probe", False)  # Available, but actual playback fails.

    speech = GroundSpeech(None, open_writer, [], relay_request=request)
    speech.reserve()
    assert len(calls) == 1 and calls[0]["probe"] is True
    speech.start()
    assert speech.done.wait(2)
    speech.close()
    assert len(calls) == 2 and calls[1]["text"] == "hello specimen, are you in trouble"
    assert calls[1]["require_success"] is True
    assert "not confirmed" in speech.error


def test_unavailable_relay_is_refused_before_driving():
    def open_writer():
        raise RuntimeError("Writer for speaker.audio already exists")

    speech = GroundSpeech(None, open_writer, [], relay_request=lambda **_: False)
    with pytest.raises(RuntimeError, match="speech relay"):
        speech.reserve()
    speech.close()


def test_unrelated_speaker_errors_are_not_hidden_by_relay():
    def open_writer():
        raise RuntimeError("speaker daemon broken")

    def unexpected(**_):
        pytest.fail("unrelated writer errors must not use relay")

    speech = GroundSpeech(None, open_writer, [], relay_request=unexpected)
    speech.start()
    assert speech.done.wait(2)
    speech.close()
    assert speech.error == "speaker daemon broken"


def test_speech_plays_once_and_releases_writer():
    frames, closed = [], []

    class Writer:
        @contextmanager
        def buf(self):
            frame = {}
            yield frame
            frames.append(frame["audio"].copy())

    @contextmanager
    def open_writer():
        try:
            yield Writer()
        finally:
            closed.append(True)

    cfg = SimpleNamespace(chunk_size=10, sample_rate=1000, channels=1)
    speech = GroundSpeech(cfg, open_writer, [np.ones(10, dtype=np.int16)] * 3)
    speech.start()
    speech.start()
    assert speech.done.wait(2)
    speech.close()
    assert len(frames) == 3 and closed == [True] and speech.error is None


def test_cancelled_speech_does_not_stream_remaining_chunks():
    written = threading.Event()
    frames, closed = [], []

    class Writer:
        @contextmanager
        def buf(self):
            frame = {}
            yield frame
            frames.append(frame)
            written.set()

    @contextmanager
    def open_writer():
        try:
            yield Writer()
        finally:
            closed.append(True)

    cfg = SimpleNamespace(chunk_size=10, sample_rate=10, channels=1)
    speech = GroundSpeech(cfg, open_writer, [np.ones(10, dtype=np.int16)] * 100)
    speech.start()
    assert written.wait(2)
    speech.close()
    assert speech.done.is_set() and speech.error is None
    assert 0 < len(frames) < 100 and closed == [True]
