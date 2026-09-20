from contextlib import contextmanager
from types import SimpleNamespace
import threading

import numpy as np

from ground_speech import GroundSpeech


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
