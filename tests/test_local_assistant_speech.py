from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("cancel_before_start", [False, True])
def test_assistant_stops_streaming_relay_speech_on_cancellation(monkeypatch, cancel_before_start):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "bbapps/greeter"))
    monkeypatch.setitem(sys.modules, "bbos", SimpleNamespace(Config=object, Reader=object, Type=object, Writer=object))
    module_name = "bbapps.greeter.local_assistant"
    previous_modules = set(sys.modules)
    assistant = importlib.import_module(module_name)
    frames, rendered = [], []

    class Synth:
        def synthesize(self, text, _rate):
            rendered.append(text)
            return np.ones(1000, dtype=np.int16)

    class Writer:
        @contextmanager
        def buf(self):
            frame = {}
            yield frame
            frames.append(frame["audio"])

    monkeypatch.setattr(assistant, "time", SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None))
    try:
        assistant.play_speech(Writer(), Synth(), "hello specimen, are you in trouble",
            SimpleNamespace(sample_rate=1000, chunk_size=10, channels=1), 1., stream=False,
            cancelled=lambda: cancel_before_start or bool(frames))
        assert len(frames) == (0 if cancel_before_start else 1)
        assert len(rendered) == (0 if cancel_before_start else 1)
    finally:
        for name in set(sys.modules) - previous_modules:
            if name.startswith("bbapps.greeter."):
                sys.modules.pop(name, None)
