from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


def load_runtime(monkeypatch, writer_factory=None):
    fake_bbos = ModuleType("bbos")
    fake_bbos.Config = object
    fake_bbos.Reader = object
    fake_bbos.Type = lambda name: name
    fake_bbos.Writer = writer_factory or object
    monkeypatch.setitem(sys.modules, "bbos", fake_bbos)
    sys.modules.pop("bbapps.greeter.gesture_runtime", None)
    return importlib.import_module("bbapps.greeter.gesture_runtime")


def test_set_arms_limp_disables_both_arms(monkeypatch):
    writes = {}

    class FakeWriter:
        def __init__(self, topic, data_type, keeptime):
            self.topic = topic

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @contextmanager
        def buf(self):
            data = {
                "enable": np.ones(8, dtype=np.bool_),
                "tau_mode": np.ones(8, dtype=np.bool_),
                "compliance_mode": True,
            }
            yield data
            writes[self.topic] = data

    runtime = load_runtime(monkeypatch, FakeWriter)
    try:
        runtime.set_arms_limp()
    finally:
        sys.modules.pop("bbapps.greeter.gesture_runtime", None)

    assert set(writes) == {"arm_left.torque", "arm_right.torque"}
    for data in writes.values():
        assert not data["enable"].any()
        assert not data["tau_mode"].any()
        assert data["compliance_mode"] is False


def test_goodbye_plays_wave_before_making_both_arms_limp(monkeypatch):
    runtime = load_runtime(monkeypatch)
    events = []
    assert runtime.recorded_movement_name("goodbye") == "wave"
    controller = runtime.RecordedGestureController(Path("movements"))
    monkeypatch.setattr(controller, "_load", lambda name: ["wave frames"])
    monkeypatch.setattr(
        runtime,
        "prepare_recorded_movement",
        lambda frames: SimpleNamespace(sides=("left",)),
    )
    monkeypatch.setattr(
        runtime,
        "play_recorded_movement",
        lambda name, plan, cancel, shutdown: events.append(("wave", name)),
    )
    monkeypatch.setattr(
        runtime,
        "set_arms_limp",
        lambda: events.append(("limp", ("left", "right"))),
    )
    try:
        started, message = controller.start("goodbye")
        controller._thread.join(timeout=1.0)
    finally:
        sys.modules.pop("bbapps.greeter.gesture_runtime", None)

    assert started is True
    assert message == "Started goodbye"
    assert events == [
        ("wave", "goodbye"),
        ("limp", ("left", "right")),
    ]


def test_demo_arm_reservation_blocks_voice_gestures(monkeypatch, tmp_path):
    runtime = load_runtime(monkeypatch)
    reservation = tmp_path / "arm-reserved"
    reservation.write_text("packing\n")
    monkeypatch.setattr(runtime, "ARM_RESERVATION_PATH", reservation)
    controller = runtime.RecordedGestureController(Path("movements"))
    try:
        assert controller.start("wave") == (
            False,
            "My arms are busy packing, but I can still answer questions.",
        )
        assert controller.preflight("wave") == (
            False,
            "My arms are busy packing, but I can still answer questions.",
        )
    finally:
        sys.modules.pop("bbapps.greeter.gesture_runtime", None)


def test_stop_only_requests_cancellation_while_a_movement_is_running(monkeypatch):
    runtime = load_runtime(monkeypatch)
    controller = runtime.RecordedGestureController(Path("movements"))
    try:
        assert controller.stop() == (False, "No movement is running.")
        controller._lock.acquire()
        assert controller.stop() == (True, "Okay. Stopping safely.")
        assert controller._cancel.is_set()
    finally:
        if controller._lock.locked():
            controller._lock.release()
        sys.modules.pop("bbapps.greeter.gesture_runtime", None)
