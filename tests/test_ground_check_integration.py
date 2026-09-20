"""Exercise the real ground producer, runner, PID and speech with a synthetic robot.

Only hardware, inference output (3D joints), eSpeak, and the clock are simulated.
These are software integration tests, not evidence of physical calibration.
"""

from contextlib import contextmanager
from functools import partial
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import robot_follow
from ground_approach import GROUND_LINE, STANDOFF


@pytest.fixture
def ground_robot(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "bbapps" / "emotion_greeter"))
    monkeypatch.syspath_prepend(str(root / "bbapps" / "greeter"))
    spec = importlib.util.spec_from_file_location("ground_integration_vision", root / "bbapps/emotion_greeter/main.py")
    vision = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, vision)
    spec.loader.exec_module(vision)
    from ground_safety import GroundAlertTracker, assess_ground_pose
    import local_voice
    import speech_relay

    alert_path = tmp_path / "ground.json"
    calibration = tmp_path / "follow.json"
    calibration.write_text(json.dumps({
        "schema_version": 1, "robot_id": socket.gethostname(),
        "evidence": "synthetic integration fixture only", "left_sign": -1,
        "self_mask": [], "wheel_order": [0, 1], "wheel_signs": [1, 1], "motion_speed_limit": .15,
    }))
    state = SimpleNamespace(t=0., x=0., y=0., heading=0., v=0., w=0., writes=[], audio=[],
                            synthesized=[], spoken=[], queue=None, fault=None, shared_speaker=False,
                            opened=set(), last_depth=-1., last_vision=-1., last_heartbeat=-1., radius=0.)
    tracker = GroundAlertTracker()
    # A stationary body in world (forward, left, up), displaced left to require steering.
    pose_base = {
        5: [-.25, 1.8, .22], 6: [.25, 1.8, .23], 7: [-.4, 1.55, .18], 8: [.4, 1.55, .2],
        11: [-.2, 1.25, .18], 12: [.2, 1.25, .2], 13: [-.18, .8, .14], 14: [.18, .8, .15],
        15: [-.16, .35, .10], 16: [.16, .35, .11],
    }
    world = {i: np.array([p[1] + 1.1, -p[0] + .9, p[2]]) for i, p in pose_base.items()}

    def publish():
        if state.t - state.last_vision < .25 or (state.fault == "vision" and state.t >= 4):
            return
        state.last_vision = state.t
        c, s = math.cos(state.heading), math.sin(state.heading)
        joints = {}
        for i, p in world.items():
            dx, dy = p[0] - state.x, p[1] - state.y
            joints[i] = np.array([s * dx - c * dy, c * dx + s * dy, p[2]])
        assessment = assess_ground_pose(joints)
        latch = tracker.update({1: assessment}, state.t)[1]
        observation = dict(track_id=1, state=assessment.state, latch_status=latch,
                           base_position=assessment.base_position, body_radius_m=assessment.body_radius_m,
                           confidence=assessment.confidence)
        vision.publish_ground_safety_file(alert_path, latch, [{"track_id": 1}] if latch == "alert" else [],
            camera_timestamp_ns=int((100 + state.t) * 1e9), map_epoch=1,
            observations=[observation], depth_aligned=True)
        # The controller never shrinks its envelope. The coordinate-wise median
        # changes slightly while turning, even with the same visible joints.
        state.radius = max(state.radius, assessment.body_radius_m)
        state.clearance = math.hypot(*assessment.base_position[:2]) - state.radius

    def sleep(dt):
        dt = max(dt, .002)
        state.heading += state.w * dt
        state.x += state.v * math.cos(state.heading) * dt
        state.y += state.v * math.sin(state.heading) * dt
        state.t += dt
        assert state.t < 110, "runner failed to finish the synthetic approach"
        publish()
        if state.queue is not None:
            if (state.fault == "stop" and state.t >= 4) or (state.fault and state.t >= 7):
                state.queue.put('{"type":"stop"}')
            if state.t - state.last_heartbeat >= .2 and not (state.fault == "heartbeat" and state.t >= 4):
                state.queue.put('{"type":"heartbeat"}')
                state.last_heartbeat = state.t
        time.sleep(.0005)  # Let real speech and relay threads run alongside fake robot time.

    monkeypatch.setattr(vision, "time", SimpleNamespace(time=lambda: 100 + state.t))
    monkeypatch.setattr(robot_follow, "time", SimpleNamespace(
        monotonic=lambda: state.t, time=lambda: 100 + state.t, sleep=sleep, strftime=lambda _: "integration"))
    monkeypatch.setattr(robot_follow, "STOP_REQUESTED", False)
    monkeypatch.setattr(robot_follow, "other_drive_writers", lambda *_: [])
    monkeypatch.setattr(robot_follow, "start_command_reader", lambda q: setattr(state, "queue", q))
    publish()

    rng = np.random.default_rng(40)
    forward = rng.uniform(.2, 1.8, 1800)
    floor = np.column_stack([rng.uniform(-.9, .9, len(forward)), forward,
                             .1 * forward + .02 + rng.normal(0, .008, len(forward))])

    class Reader:
        def __init__(self, topic, **_):
            self.topic = topic

        def __enter__(self): return self
        def __exit__(self, *_): pass

        def ready(self):
            if self.topic != "camera.points": return True
            if state.fault == "depth" and state.t >= 4: return False
            if state.t - state.last_depth < .1: return False
            state.last_depth = state.t
            return True

        @property
        def data(self):
            if self.topic == "imu.orientation": return {"rpy": np.zeros(3)}
            if self.topic == "drive.status": return {"voltage": 24.}
            if self.topic == "drive.state":
                return {"vel": np.array([state.v - state.w * .3275 / 2,
                                          state.v + state.w * .3275 / 2]) / (math.pi * .165)}
            points = floor
            if state.fault == "obstacle" and state.t >= 4:
                points = np.vstack([floor, np.tile([0, .5, .20], (60, 1))])
            return {"timestamp": np.datetime64(int((100 + state.last_depth) * 1e9), "ns"),
                    "num_points": len(points), "points": points}

    class Writer:
        def __init__(self, topic, *_, **__): self.topic = topic

        def __enter__(self):
            if self.topic == "led.ctrl" or (self.topic == "speaker.audio" and state.shared_speaker):
                raise RuntimeError(f"Writer for {self.topic} already exists")
            assert self.topic not in state.opened, "two writers attempted to own a topic"
            state.opened.add(self.topic)
            return self

        def __exit__(self, *_): state.opened.remove(self.topic)

        @contextmanager
        def buf(self):
            frame = {}
            yield frame
            if self.topic == "drive.ctrl":
                state.v, state.w = (float(v) for v in frame["twist"])
                state.writes.append((state.t, state.v, state.w))
            else:
                assert (state.v, state.w) == (0, 0), "speech must hold the wheels at zero"
                state.audio.append(frame["audio"])

    speaker_cfg = SimpleNamespace(chunk_size=10, sample_rate=1000, channels=1)
    configs = {"drive": SimpleNamespace(wheel_diam=.165, robot_width=.3275),
               "base": SimpleNamespace(low_battery_v=20), "speaker": speaker_cfg}
    monkeypatch.setitem(sys.modules, "bbos", SimpleNamespace(
        Reader=Reader, Writer=Writer, Type=lambda x: x, Config=configs.__getitem__))

    class Synth:
        def synthesize(self, text, _rate):
            state.synthesized.append(text)
            return np.ones(30, dtype=np.int16)

    monkeypatch.setattr(local_voice, "EspeakSynthesizer", Synth)
    monkeypatch.setattr(speech_relay, "request", partial(speech_relay.request, spool=tmp_path / "speech"))

    class InlineWorker(robot_follow.PerceptionWorker):
        def __enter__(self): return self
        def take(self):
            self.step()
            return super().take()

    monkeypatch.setattr(robot_follow, "PerceptionWorker", InlineWorker)
    finished = threading.Event()

    def speak(text, cancelled):
        assert not cancelled() and (state.v, state.w) == (0, 0)
        state.spoken.append(text)

    def serve():
        while not finished.wait(.002):
            speech_relay.serve_pending(lambda _: pytest.fail("must use cancellable speech"),
                spool=tmp_path / "speech", speak_cancellable=speak)

    owner = threading.Thread(target=serve, daemon=True)
    owner.start()

    def run():
        robot_follow.run(robot_follow.parse_args([
            "--ground-approach", "--ground-alert-file", str(alert_path),
            "--calibration", str(calibration), "--log-dir", str(tmp_path)]))

    state.run = run
    yield state
    finished.set()
    owner.join(2)


@pytest.mark.parametrize("shared_speaker", [False, True])
def test_detection_to_approach_stop_and_one_utterance(ground_robot, shared_speaker, capsys):
    robot = ground_robot
    robot.shared_speaker = shared_speaker
    robot.run()
    log = capsys.readouterr().out
    assert "ground approach complete" in log
    assert "LEDs already owned" in log
    assert any(v > 0 for _, v, _ in robot.writes)
    assert any(w > 0 for _, _, w in robot.writes)
    assert all(0 <= v <= .050001 and abs(w) <= .200001 for _, v, w in robot.writes)
    assert all(v == 0 and w == 0 for t, v, w in robot.writes if t < 2)  # confirmation hold
    assert STANDOFF - .02 <= robot.clearance <= STANDOFF + .06
    assert robot.synthesized == [GROUND_LINE]
    assert robot.spoken == ([GROUND_LINE] if shared_speaker else [])
    assert bool(robot.audio) != shared_speaker
    assert robot.opened == set()
    assert all((v, w) == (0, 0) for _, v, w in robot.writes[-6:])


@pytest.mark.parametrize("fault,stopped_by", [
    ("obstacle", 4.2), ("depth", 4.6), ("vision", 6.2), ("heartbeat", 5.1), ("stop", 4.1),
])
def test_faults_stop_real_runner_without_speaking(ground_robot, fault, stopped_by, capsys):
    robot = ground_robot
    robot.fault = fault
    robot.run()
    log = capsys.readouterr().out
    assert "ground approach complete" not in log
    assert any(v > 0 or w != 0 for _, v, w in robot.writes)
    assert all((v, w) == (0, 0) for t, v, w in robot.writes if t >= stopped_by)
    assert all((v, w) == (0, 0) for _, v, w in robot.writes[-6:])
    assert robot.audio == [] and robot.spoken == []
    assert robot.opened == set()
