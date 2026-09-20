import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import robot_follow


def test_defaults_are_the_bring_up_limits():
    args = robot_follow.parse_args([])
    assert (args.gap, args.v_max, args.dry_run, args.rotate_only) == (1.0, 0.15, False, False)
    assert robot_follow.loop_config(args).v_max == 0.15


def test_rotate_only_holds_forward_speed_at_zero():
    assert robot_follow.loop_config(robot_follow.parse_args(["--rotate-only"])).v_max == 0.0


@pytest.mark.parametrize("argv", [
    ["--gap", "2.0"], ["--gap", "0.3"], ["--v-max", "0.5"], ["--v-max", "0"], ["--no-heartbeat"],
])
def test_unsafe_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        robot_follow.parse_args(argv)


def test_only_a_dry_run_may_skip_the_heartbeat():
    assert robot_follow.parse_args(["--dry-run", "--no-heartbeat"]).no_heartbeat is True


def test_perceive_finds_a_person_in_a_base_frame_cloud():
    rng = np.random.default_rng(3)
    phi = rng.uniform(-math.pi / 2, math.pi / 2, 900)
    forward = 1.4 - 0.18 * np.cos(phi)  # front half of a person standing 1.4 m ahead
    left = 0.2 + 0.18 * np.sin(phi)  # ... and 0.2 m to the left
    z = rng.uniform(0.05, 1.75, 900)
    points_base = np.column_stack([-left, forward, z])  # base +x points right (BASE_LEFT_SIGN = -1)

    perception = robot_follow.perceive(points_base, 5.0)

    assert perception.t == 5.0
    assert perception.points.shape == (900, 3)
    [person] = perception.people
    assert person.forward == pytest.approx(1.4 - 0.7 * 0.18, abs=0.05)
    assert person.left == pytest.approx(0.2, abs=0.05)
    assert person.hist is None


def test_perceive_with_nobody_there():
    assert robot_follow.perceive(np.empty((0, 3)), 1.0).people == ()


def test_runner_imports_without_bbos():
    assert "bbos" not in sys.modules


def test_clamped_twist_reclamps_out_of_range_loop_output():
    cfg = robot_follow.FollowConfig(v_max=0.15, omega_max=0.8)
    stub_out = SimpleNamespace(v=-0.5, omega=5.0)

    v, omega = robot_follow.clamped_twist(stub_out, cfg)

    assert v == 0.0
    assert omega == 0.8


def test_other_drive_writers_ignores_self_and_parent_pids(monkeypatch):
    my_pid = os.getpid()
    parent_pid = os.getppid()
    other_pid = 999999

    def fake_run(command, **kwargs):
        if "robot_follow.py" in command:
            stdout = (
                f"{my_pid} uv run --no-sync --project ~/bbos python /tmp/robot_follow.py\n"
                f"{parent_pid} python /tmp/robot_follow.py\n"
                f"{other_pid} python /tmp/robot_follow.py --gap 1.0\n"
            )
        else:
            stdout = ""
        return SimpleNamespace(stdout=stdout, returncode=0)

    monkeypatch.setattr(robot_follow.subprocess, "run", fake_run)
    found = robot_follow.other_drive_writers()

    assert len(found) == 1
    assert str(other_pid) in found[0]


def test_ground_mode_is_capped_independently_of_follow_speed():
    args = robot_follow.parse_args(["--ground-approach", "--v-max", "0.3"])
    assert robot_follow.loop_config(args).v_max == 0.05
    assert robot_follow.loop_config(robot_follow.parse_args(["--ground-approach", "--rotate-only"])).v_max == 0


@pytest.mark.parametrize("age,count,valid", [(0.1, 200, True), (2, 200, False), (-1, 200, False), (0.1, 0, False)])
def test_ground_depth_checks_capture_time_and_coverage(age, count, valid):
    data = {"timestamp": np.datetime64(int((100 - age) * 1e9), "ns"), "num_points": count,
            "points": np.tile([0, 3, 0], (count, 1))}
    result = robot_follow.ground_perception(data, 10, 100, robot_follow.Calibration())
    assert (result is not None) == valid
    if valid:
        assert result.t == pytest.approx(9.9)


def test_old_vision_service_refused_before_motion(tmp_path):
    path = tmp_path / "ground.json"
    path.write_text('{"schema_version":1,"status":"alert"}')
    with pytest.raises(RuntimeError, match="deploy the updated emotion_greeter"):
        robot_follow.check_ground_service(path)


@pytest.mark.parametrize("arrive", [True, False])
def test_ground_runner_holds_zero_while_speaking_and_stops_on_signal(tmp_path, monkeypatch, arrive):
    import threading
    from ground_approach import GroundTarget

    clock = [0.0]
    monkeypatch.setattr(robot_follow.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(robot_follow.time, "time", lambda: 100 + clock[0])

    def sleep(seconds):
        clock[0] += max(0.02, seconds)
        if not arrive and clock[0] > 1:
            robot_follow.STOP_REQUESTED = True
        assert clock[0] < 3, "runner did not terminate"

    monkeypatch.setattr(robot_follow.time, "sleep", sleep)
    monkeypatch.setattr(robot_follow, "STOP_REQUESTED", False)
    monkeypatch.setattr(robot_follow, "read_ground_target", lambda *_: GroundTarget(
        "test", 1, 100 + clock[0], 1.8 if arrive else 3, 0, 0.8))
    writes = []
    monkeypatch.setattr(robot_follow, "write_twist", lambda _writer, v, w: writes.append((v, w)))
    monkeypatch.setattr(robot_follow, "write_led", lambda *_: None)

    class Reader:
        data = {"rpy": np.zeros(3), "vel": np.zeros(2), "num_points": 200,
                "points": np.tile([0, 3, 0], (200, 1))}

        def ready(self):
            self.data["timestamp"] = np.datetime64(int((100 + clock[0]) * 1e9), "ns")
            return True

    class Speech:
        done = threading.Event()
        error = None
        calls = 0

        def start(self):
            assert writes[-1] == (0, 0)
            assert clock[0] >= 0.6
            self.calls += 1
            self.done.set()

    args = robot_follow.parse_args(["--ground-approach", "--dry-run", "--no-heartbeat", "--log-dir", str(tmp_path)])
    speech = Speech()
    robot_follow.control_loop(args, robot_follow.loop_config(args), [Reader(), Reader(), Reader()],
                              object(), object(), 0.2, 0.3, speech=speech)
    assert speech.calls == int(arrive)
    assert writes[-1] == (0, 0)
    if not arrive:
        assert any(v > 0 for v, _ in writes)


class OneFrame:
    def __init__(self):
        self.data, self.left = "frame", 1

    def ready(self):
        self.left -= 1
        return self.left >= 0


def wait_for(worker):
    import time
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            latest = worker.take()
        except Exception as exc:
            return exc
        if latest is not None:
            return latest
        time.sleep(0.005)
    raise AssertionError("worker produced nothing")


def test_perception_worker_hands_over_each_frame_once():
    with robot_follow.PerceptionWorker(OneFrame(), lambda data, t: (data.upper(), "[]")) as worker:
        perception, ms, candidates = wait_for(worker)
        assert (perception, candidates) == ("FRAME", "[]") and ms >= 0
        assert worker.take() is None


def test_perception_worker_failure_reaches_the_control_thread():
    def boom(data, t):
        raise RuntimeError("non-finite camera.points")

    with robot_follow.PerceptionWorker(OneFrame(), boom) as worker:
        assert isinstance(wait_for(worker), RuntimeError)
        with pytest.raises(RuntimeError):
            worker.take()
