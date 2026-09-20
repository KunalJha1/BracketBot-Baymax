"""BBOS schema and runner integration checks without connecting to a robot."""

from contextlib import contextmanager
import json
import socket
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from follow_calibration import Calibration, load_calibration
from probe_follow import cloud_summary, save_cloud
import robot_follow


def profile():
    return dict(schema_version=1, robot_id=socket.gethostname(),
                evidence="synthetic test fixture, not hardware evidence",
                left_sign=-1, self_mask=[], wheel_order=[0, 1],
                wheel_signs=[1, 1], motion_speed_limit=0.15)


@pytest.mark.parametrize("index_field", ["idx_2d", "mask", None])
def test_probe_saves_current_legacy_and_index_free_clouds(tmp_path, index_field):
    fields = [("num_points", "i4"), ("points", "f4", (4, 3)),
              ("colors", "u1", (4, 3)), ("timestamp", "i8")]
    if index_field:
        fields.append((index_field, "i4", (4,)))
    live = np.zeros((), dtype=fields)
    live["num_points"] = 2
    live["points"][:2] = [[0.1, 1.0, 1.2], [0.2, 1.0, 1.3]]
    data = {name: np.array(live[name]).copy() for name in live.dtype.names}
    path = tmp_path / "points.npz"
    save_cloud(path, data)
    with np.load(path) as saved:
        assert saved["points"].shape == (2, 3)
        assert saved["colors"].shape == (2, 3)
        if index_field:
            assert saved[index_field].shape == (2,)
    np.testing.assert_allclose(robot_follow.cloud(live), data["points"][:2])


def test_nearby_points_are_not_automatically_classified_as_robot_body():
    report = cloud_summary([[0.1, 0.2, 1.1], [0.1, -1.0, 1.1]], False)
    assert report["nearby_points"] == 1
    assert "self_points" not in report
    assert "inspect" in report["nearby_note"]
    assert cloud_summary([[0, 1, 1.2]], True)["verdict"].startswith("INCONCLUSIVE")


@pytest.mark.parametrize("change", [
    {"robot_id": "a-different-robot"}, {"left_sign": 0}, {"left_sign": True},
    {"self_mask": [[0, 1, 0, 1, 0, float("nan")]]},
    {"self_mask": [[1, 0, 0, 1, 0, 1]]}, {"self_mask": [[0, 1]]},
    {"wheel_order": [0, 0]}, {"wheel_signs": [1, 0]},
    {"motion_speed_limit": 0.31}, {"evidence": ""}, {"schema_version": 2},
])
def test_invalid_or_wrong_robot_calibration_is_rejected(tmp_path, change):
    path = tmp_path / "follow.json"
    path.write_text(json.dumps(profile() | change))
    with pytest.raises(RuntimeError, match="calibration unavailable or invalid"):
        load_calibration(path)


def test_missing_calibration_prevents_motion_before_importing_bbos(tmp_path):
    args = robot_follow.parse_args(["--calibration", str(tmp_path / "missing.json")])
    with pytest.raises(RuntimeError, match="motion is disabled"):
        robot_follow.run(args)


def test_self_mask_is_applied_before_both_detection_and_corridor_checks():
    rng = np.random.default_rng(12)
    body = np.column_stack([rng.uniform(-.2, .2, 900), rng.uniform(.31, .42, 900), rng.uniform(.1, 1.6, 900)])
    person = body + [0, 1.2, 0]
    points = np.vstack([body, person])
    cal = Calibration(self_mask=((0, .45, -.25, .25, .05, 1.7),))
    perception = robot_follow.perceive(points, 0, calibration=cal)
    assert len(perception.people) == 1
    assert perception.people[0].forward > 1
    assert len(perception.points) == len(person)
    np.testing.assert_allclose(perception.points[:, 0], person[:, 1])


def test_calibrated_left_sign_reaches_perception():
    points = np.array([[.5, 1.0, 1.2]])
    assert robot_follow.perceive(points, 0, calibration=Calibration(left_sign=1)).points[0, 1] == .5


@pytest.mark.parametrize("data", [
    {"num_points": -1, "points": np.zeros((4, 3))},
    {"num_points": 5, "points": np.zeros((4, 3))},
    {"num_points": 1, "points": np.zeros((4, 2))},
    {"num_points": 1, "points": np.array([[np.nan, 1, 1]])},
])
def test_invalid_live_cloud_rejected(data):
    with pytest.raises(RuntimeError):
        robot_follow.cloud(data)


@pytest.fixture
def robot(tmp_path, monkeypatch):
    """Run the actual runner/loop against deterministic fake readers and writers."""
    cal = tmp_path / "follow.json"
    cal.write_text(json.dumps(profile()))
    rng = np.random.default_rng(12)
    points = np.column_stack([rng.uniform(-.2, .2, 900), rng.uniform(1.4, 1.6, 900), rng.uniform(.1, 1.7, 900)])
    state = SimpleNamespace(t=0.0, opened=[], writes=[], stale=None, missing=None,
                            broken_cloud=False, cal=cal, log_dir=tmp_path)
    data = {"camera.points": dict(num_points=900, points=points, idx_2d=np.arange(900)),
            "imu.orientation": dict(rpy=np.zeros(3)),
            "drive.state": dict(vel=np.zeros(2)),
            "drive.status": dict(voltage=24.0)}

    class Reader:
        def __init__(self, topic, **kwargs):
            self.topic = topic

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def ready(self):
            return self.topic != state.missing and not (self.topic == state.stale and state.t > .7)

        @property
        def data(self):
            if self.topic == "camera.points" and state.broken_cloud and state.t > .7:
                return dict(num_points=1, points=np.array([[np.nan, 1, 1]]))
            return data[self.topic]

    class Writer(Reader):
        def __init__(self, topic, *_args, **kwargs):
            super().__init__(topic)
            state.opened.append(topic)

        @contextmanager
        def buf(self):
            frame = {}
            yield frame
            state.writes.append((self.topic, frame))

    def sleep(dt):
        state.t += dt
        assert state.t < 10, "runner failed to stop"

    configs = {"drive": SimpleNamespace(wheel_diam=.165, robot_width=.3275),
               "base": SimpleNamespace(low_battery_v=20)}
    monkeypatch.setitem(sys.modules, "bbos", SimpleNamespace(
        Config=configs.__getitem__, Reader=Reader, Writer=Writer, Type=lambda name: name))
    monkeypatch.setattr(robot_follow, "time", SimpleNamespace(
        monotonic=lambda: state.t, sleep=sleep, strftime=lambda _: "test"))
    class InlineWorker(robot_follow.PerceptionWorker):
        """Fake time is single-threaded: process each frame on the control thread."""

        def __enter__(self):
            return self

        def take(self):
            self.step()
            return super().take()

    monkeypatch.setattr(robot_follow, "PerceptionWorker", InlineWorker)
    monkeypatch.setattr(robot_follow, "other_drive_writers", lambda ignore=(): [])
    monkeypatch.setattr(robot_follow, "start_command_reader", lambda _: None)
    monkeypatch.setattr(robot_follow, "STOP_REQUESTED", False)

    def run(*argv):
        args = robot_follow.parse_args(["--calibration", str(cal), "--log-dir", str(tmp_path), *argv])
        robot_follow.run(args)

    state.run = run
    return state


def test_preflight_opens_no_writers(robot, capsys):
    robot.run("--preflight")
    assert robot.opened == []
    assert "PREFLIGHT OK" in capsys.readouterr().out


def test_busy_led_does_not_prevent_follow_and_drive_is_still_zeroed(robot, monkeypatch, capsys):
    bbos = sys.modules["bbos"]
    writer = bbos.Writer

    def open_writer(topic, *args, **kwargs):
        if topic == "led.ctrl":
            raise RuntimeError("Writer for led.ctrl already exists")
        return writer(topic, *args, **kwargs)

    monkeypatch.setattr(bbos, "Writer", open_writer)
    robot.run()
    assert_stopped(robot)
    assert "LEDs already owned" in capsys.readouterr().out


def test_busy_drive_never_sends_commands(robot, monkeypatch):
    def open_writer(topic, *args, **kwargs):
        raise RuntimeError(f"Writer for {topic} already exists")

    monkeypatch.setattr(sys.modules["bbos"], "Writer", open_writer)
    with pytest.raises(RuntimeError, match="drive.ctrl already exists"):
        robot.run()
    assert robot.writes == []


def test_missing_live_input_never_opens_writers(robot):
    robot.missing = "camera.points"
    with pytest.raises(RuntimeError, match="refusing to start"):
        robot.run()
    assert robot.opened == []


def test_uncalibrated_dry_run_never_opens_drive_writer(robot, capsys):
    robot.cal.unlink()
    robot.run("--dry-run")
    assert robot.opened == ["led.ctrl"]
    assert "UNCALIBRATED" in capsys.readouterr().out


def assert_stopped(robot):
    twists = [frame["twist"] for topic, frame in robot.writes if topic == "drive.ctrl"]
    assert any(twist[0] > 0 for twist in twists), "scenario must exercise motion before the stop"
    np.testing.assert_array_equal(twists[-6:], np.zeros((6, 2)))


def test_heartbeat_loss_stops_actual_runner_and_zeros_on_exit(robot, capsys):
    robot.run()
    assert_stopped(robot)
    assert "heartbeat" in capsys.readouterr().out


@pytest.mark.parametrize("topic", ["imu.orientation", "drive.state"])
def test_state_dropout_stops_actual_runner_and_zeros_on_exit(robot, topic):
    robot.stale = topic
    with pytest.raises(RuntimeError, match="stale"):
        robot.run()
    assert_stopped(robot)


def test_invalid_cloud_stops_actual_runner_and_zeros_on_exit(robot):
    robot.broken_cloud = True
    with pytest.raises(RuntimeError, match="non-finite"):
        robot.run()
    assert_stopped(robot)


def test_speed_above_robot_profile_is_refused_before_writers(robot):
    with pytest.raises(RuntimeError, match="calibrated limit"):
        robot.run("--v-max", "0.3")
    assert robot.opened == []


def test_omega_sign_defaults_to_one_and_accepts_only_a_sign(tmp_path):
    path = tmp_path / "follow.json"
    path.write_text(json.dumps(profile()))
    assert load_calibration(path).omega_sign == 1.0
    path.write_text(json.dumps(profile() | {"omega_sign": -1}))
    assert load_calibration(path).omega_sign == -1.0
    path.write_text(json.dumps(profile() | {"omega_sign": 0.5}))
    with pytest.raises(RuntimeError):
        load_calibration(path)
