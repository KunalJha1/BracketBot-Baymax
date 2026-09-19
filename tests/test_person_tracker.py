import threading

import pytest

from scripts import person_tracker as pt


def test_pixel_bearing_is_zero_ahead_and_positive_to_the_left():
    cx = float(pt.DEFAULT_K[0, 2])
    cy = float(pt.DEFAULT_K[1, 2])

    assert abs(pt.pixel_bearing_deg(cx, cy)) < 2.0
    assert pt.pixel_bearing_deg(cx - 300, cy) > 20.0
    assert pt.pixel_bearing_deg(cx + 300, cy) < -20.0


def test_distance_band_uses_face_width():
    fx = float(pt.DEFAULT_K[0, 0])
    assert pt.estimate_distance_m(fx * 0.16) == pytest.approx(1.0)
    assert pt.distance_band(2.0, "scan") == "far"
    assert pt.distance_band(0.8, "scan") == "ok"
    assert pt.distance_band(0.3, "gesture") == "close"
    assert pt.distance_band(1.0, "gesture") == "far"


def test_pick_face_prefers_largest_confident_face():
    faces = [(0, 0, 200, 200, 0.5), (0, 0, 40, 40, 0.9), (0, 0, 80, 80, 0.8)]
    assert pt.pick_face(faces)[2] == 80
    assert pt.pick_face([(0, 0, 90, 90, 0.2)]) is None


def test_search_plan_tries_hint_first_and_never_repeats_the_start_view():
    assert pt.search_plan() == [60.0] * 5
    assert pt.search_plan(-90) == [-90.0, -60.0, -60.0, -60.0]
    assert pt.search_plan(10) == [60.0] * 5      # tiny hints are already in view
    assert sum(abs(step) for step in pt.search_plan(150)) <= 330


def test_turn_command_is_bounded_and_signed():
    assert pt.turn_command(2.0) == 0.0
    assert pt.turn_command(90.0) == pt.MAX_TURN_RAD_S
    assert pt.turn_command(-10.0) == -pt.MIN_TURN_RAD_S
    assert pt.slew(0.0, 0.6, 0.02) == pytest.approx(0.03)


class FakeWriter:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


class FakeRobot:
    """A robot in a room with one person at a fixed world heading."""

    def __init__(self, person_yaw=None, refuse=None):
        self.heading = 0.0
        self.person_yaw = person_yaw
        self.refuse = refuse
        self.yaw_sign = 1.0
        self.turns = []
        self.writers = []
        self.bbos = self

    def Writer(self, *args, **kwargs):  # noqa: N802 - mirrors bbos
        writer = FakeWriter()
        self.writers.append(writer)
        return writer

    def Type(self, name):  # noqa: N802
        return name

    def yaw(self):
        return self.heading

    def preflight(self):
        if self.refuse:
            raise pt.Refused(self.refuse)

    def look(self, frames=2):
        if self.person_yaw is None:
            return None
        bearing = pt.wrap_deg(self.person_yaw - self.heading)
        if abs(bearing) > 55:
            return None
        return {"bearing_deg": bearing, "face_width_px": 110, "distance_m": 0.65, "score": 0.9}

    def turn_by(self, writer, delta, cancel):
        if cancel.is_set():
            raise pt.Cancelled()
        self.turns.append(delta)
        self.heading = pt.wrap_deg(self.heading + delta)


def test_acquire_does_not_move_when_person_is_already_centered():
    robot = FakeRobot(person_yaw=3.0)

    result = pt.Tracker(robot).acquire("scan", None, threading.Event())

    assert result["found"] and result["centered"]
    assert robot.turns == [] and robot.writers == []


def test_acquire_sweeps_to_find_and_center_a_person_behind():
    robot = FakeRobot(person_yaw=170.0)

    result = pt.Tracker(robot).acquire("scan", None, threading.Event())

    assert result["found"] and result["centered"]
    assert result["distance"] == "ok"
    assert abs(pt.wrap_deg(robot.heading - 170.0)) <= pt.CENTER_TOLERANCE_DEG
    assert all(writer.closed for writer in robot.writers)


def test_acquire_turns_toward_remembered_person_first():
    robot = FakeRobot(person_yaw=-100.0)
    tracker = pt.Tracker(robot)
    robot.heading = -100.0
    tracker.observe()                     # saw them earlier
    robot.heading = 0.0                   # then the robot turned away

    result = tracker.acquire("gesture", None, threading.Event())

    assert result["found"]
    assert robot.turns[0] == pytest.approx(-100.0)


def test_acquire_reports_nobody_after_one_look_around():
    robot = FakeRobot(person_yaw=None)

    result = pt.Tracker(robot).acquire("scan", None, threading.Event())

    assert result["found"] is False
    assert sum(abs(turn) for turn in robot.turns) <= 330


def test_refusal_without_a_visible_person_raises_but_visible_person_is_used():
    with pytest.raises(pt.Refused):
        pt.Tracker(FakeRobot(person_yaw=None, refuse="lean mode")).acquire(
            "scan", None, threading.Event()
        )

    robot = FakeRobot(person_yaw=30.0, refuse="lean mode")
    result = pt.Tracker(robot).acquire("scan", None, threading.Event())
    assert result["found"] and not result["centered"]
    assert robot.turns == []


def test_cancel_stops_the_search():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(pt.Cancelled):
        pt.Tracker(FakeRobot(person_yaw=None)).acquire("scan", None, cancel)
