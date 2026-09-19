import dataclasses
import json
import math

import numpy as np
import pytest

from follow_core import (
    FollowConfig,
    LockOn,
    PersonObservation,
    Pose2D,
    Tracker,
    hist_distance,
)

CFG = FollowConfig()
FAST = dataclasses.replace(CFG, v_max=0.30)
FRAME = 1 / 15


def person(forward, left=0.0, raised=False, hist=None):
    return PersonObservation(forward, left, 0.9, raised, hist)


def onehot(index):
    hist = np.zeros(64)
    hist[index] = 1.0
    return hist


# --- geometry -------------------------------------------------------------

def test_pose_local_and_odom_round_trip():
    pose = Pose2D(1.0, 2.0, 0.5)
    x, y = pose.to_odom(0.8, -0.3)
    assert pose.to_local(x, y) == pytest.approx((0.8, -0.3))


def test_pose_integrates_forward_and_turning_motion():
    pose = Pose2D()
    pose.integrate(0.2, 0.0, 1.0)
    assert (pose.x, pose.y, pose.h) == pytest.approx((0.2, 0.0, 0.0))
    pose.integrate(0.0, math.pi / 2, 1.0)
    assert pose.h == pytest.approx(math.pi / 2)
    assert pose.to_odom(1.0, 0.0) == pytest.approx((0.2, 1.0))


def test_histogram_distance():
    assert hist_distance(onehot(3), onehot(3)) == pytest.approx(0.0)
    assert hist_distance(onehot(3), onehot(40)) == pytest.approx(1.0)
    assert hist_distance(None, onehot(3)) == 0.0


# --- lock-on --------------------------------------------------------------

def run_lock_on(frames):
    """frames: list of people lists at 15 Hz. Returns (time, result) of the first lock or None."""
    lock_on = LockOn(CFG)
    for i, people in enumerate(frames):
        t = i * FRAME
        positions = [(p.forward, p.left) for p in people]
        result = lock_on.update(t, people, positions)
        if result is not None:
            return t, result
    return None


def test_hand_raised_for_half_a_second_locks_that_person():
    frames = [[person(1.2, 0.1, raised=True), person(1.5, -0.6)] for _ in range(15)]
    locked = run_lock_on(frames)
    assert locked is not None
    t, (obs, xy) = locked
    assert t >= CFG.lock_window
    assert xy == pytest.approx((1.2, 0.1))


def test_brief_raise_does_not_lock():
    frames = [[person(1.2, raised=i < 4)] for i in range(20)]
    assert run_lock_on(frames) is None


def test_two_people_raising_hands_lock_neither():
    frames = [[person(1.2, 0.5, raised=True), person(1.2, -0.5, raised=True)] for _ in range(20)]
    assert run_lock_on(frames) is None


def test_people_outside_lock_zone_are_ignored():
    far = [[person(3.0, raised=True)] for _ in range(20)]
    wide = [[person(0.5, 1.2, raised=True)] for _ in range(20)]
    assert run_lock_on(far) is None
    assert run_lock_on(wide) is None


# --- tracker --------------------------------------------------------------

def test_tracker_estimates_walking_speed():
    tracker = Tracker(CFG)
    tracker.start(0.0, (1.0, 0.0), onehot(3))
    for i in range(1, 46):
        t = i * FRAME
        assert tracker.update(t, [person(1.0 + 0.3 * t, hist=onehot(3))], [(1.0 + 0.3 * t, 0.0)]) == "updated"
    view = tracker.track(45 * FRAME, Pose2D())
    assert view.v_radial == pytest.approx(0.3, abs=0.05)
    assert view.range == pytest.approx(1.0 + 0.3 * 3.0, abs=0.05)


def test_bystander_in_different_clothes_is_not_accepted():
    tracker = Tracker(CFG)
    tracker.start(0.0, (1.0, 0.0), onehot(3))
    bystander = person(1.02, 0.02, hist=onehot(40))
    assert tracker.update(FRAME, [bystander], [(1.02, 0.02)]) == "coasted"
    assert tracker.age(FRAME) == pytest.approx(FRAME)


def test_two_lookalikes_inside_the_gate_are_ambiguous():
    tracker = Tracker(CFG)
    tracker.start(0.0, (1.0, 0.0), onehot(3))
    people = [person(1.0, 0.05, hist=onehot(3)), person(1.0, -0.05, hist=onehot(3))]
    assert tracker.update(FRAME, people, [(1.0, 0.05), (1.0, -0.05)]) == "ambiguous"


def test_track_view_does_not_modify_the_filter():
    tracker = Tracker(CFG)
    tracker.start(0.0, (1.0, 0.0), None)
    before = tracker.kf.s.copy()
    tracker.track(5.0, Pose2D())
    assert np.array_equal(tracker.kf.s, before)
