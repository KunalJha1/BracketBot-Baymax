import dataclasses
import json
import math

import numpy as np
import pytest

from follow_core import (
    FollowConfig,
    PersonObservation,
    Pose2D,
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
