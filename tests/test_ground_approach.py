from dataclasses import replace
import math

import numpy as np
import pytest

from follow_core import Perception, TickInputs
from ground_approach import (
    GroundApproachLoop, GroundTarget, PID, approach_config, target_from_payload,
)


def payload(now=100.0):
    return {
        "schema_version": 1, "session_id": "vision-a", "depth_aligned": True,
        "status": "alert", "camera_timestamp_ns": int(now * 1e9), "published_at": now,
        "alerts": [{"track_id": 7}], "observations": [{
            "track_id": 7, "latch_status": "alert", "state": "possible_person_on_ground",
            "base_position": [0.0, 3.0, 0.2], "body_radius_m": 0.8, "confidence": 0.9,
        }],
    }


def target(t=0, forward=3, left=0, radius=0.8):
    return GroundTarget("vision-a", 7, 100 + t, forward, left, radius)


def tick(loop, t, observation="default", points=True, **kwargs):
    if observation == "default":
        observation = target(t)
    loop.observe(observation)
    frame = None if points is False else Perception(t, (), np.empty((0, 3)) if points is True else points)
    values = dict(t=t, heartbeat_age=0, stop_requested=False, roll_deg=0, pitch_deg=0,
                  measured_v=0, measured_omega=0, perception=frame)
    values.update(kwargs)
    return loop.tick(TickInputs(**values), 100 + t)


def test_uses_aligned_current_pose_with_calibrated_bearing():
    data = payload()
    data["observations"][0]["base_position"][0] = 0.5
    result = target_from_payload(data, 100.1)
    assert result.left == -0.5
    assert result.clearance == pytest.approx(math.hypot(3, 0.5) - 0.8)
    assert target_from_payload(data, 100.1, 1).left == 0.5


@pytest.mark.parametrize("field,value", [
    ("depth_aligned", False), ("status", "checking"), ("schema_version", 2),
    ("session_id", ""), ("camera_timestamp_ns", 98e9), ("camera_timestamp_ns", 102e9),
    ("camera_timestamp_ns", float("nan")), ("published_at", 98), ("observations", []),
    ("alerts", [{"track_id": 7}, {"track_id": 8}]),
])
def test_invalid_or_stale_payload_cannot_move(field, value):
    data = payload()
    data[field] = value
    assert target_from_payload(data, 100.1) is None


@pytest.mark.parametrize("field,value", [
    ("state", "unknown"), ("state", "clear"), ("latch_status", "checking"),
    ("body_radius_m", None), ("body_radius_m", float("nan")),
    ("base_position", [0, float("inf"), 0.2]), ("base_position", [0, -1, 0.2]),
    ("track_id", 8), ("confidence", 0.2),
])
def test_latched_alert_without_current_evidence_cannot_move(field, value):
    data = payload()
    data["observations"][0][field] = value
    assert target_from_payload(data, 100.1) is None


@pytest.mark.parametrize("data", [None, [], 7, {}, {"schema_version": 1}])
def test_malformed_payload_fails_closed(data):
    assert target_from_payload(data, 100) is None


def moving_loop(left=0):
    loop = GroundApproachLoop(approach_config())
    for i in range(100):
        out = tick(loop, i * 0.02, target(i * 0.02, left=left))
    return loop, out


def test_speed_acceleration_and_turn_are_bounded():
    loop = GroundApproachLoop(approach_config(0.3))
    previous = 0
    for i in range(100):
        out = tick(loop, i * 0.02, target(i * 0.02, left=0.4))
        assert 0 <= out.v <= 0.05
        assert out.v - previous <= 0.04 * 0.02 + 1e-9
        assert 0 <= out.omega <= 0.2
        previous = out.v
    assert out.v > 0


def test_aligns_before_forward_motion():
    loop = GroundApproachLoop(approach_config())
    for i in range(100):
        out = tick(loop, i * 0.02, target(i * 0.02, left=-2))
    assert out.v == 0
    assert out.omega < 0


@pytest.mark.parametrize("failure", ["missing", "stale", "points", "obstacle", "stop", "heartbeat", "tilt"])
def test_hazards_zero_both_commands_immediately(failure):
    loop, before = moving_loop(0.4)
    assert before.v > 0 and before.omega > 0
    kwargs = {}
    if failure == "missing":
        kwargs["observation"] = None
    elif failure == "stale":
        kwargs["observation"] = target(0, left=0.4)
    elif failure == "points":
        kwargs["points"] = False
    elif failure == "obstacle":
        kwargs["points"] = np.tile([0.5, 0, 0.08], (30, 1))
    elif failure == "stop":
        kwargs["stop_requested"] = True
    elif failure == "heartbeat":
        kwargs["heartbeat_age"] = 2
    else:
        kwargs["pitch_deg"] = 30
    kwargs.setdefault("observation", target(2.5, left=0.4))
    out = tick(loop, 2.5, **kwargs)
    assert out.v == out.omega == 0
    assert not loop.arrived


@pytest.mark.parametrize("change", ["identity", "session", "position"])
def test_target_switch_or_jump_aborts_attempt(change):
    loop, _ = moving_loop()
    new = target(2)
    new = replace(new, **{"identity": {"track_id": 9}, "session": {"session": "new"},
                         "position": {"forward": 4}}[change])
    out = tick(loop, 2, new)
    assert out.exit and out.v == out.omega == 0


def test_body_envelope_never_shrinks_and_gap_cannot_be_shortened():
    loop = GroundApproachLoop(approach_config())
    tick(loop, 0, target(0, forward=2, radius=1.1))
    out = tick(loop, 0.02, target(0.02, forward=2, radius=0.3))
    assert out.range == pytest.approx(0.9)
    assert out.v == 0
    assert loop.set_gap(0.1) == 1


def test_arrival_waits_for_stationary_wheels_and_is_one_shot():
    loop = GroundApproachLoop(approach_config())
    for i in range(70):
        t = i * 0.02
        tick(loop, t, target(t, forward=1.8), measured_v=0.025)
        assert not loop.arrived
    for i in range(70, 120):
        t = i * 0.02
        out = tick(loop, t, target(t, forward=1.8))
    assert loop.arrived and out.v == out.omega == 0
    for i in range(120, 140):
        t = i * 0.02
        out = tick(loop, t, target(t, forward=1.8 + (i - 120) * 0.02))
        assert out.v == out.omega == 0


def test_stale_pose_cannot_satisfy_stationary_dwell():
    loop = GroundApproachLoop(approach_config())
    tick(loop, 0, target(0, forward=1.8))
    tick(loop, 0.5, None)
    tick(loop, 0.7, target(0.7, forward=1.8))
    assert not loop.arrived


def test_straight_approach_simulation_reaches_body_standoff():
    loop = GroundApproachLoop(approach_config())
    distance, speed = 3.3, 0
    for i in range(6000):
        t = i * 0.02
        out = tick(loop, t, target(t, forward=distance), measured_v=speed)
        assert not out.exit
        speed = out.v
        distance -= speed * 0.02
        assert distance - 0.8 >= 1.0
        if loop.arrived:
            break
    assert loop.arrived
    assert 1.0 <= distance - 0.8 <= 1.06


def test_pid_does_not_wind_up_and_resets_derivative():
    pid = PID(1, 1, 0.1, 0, 0.05)
    for _ in range(500):
        assert pid.step(10, 0.02) == 0.05
    assert pid.integral == 0
    pid.reset()
    assert pid.step(0, 0.02) == 0


@pytest.mark.parametrize("turning", [True, False])
def test_wrong_direction_feedback_is_detected_at_creep_speed(turning):
    loop = GroundApproachLoop(approach_config())
    for i in range(100):
        t = i * 0.02
        out = tick(loop, t, target(t, left=2 if turning else 0.3),
                   measured_v=0 if turning else -0.025,
                   measured_omega=-0.08 if turning else 0)
        if out.exit:
            break
    assert out.exit and out.rule == "odometry-mismatch"
    assert out.v == out.omega == 0
