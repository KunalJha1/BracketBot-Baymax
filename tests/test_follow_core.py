import dataclasses
import json
import math

import numpy as np
import pytest

from follow_core import (
    BLOCKED,
    FOLLOWING,
    SEARCHING,
    STATUS_PREFIX,
    Command,
    CorridorGuard,
    FollowConfig,
    FollowLoop,
    LockOn,
    OdometryCheck,
    Perception,
    PersonObservation,
    Pose2D,
    RateLimiter,
    TickInputs,
    Track,
    Tracker,
    corridor_count,
    follow_command,
    hist_distance,
    led_color,
    parse_command,
    start_refusal,
    status_line,
    supervise,
    timestamp_to_seconds,
    wheel_twist,
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


def track(range_m, bearing_deg=0.0, v_radial=0.0, age=0.05):
    b = math.radians(bearing_deg)
    return Track(range_m * math.cos(b), range_m * math.sin(b), range_m, b, v_radial, age)


def safe_inputs(**overrides):
    values = dict(
        v=0.1, omega=0.1, stop_requested=False, heartbeat_age=0.1, roll_deg=0.0,
        pitch_deg=2.0, odom_mismatch=False, tracking=True, track_age=0.05,
        points_age=0.05, blocked=False, range_m=1.0,
    )
    values.update(overrides)
    return values


# --- controller -----------------------------------------------------------

def test_inside_deadband_the_robot_holds_still():
    assert follow_command(track(1.04), 1.0, FAST) == (0.0, 0.0)


def test_range_error_drives_forward_proportionally():
    v, omega = follow_command(track(1.25), 1.0, FAST)
    assert v == pytest.approx(0.8 * 0.20)
    assert omega == 0.0


def test_person_speed_is_fed_forward():
    v, _ = follow_command(track(1.0, v_radial=0.12), 1.0, FAST)
    assert v == pytest.approx(0.12)


def test_never_reverses_when_the_person_comes_closer():
    v, _ = follow_command(track(0.6, v_radial=-0.3), 1.0, FAST)
    assert v == 0.0


def test_turns_in_place_beyond_35_degrees():
    v, omega = follow_command(track(1.5, bearing_deg=40), 1.0, FAST)
    assert v == 0.0
    assert omega == FAST.omega_max  # positive: turn left toward the person


def test_speeds_are_clamped():
    v, omega = follow_command(track(3.0, bearing_deg=-60), 1.0, FAST)
    assert v == 0.0
    assert omega == -FAST.omega_max
    v, _ = follow_command(track(3.0), 1.0, CFG)
    assert v == CFG.v_max


def test_rate_limiter_uses_separate_accel_brake_and_turn_limits():
    limiter = RateLimiter(CFG)
    assert limiter.step(0.3, 1.0, 0.02) == pytest.approx((0.008, 0.03))
    limiter.v = 0.3
    assert limiter.step(0.0, 0.03, 0.02)[0] == pytest.approx(0.3 - 0.016)


# --- obstacle corridor ----------------------------------------------------

def test_corridor_ignores_floor_person_and_self_and_counts_obstacles():
    floor = np.column_stack([np.linspace(0.1, 0.6, 50), np.zeros(50), np.zeros(50)])
    box = np.column_stack([np.full(40, 0.5), np.linspace(-0.1, 0.1, 40), np.full(40, 0.2)])
    body = np.column_stack([np.full(40, 0.05), np.zeros(40), np.full(40, 1.0)])
    cfg = dataclasses.replace(CFG, self_mask=((0.0, 0.1, -0.2, 0.2, 0.3, 1.6),))
    points = np.vstack([floor, box, body])
    assert corridor_count(points, None, cfg) == 40
    assert corridor_count(points, (0.55, 0.0), cfg) == 0  # the box is where the person is
    assert corridor_count(np.empty((0, 3)), None, cfg) == 0


def test_corridor_blocks_at_once_and_clears_with_hysteresis():
    guard = CorridorGuard(CFG)
    assert guard.update(0.0, 30) is True
    assert guard.update(0.1, 0) is True
    assert guard.update(0.55, 0) is True
    assert guard.update(0.61, 0) is False


def test_odometry_check_needs_a_sustained_opposite_sign():
    check = OdometryCheck(CFG)
    assert check.update(0.0, 0.2, 0.0, -0.1, 0.0) is False
    assert check.update(0.3, 0.2, 0.0, -0.1, 0.0) is False
    assert check.update(0.6, 0.2, 0.0, -0.1, 0.0) is True
    assert check.update(0.7, 0.2, 0.0, 0.1, 0.0) is False


# --- supervisor -----------------------------------------------------------

@pytest.mark.parametrize("override, rule", [
    ({"stop_requested": True}, "stop"),
    ({"heartbeat_age": 1.2}, "heartbeat"),
    ({"roll_deg": 26.0}, "not-upright"),
    ({"pitch_deg": -25.0}, "not-upright"),
    ({"odom_mismatch": True}, "odometry-mismatch"),
])
def test_exit_rules(override, rule):
    verdict = supervise(CFG, **safe_inputs(**override))
    assert (verdict.v, verdict.omega, verdict.rule, verdict.exit) == (0.0, 0.0, rule, True)


def test_exit_rule_priority():
    verdict = supervise(CFG, **safe_inputs(heartbeat_age=5.0, roll_deg=40.0, blocked=True))
    assert verdict.rule == "heartbeat"


@pytest.mark.parametrize("override, rule, omega", [
    ({"tracking": False, "track_age": None, "range_m": None}, "no-track", 0.0),
    ({"points_age": 0.5}, "points-stale", 0.1),
    ({"track_age": 0.4}, "track-stale", 0.0),
    ({"blocked": True}, "blocked", 0.1),
    ({"range_m": 0.4}, "min-range", 0.1),
])
def test_restriction_rules(override, rule, omega):
    verdict = supervise(CFG, **safe_inputs(**override))
    assert (verdict.v, verdict.omega, verdict.rule, verdict.exit) == (0.0, omega, rule, False)


def test_supervisor_passes_safe_commands_and_never_reverses():
    assert supervise(CFG, **safe_inputs()).rule == "ok"
    assert supervise(CFG, **safe_inputs(v=0.1)).v == pytest.approx(0.1)
    assert supervise(CFG, **safe_inputs(v=-0.2)).v == 0.0
    verdict = supervise(CFG, **safe_inputs(v=0.9, omega=-3.0))
    assert (verdict.v, verdict.omega) == (CFG.v_max, -CFG.omega_max)


def test_start_refusal():
    ok = dict(roll_deg=0.0, pitch_deg=3.0, voltage=24.0, low_battery_v=21.0,
              drive_writers=[], points_fresh=True)
    assert start_refusal(CFG, **ok) is None
    assert start_refusal(CFG, **{**ok, "pitch_deg": 30.0}) == "robot is not upright"
    assert "depth daemon" in start_refusal(CFG, **{**ok, "points_fresh": False})
    assert "greeter" in start_refusal(CFG, **{**ok, "drive_writers": ["123 python greeter/main.py"]})
    assert start_refusal(CFG, **{**ok, "voltage": 20.0}) == "battery low (20.0 V < 21.0 V)"
    assert start_refusal(CFG, **{**ok, "voltage": None}) is None


# --- loop -----------------------------------------------------------------

def tick_inputs(t, **overrides):
    values = dict(t=t, heartbeat_age=0.1, stop_requested=False, roll_deg=0.0, pitch_deg=0.0,
                  measured_v=0.0, measured_omega=0.0, perception=None)
    values.update(overrides)
    return TickInputs(**values)


def test_loop_holds_still_while_searching():
    loop = FollowLoop(CFG)
    frame = Perception(0.0, (person(1.5, 0.3),), np.empty((0, 3)))
    out = loop.tick(tick_inputs(0.0, perception=frame))
    assert (out.state, out.v, out.omega, out.rule) == (SEARCHING, 0.0, 0.0, "no-track")


def test_loop_locks_on_then_follows():
    loop = FollowLoop(FAST)
    out = None
    for i in range(100):
        t = i * 0.02
        frame = None
        if i % 3 == 0:
            frame = Perception(t, (person(1.6, raised=t < 1.0),), np.empty((0, 3)))
        out = loop.tick(tick_inputs(t, perception=frame))
    assert out.state == FOLLOWING
    assert out.v > 0.0


def test_loop_exits_on_heartbeat_loss():
    out = FollowLoop(CFG).tick(tick_inputs(0.0, heartbeat_age=1.5))
    assert out.exit is True
    assert (out.v, out.omega, out.rule) == (0.0, 0.0, "heartbeat")


def test_gap_is_clamped():
    loop = FollowLoop(CFG)
    assert loop.set_gap(3.0) == 1.5
    assert loop.set_gap(0.1) == 0.6


# --- protocol and helpers -------------------------------------------------

def test_parse_command():
    assert parse_command('{"type":"heartbeat"}', CFG) == Command("heartbeat")
    assert parse_command('{"type":"stop"}', CFG) == Command("stop")
    assert parse_command('{"type":"gap","m":1.2}', CFG) == Command("gap", 1.2)
    assert parse_command('{"type":"gap","m":9}', CFG) == Command("gap", 1.5)
    for bad in ("", "nope", "[]", '{"type":"gap","m":true}', '{"type":"gap","m":NaN}',
                '{"type":"gap"}', '{"type":"drive","v":1}'):
        assert parse_command(bad, CFG) is None, bad


def test_status_line_is_prefixed_json():
    loop = FollowLoop(CFG)
    out = loop.tick(tick_inputs(0.0))
    line = status_line(out)
    assert line.startswith(STATUS_PREFIX)
    payload = json.loads(line[len(STATUS_PREFIX):])
    assert payload["state"] == SEARCHING
    assert payload["gap"] == 1.0
    assert payload["range"] is None
    assert set(payload) == {"state", "range", "gap", "error", "bearing_deg", "v", "w",
                            "blocked", "age_ms", "rule"}


def test_led_patterns():
    assert led_color(FOLLOWING, 0.0) == (70, 220, 120)
    assert led_color(BLOCKED, 5.0) == (255, 160, 0)
    assert led_color("LOST", 0.0) == (255, 160, 0)
    assert led_color("LOST", 0.4) == (20, 13, 0)
    assert led_color(SEARCHING, 0.0) == (14, 25, 51)


def test_timestamp_units_are_normalised():
    for value in (1_760_000_000.5, 1_760_000_000_500, 1_760_000_000_500_000, 1_760_000_000_500_000_000):
        assert timestamp_to_seconds(value) == pytest.approx(1_760_000_000.5)


def test_wheel_twist():
    v, omega = wheel_twist((1.0, 1.0), 0.165, 0.3275)
    assert (v, omega) == pytest.approx((math.pi * 0.165, 0.0))
    v, omega = wheel_twist((-1.0, 1.0), 0.165, 0.3275)
    assert v == pytest.approx(0.0)
    assert omega == pytest.approx(2 * math.pi * 0.165 / 0.3275)
    assert wheel_twist((1.0, 1.0), 0.165, 0.3275, (-1.0, -1.0))[0] < 0
