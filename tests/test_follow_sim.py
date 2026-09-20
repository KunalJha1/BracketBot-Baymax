"""Closed-loop kinematic simulation of FollowLoop.

SIMULATION EVIDENCE ONLY. The robot is a unicycle with a first-order velocity
lag; perception is 15 Hz with 100 ms latency and Gaussian range/bearing noise,
and observations carry no appearance cue (as with depth-only perception).
Passing here says the logic and tuning are coherent, not that the robot works.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np
import pytest

from follow_core import (
    BLOCKED, FOLLOWING, LOST, FollowConfig, FollowLoop, Perception, PersonObservation,
    Pose2D, TickInputs,
)

CFG = replace(FollowConfig(), v_max=0.30)  # the configuration targeted after gate G4b
DT = 0.02
FRAME_EVERY = 1 / 15
LATENCY = 0.10
LAG = 0.15  # balancing base velocity response time constant (s)


@dataclass
class Scenario:
    target: callable  # t -> (x, y) in the world
    duration: float
    bystander: callable | None = None
    visible: callable = lambda t: True
    obstacle: callable = lambda t: None  # t -> (x, y) of a 0.3 m box in the world, or None
    heartbeat_until: float = math.inf
    seed: int = 0
    appearance: bool = False
    frame_period: float = FRAME_EVERY
    latency: float = LATENCY


@dataclass
class Result:
    t: list = field(default_factory=list)
    true_range: list = field(default_factory=list)
    true_bearing: list = field(default_factory=list)
    robot_x: list = field(default_factory=list)
    out: list = field(default_factory=list)

    def samples(self, start, end=math.inf):
        return [
            (t, r, b, o) for t, r, b, o in zip(self.t, self.true_range, self.true_bearing, self.out)
            if start <= t < end
        ]


def box_points(center_local):
    xs, ys, zs = np.meshgrid(
        np.linspace(-0.15, 0.15, 6), np.linspace(-0.15, 0.15, 6), np.linspace(0.05, 0.3, 4)
    )
    local = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    local[:, 0] += center_local[0]
    local[:, 1] += center_local[1]
    return local


def run(scenario, cfg=CFG, gap=1.0):
    rng = np.random.default_rng(scenario.seed)
    loop = FollowLoop(cfg, gap)
    robot = Pose2D()
    v_actual = omega_actual = 0.0
    result = Result()
    next_frame = 0.0
    pending = []  # (deliver_at, Perception)
    steps = int(scenario.duration / DT)
    for i in range(steps):
        t = i * DT
        if t >= next_frame:
            next_frame += scenario.frame_period
            capture_t = t
            people = []
            actors = [(scenario.target, scenario.visible(t))]
            if scenario.bystander is not None:
                actors.append((scenario.bystander, True))
            for actor_id, (path, visible) in enumerate(actors):
                if not visible:
                    continue
                f, l = robot.to_local(*path(t))
                r = math.hypot(f, l) + rng.normal(0, 0.025)
                b = math.atan2(l, f) + rng.normal(0, math.radians(1.0))
                hist = np.eye(64)[actor_id] if scenario.appearance else None
                people.append(PersonObservation(r * math.cos(b), r * math.sin(b), hist=hist))
            points = np.empty((0, 3))
            box = scenario.obstacle(t)
            if box is not None:
                points = box_points(robot.to_local(*box))
            pending.append((capture_t + scenario.latency, Perception(capture_t, tuple(people), points)))
        frame = None
        if pending and pending[0][0] <= t + 1e-9:
            frame = pending.pop(0)[1]
        out = loop.tick(TickInputs(
            t=t,
            heartbeat_age=0.1 if t < scenario.heartbeat_until else t - scenario.heartbeat_until,
            stop_requested=False, roll_deg=0.0, pitch_deg=1.0,
            measured_v=v_actual + rng.normal(0, 0.005),
            measured_omega=omega_actual + rng.normal(0, 0.01),
            perception=frame,
        ))
        tx, ty = scenario.target(t)
        f, l = robot.to_local(tx, ty)
        result.t.append(t)
        result.true_range.append(math.hypot(f, l))
        result.true_bearing.append(math.atan2(l, f))
        result.robot_x.append(robot.x)
        result.out.append(out)
        if out.exit:
            break
        v_actual += (out.v - v_actual) * DT / LAG
        omega_actual += (out.omega - omega_actual) * DT / LAG
        robot.integrate(v_actual, omega_actual, DT)
    return result


def standing(x, y=0.0):
    return lambda t: (x, y)


def walking_away(speed, start=2.0, x0=1.0):
    return lambda t: (x0 + speed * max(0.0, t - start), 0.0)


def fraction_in_band(samples, gap=1.0, band=0.20):
    inside = sum(abs(r - gap) <= band for _, r, _, _ in samples)
    return inside / len(samples)


def test_standing_person_settles_near_the_gap():
    result = run(Scenario(standing(1.5), duration=12.0))
    final = result.samples(10.0)
    assert all(o.state == FOLLOWING for *_, o in final)
    assert abs(np.mean([r for _, r, _, _ in final]) - 1.0) <= 0.07


def test_slow_walk_stays_inside_the_band():
    result = run(Scenario(walking_away(0.20), duration=25.0))
    assert fraction_in_band(result.samples(6.0)) >= 0.95


def test_fast_walk_opens_the_gap_then_recovers_when_the_person_stops():
    def path(t):
        return (1.0 + 1.0 * min(max(0.0, t - 2.0), 3.0), 0.0)

    result = run(Scenario(path, duration=20.0))
    assert max(r for _, r, _, _ in result.samples(2.0, 6.0)) > 2.5
    assert fraction_in_band(result.samples(16.0)) == 1.0


def test_one_metre_position_jump_without_identity_does_not_redirect_the_robot():
    def path(t):
        return (1.0, 0.0) if t < 3.0 else (1.0, 1.0)

    result = run(Scenario(path, duration=10.0))
    after = result.samples(3.1, 10.0)
    assert result.out[-1].state == LOST
    assert all(o.v == o.omega == 0 for *_, o in result.samples(5, 10))


def test_person_approaching_never_makes_the_robot_reverse():
    def path(t):
        return (max(0.5, 1.4 - 0.3 * max(0.0, t - 2.0)), 0.0)

    result = run(Scenario(path, duration=8.0))
    assert all(o.v >= 0.0 for o in result.out)
    assert all(o.v == 0.0 for t, r, _, o in result.samples(5.0) if r < 1.0 - 0.1)


def test_occlusion_stops_goes_lost_and_recovers():
    result = run(Scenario(walking_away(0.15), duration=14.0, visible=lambda t: not 5.0 <= t < 7.0, appearance=True))
    during = result.samples(5.8, 7.0)
    assert any(o.state == LOST for *_, o in during)
    assert all(o.v == 0.0 for *_, o in result.samples(6.2, 7.0))
    assert result.out[-1].state == FOLLOWING


def test_bystander_crossing_does_not_steal_the_track():
    def bystander(t):
        return (1.3, 1.5 - 0.6 * t)  # crosses between robot and target around t = 2.5 s

    result = run(Scenario(walking_away(0.1, start=3.0), duration=10.0, bystander=bystander))
    final = result.out[-1]
    assert final.state == FOLLOWING
    assert final.range == pytest.approx(result.true_range[-1], abs=0.15)


def test_obstacle_blocks_forward_motion_before_contact():
    box_x = 1.25  # a 0.3 m box appears between the robot (~0.75 m) and the person (~1.8 m)
    result = run(Scenario(walking_away(0.2), duration=12.0,
                          obstacle=lambda t: (box_x, 0.0) if t >= 6.0 else None))
    blocked = [o for *_, o in result.samples(6.0) if o.state == BLOCKED]
    assert blocked
    first = blocked[0].t
    assert first - 6.0 <= LATENCY + FRAME_EVERY + 2 * DT
    assert all(o.v == 0.0 for *_, o in result.samples(first + 0.5))
    assert max(result.robot_x) < box_x - 0.15 - 0.3  # base origin stays >= 0.3 m from the box


def test_heartbeat_loss_exits():
    result = run(Scenario(walking_away(0.2), duration=10.0, heartbeat_until=5.0))
    assert result.out[-1].exit is True
    assert result.out[-1].rule == "heartbeat"
    assert result.t[-1] <= 5.0 + CFG.heartbeat_timeout + 2 * DT
