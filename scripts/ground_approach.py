"""One-shot, depth-grounded approach. Pure control; no BBOS or motor access.

The vision alert is latched for navigation safety. Motion instead requires a
fresh, currently observed low pose with the same session and track ID. The
standoff is measured outside a conservative body envelope, not to the torso.
"""

from dataclasses import dataclass, replace
import math

from follow_core import (
    FollowConfig, CorridorGuard, OdometryCheck, RateLimiter, TickOutput,
    clamp, corridor_count, supervise,
)

GROUND_LINE = "hello specimen, are you in trouble"
TARGET_MAX_AGE = 0.65
STANDOFF = 1.0


def approach_config(v_max=0.05):
    return replace(
        FollowConfig(), v_max=min(v_max, 0.05), omega_max=0.20,
        accel_up=0.04, accel_down=0.2, alpha_max=0.30,
        perception_stale=TARGET_MAX_AGE, corridor_z_min=0.03,
        corridor_length=0.85, corridor_margin=0.20, corridor_min_points=10,
        # Creep speed: opposite travel never accumulates, so trip on time alone.
        odom_mismatch_travel=0.0, odom_mismatch_turn=0.0,
    )


@dataclass(frozen=True)
class GroundTarget:
    session: str
    track_id: int
    captured: float
    forward: float
    left: float
    radius: float

    @property
    def distance(self):
        return math.hypot(self.forward, self.left)

    @property
    def clearance(self):
        return self.distance - self.radius

    @property
    def bearing(self):
        return math.atan2(self.left, self.forward)


def target_from_payload(payload, wall_now, left_sign=-1.0):
    """Reject stale/latching-only/ambiguous observations and malformed telemetry."""
    try:
        if (payload["schema_version"] != 1 or payload["depth_aligned"] is not True
                or payload["status"] != "alert"):
            return None
        captured = payload["camera_timestamp_ns"] / 1e9
        published = payload["published_at"]
        if not all(math.isfinite(v) and 0 <= wall_now - v <= TARGET_MAX_AGE
                   for v in (captured, published)):
            return None
        session = payload["session_id"]
        if not isinstance(session, str) or not session:
            return None
        # Even an unobserved second alert makes target selection ambiguous.
        if len(payload["alerts"]) != 1:
            return None
        track_id = payload["alerts"][0]["track_id"]
        if type(track_id) is not int:
            return None
        candidates = [o for o in payload["observations"] if o["track_id"] == track_id]
        if len(candidates) != 1:
            return None
        obs = candidates[0]
        if obs["state"] != "possible_person_on_ground" or obs["latch_status"] != "alert":
            return None
        right, forward, up = obs["base_position"]
        radius = obs["body_radius_m"]
        confidence = obs["confidence"]
        if not all(type(v) in (int, float) and math.isfinite(v)
                   for v in (right, forward, up, radius, confidence)):
            return None
        if not (0.1 < forward <= 6 and 0.25 <= radius <= 2.5 and 0.62 <= confidence <= 1):
            return None
        return GroundTarget(session, track_id, captured, forward, right * left_sign, radius)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


class PID:
    """Bounded integral and filtered derivative; reset whenever motion is inhibited."""

    def __init__(self, kp, ki, kd, low, high):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.low, self.high = low, high
        self.reset()

    def reset(self):
        self.integral = self.derivative = 0.0
        self.previous = None

    def step(self, error, dt):
        if not 0 < dt <= 0.2:
            self.reset()
            dt = 0.0
        raw_d = 0.0 if self.previous is None or not dt else (error - self.previous) / dt
        self.derivative += (dt / (0.25 + dt)) * (raw_d - self.derivative)
        integral = clamp(self.integral + error * dt, -0.5, 0.5)
        output = self.kp * error + self.ki * integral + self.kd * self.derivative
        # Conditional integration prevents windup at either actuator limit.
        if self.low <= output <= self.high or (output > self.high and error < 0) or (output < self.low and error > 0):
            self.integral = integral
        self.previous = error
        return clamp(self.kp * error + self.ki * self.integral + self.kd * self.derivative,
                     self.low, self.high)


class GroundApproachLoop:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gap = STANDOFF
        self.range_pid = PID(0.12, 0.01, 0.025, 0, cfg.v_max)
        self.turn_pid = PID(0.65, 0.015, 0.05, -cfg.omega_max, cfg.omega_max)
        self.limiter = RateLimiter(cfg)
        self.corridor = CorridorGuard(cfg)
        self.odom_check = OdometryCheck(cfg, v_sent_min=0.01, v_measured_min=0.01,
                                        omega_sent_min=0.05, omega_measured_min=0.04)
        self.last_t = self.started = self.last_points_t = None
        self.identity = self.last_target = None
        self.radius = 0.0
        self.settled_since = None
        self.arrived = False
        self.fault = None
        self.corridor_points = 0
        self.target = None

    def set_gap(self, _gap):
        return self.gap  # fixed standoff; the follow slider cannot shorten it

    def observe(self, target):
        self.target = target
        if target is None:
            return
        identity = (target.session, target.track_id)
        if self.identity is None:
            self.identity = identity
        elif identity != self.identity:
            self.fault = "target-changed"
        if self.last_target and target.captured > self.last_target.captured:
            if math.hypot(target.forward - self.last_target.forward,
                          target.left - self.last_target.left) > 0.4:
                self.fault = "target-jumped"
        elif self.last_target and target.captured < self.last_target.captured:
            self.fault = "camera-clock-reset"
        self.last_target = target
        # A disappearing limb must never reduce the standoff during this attempt.
        self.radius = max(self.radius, target.radius)

    def tick(self, inp, wall_now):
        cfg, target = self.cfg, self.target
        dt = 0.0 if self.last_t is None else inp.t - self.last_t
        if self.started is None:
            self.started = inp.t
        self.last_t = inp.t
        mismatch = self.odom_check.update(inp.t, self.limiter.v, self.limiter.omega,
                                          inp.measured_v, inp.measured_omega)
        if inp.perception is not None:
            self.last_points_t = inp.perception.t
            # Never exclude the person from collision checking.
            self.corridor_points = corridor_count(inp.perception.points, None, cfg)
            self.corridor.update(inp.perception.t, self.corridor_points)
        points_age = math.inf if self.last_points_t is None else inp.t - self.last_points_t
        age = None if target is None else wall_now - target.captured
        valid = target is not None and 0 <= age <= TARGET_MAX_AGE
        clearance = None if target is None else target.distance - self.radius
        bearing = None if target is None else target.bearing
        v_cmd = omega_cmd = 0.0
        rule = "ok"
        if self.fault:
            rule = self.fault
        elif inp.t - self.started > 120:
            rule = "approach-timeout"
        elif not valid:
            rule = "target-unavailable"
        elif points_age < 0 or points_age > cfg.points_stale:
            rule = "points-stale"
        elif self.corridor.blocked:
            rule = "blocked"
        elif not self.arrived:
            error = clearance - STANDOFF
            if error <= 0.05:
                # Brake first; speak only after wheels have actually settled.
                rule = "settling"
                if abs(inp.measured_v) < 0.015 and abs(inp.measured_omega) < 0.04:
                    if self.settled_since is None:
                        self.settled_since = inp.t
                    if inp.t - self.settled_since >= 0.6:
                        self.arrived = True
                else:
                    self.settled_since = None
            else:
                self.settled_since = None
                omega_cmd = self.turn_pid.step(bearing if abs(bearing) > math.radians(3) else 0, dt)
                if abs(bearing) <= math.radians(15):
                    v_cmd = self.range_pid.step(error - 0.05, dt) * math.cos(bearing)
                else:
                    self.range_pid.reset()
        verdict = supervise(
            cfg, v=v_cmd, omega=omega_cmd, stop_requested=inp.stop_requested,
            heartbeat_age=inp.heartbeat_age, roll_deg=inp.roll_deg, pitch_deg=inp.pitch_deg,
            odom_mismatch=mismatch, tracking=valid, track_age=age,
            points_age=points_age, blocked=self.corridor.blocked, range_m=clearance,
        )
        exiting = verdict.exit or self.fault is not None or rule == "approach-timeout"
        if verdict.exit:
            rule = verdict.rule
        if rule != "ok" or exiting or self.arrived:
            # Safety stops bypass the acceleration ramp, including angular motion.
            self.limiter.reset()
            self.range_pid.reset()
            self.turn_pid.reset()
            if rule not in ("settling", "ok"):
                self.settled_since = None
            v = omega = 0.0
        else:
            v, omega = self.limiter.step(verdict.v, verdict.omega, max(0, min(dt, 0.05)))
        state = "ARRIVED" if self.arrived else "BLOCKED" if rule == "blocked" else "APPROACHING" if rule == "ok" else "WAITING"
        return TickOutput(inp.t, state, v, omega, v_cmd, omega_cmd, rule, exiting,
                          STANDOFF, clearance, bearing, self.corridor.blocked,
                          self.corridor_points, age)
