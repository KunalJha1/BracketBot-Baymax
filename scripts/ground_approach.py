"""One-shot, depth-grounded approach. Pure control; no BBOS or motor access.

The vision alert is latched for navigation safety. Motion instead requires a
fresh, currently observed low pose with the same session and track ID. The
standoff is measured outside a conservative body envelope, not to the torso.
"""

from collections import deque
from dataclasses import dataclass, replace
import math

from follow_core import (
    FollowConfig, FollowController, CorridorGuard, OdometryCheck, RateLimiter, TickOutput,
    Track, corridor_count, supervise,
)

GROUND_LINE = "hello specimen, are you in trouble"
# The vision app shares a saturated CPU: its frames arrive 0.3-1.5 s old, and a
# body on the floor drops out of detection for a frame now and then. Demanding a
# fresh pose every tick meant the target kept expiring, every expiry reset the
# 0.04 m/s^2 ramp, and the base never got moving. The person is not moving, so
# the loop pins them in the odometry frame when a frame arrives and tracks its
# own wheel motion in between. That also takes the vision delay out of the
# bearing loop. Obstacles still use depth no older than 0.3 s.
TARGET_MAX_AGE = 2.0       # oldest camera frame a target may come from
PUBLISH_MAX_AGE = 1.0      # the vision app must still be publishing
TARGET_HOLD = 1.0          # how long a missing observation may be bridged
TARGET_JUMP = 0.4          # a pinned target that moves this far is a different body
# Gap kept outside the body envelope (furthest joint + 0.25 m), so the wheels stay
# about 0.85 m from the nearest limb. 1.0 m left the robot too far away to check on
# anyone. It must stay above FollowConfig.min_range (0.45 m), which stops forward motion.
STANDOFF = 0.6


def approach_config(v_max=0.05):
    return replace(
        FollowConfig(), v_max=min(v_max, 0.05), omega_max=0.20,
        turn_in_place_bearing=math.radians(15.0),
        accel_up=0.04, accel_down=0.2, alpha_max=0.30,
        perception_stale=TARGET_MAX_AGE,
        # Depth arrives 0.21-0.24 s old on the loaded CPU, so the follow limit of 0.3 s
        # tripped on 14% of ticks and each trip resets the ramp. 0.5 s is 2.5 cm at creep.
        points_stale=0.5,
        # Measured on the robot: the levelled floor leaves 10-20 points between 3 and
        # 5 cm and none above. 6 cm still catches a forearm or shin lying in the way.
        corridor_z_min=0.06,
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
        if not all(math.isfinite(v) for v in (captured, published)):
            return None
        if not (0 <= wall_now - captured <= TARGET_MAX_AGE
                and 0 <= wall_now - published <= PUBLISH_MAX_AGE):
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


class GroundApproachLoop:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gap = STANDOFF
        # Share the tuned motion law and gains with normal follow. Only the
        # approach limits and destination differ; future PID tuning applies here too.
        self.controller = FollowController(cfg)
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
        # Wheel odometry since start: x forward, y left, heading, at each wall time.
        self.pose = (0.0, 0.0, 0.0)
        self.poses = deque(maxlen=400)
        self.anchor = None          # the person, pinned in the odometry frame
        self.anchor_captured = None
        self.last_seen_wall = None

    def set_gap(self, _gap):
        return self.gap  # fixed standoff; the follow slider cannot shorten it

    def _pose_at(self, wall_time):
        """Odometry pose when a camera frame was taken (the newest pose not after it)."""
        chosen = self.poses[0][1] if self.poses else self.pose
        for stamp, pose in self.poses:
            if stamp > wall_time:
                break
            chosen = pose
        return chosen

    def observe(self, target):
        self.target = target
        if target is None:
            return
        identity = (target.session, target.track_id)
        if self.identity is None:
            self.identity = identity
        elif identity != self.identity:
            self.fault = "target-changed"
        if self.anchor_captured is not None and target.captured < self.anchor_captured:
            self.fault = "camera-clock-reset"
        elif self.anchor_captured is None or target.captured > self.anchor_captured:
            x, y, heading = self._pose_at(target.captured)
            cosine, sine = math.cos(heading), math.sin(heading)
            anchor = (x + cosine * target.forward - sine * target.left,
                      y + sine * target.forward + cosine * target.left)
            # Compared where the person lies, not in the base frame: a slow vision
            # frame while the robot turns moves the base-frame position legitimately.
            if self.anchor is not None and math.dist(anchor, self.anchor) > TARGET_JUMP:
                self.fault = "target-jumped"
            self.anchor, self.anchor_captured = anchor, target.captured
        self.last_target = target
        # A disappearing limb must never reduce the standoff during this attempt.
        self.radius = max(self.radius, target.radius)

    def _pinned(self):
        """The pinned person in today's base frame: (forward, left)."""
        x, y, heading = self.pose
        dx, dy = self.anchor[0] - x, self.anchor[1] - y
        cosine, sine = math.cos(heading), math.sin(heading)
        return cosine * dx + sine * dy, -sine * dx + cosine * dy

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
        step = dt if 0 < dt <= 0.2 else 0.0
        x, y, heading = self.pose
        heading += inp.measured_omega * step
        self.pose = (x + inp.measured_v * math.cos(heading) * step,
                     y + inp.measured_v * math.sin(heading) * step, heading)
        self.poses.append((wall_now, self.pose))
        if target is not None:
            self.last_seen_wall = wall_now
        observed = target is not None
        age = None if self.anchor_captured is None else wall_now - self.anchor_captured
        valid = (
            self.anchor is not None and 0 <= age <= TARGET_MAX_AGE
            and wall_now - self.last_seen_wall <= TARGET_HOLD
        )
        if self.anchor is None:
            forward = left = clearance = bearing = None
        else:
            forward, left = self._pinned()
            clearance = math.hypot(forward, left) - self.radius
            bearing = math.atan2(left, forward)
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
            if error <= cfg.deadband_range:
                # Brake first; speak only after wheels have actually settled.
                rule = "settling"
                # Arrival is only ever declared on a pose the camera has just confirmed.
                if observed and abs(inp.measured_v) < 0.015 and abs(inp.measured_omega) < 0.04:
                    if self.settled_since is None:
                        self.settled_since = inp.t
                    if inp.t - self.settled_since >= 0.6:
                        self.arrived = True
                else:
                    self.settled_since = None
            else:
                self.settled_since = None
                control_dt = dt
                if not 0 < dt <= 0.2:
                    self.controller.reset()
                    control_dt = 0.0
                # Track uses distance to the body centre, so add the conservative
                # radius to the desired gap. No target-velocity feedforward: this
                # mode approaches a ground pose, it does not chase a walking person.
                track = Track(forward, left, math.hypot(forward, left), bearing,
                              v_radial=0.0, age=age)
                v_cmd, omega_cmd = self.controller.command(track, STANDOFF + self.radius, control_dt)
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
            self.controller.reset()
            if rule not in ("settling", "ok"):
                self.settled_since = None
            v = omega = 0.0
        else:
            v, omega = self.limiter.step(verdict.v, verdict.omega, max(0, min(dt, 0.05)))
        state = "ARRIVED" if self.arrived else "BLOCKED" if rule == "blocked" else "APPROACHING" if rule == "ok" else "WAITING"
        return TickOutput(inp.t, state, v, omega, v_cmd, omega_cmd, rule, exiting,
                          STANDOFF, clearance, bearing, self.corridor.blocked,
                          self.corridor_points, age)
