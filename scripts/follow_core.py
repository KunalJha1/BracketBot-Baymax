"""Pure person-follow logic for BracketBot: lock-on, tracking, control, and safety.

Deployed flat to the robot's /tmp next to robot_follow.py and imported by bare
name. Nothing here touches BBOS, TensorRT, or cameras, so every decision the
robot makes can be unit-tested and simulated on a laptop.

Conventions:
- Robot-local coordinates are (forward, left, up) in metres from the base origin.
- Odometry coordinates are (x, y) in metres; heading ``h`` is radians, CCW positive.
- Bearing is ``atan2(left, forward)``: positive means the person is to the
  robot's left, and positive omega turns left.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np


STATUS_PREFIX = "FOLLOW_STATUS "

SEARCHING = "SEARCHING"
FOLLOWING = "FOLLOWING"
BLOCKED = "BLOCKED"
LOST = "LOST"


@dataclass(frozen=True)
class FollowConfig:
    # Gap between the base origin and the person's torso (m).
    gap_default: float = 1.0
    gap_min: float = 0.6
    gap_max: float = 1.5
    band: float = 0.20  # acceptance band only; the controller never reads it
    # Controller
    deadband_range: float = 0.05
    k_r: float = 0.8
    v_max: float = 0.15  # default rises to 0.30 only after robot gate G4b passes
    k_theta: float = 1.5
    deadband_bearing: float = math.radians(3.0)
    omega_max: float = 0.8
    turn_in_place_bearing: float = math.radians(35.0)
    # Rate limits applied to what is actually sent
    accel_up: float = 0.4
    accel_down: float = 0.8
    alpha_max: float = 1.5
    # Supervisor
    min_range: float = 0.45
    heartbeat_timeout: float = 1.0
    perception_stale: float = 0.3
    points_stale: float = 0.3
    lost_after: float = 1.0
    lost_timeout: float = 10.0
    upright_deg: float = 25.0
    odom_mismatch_time: float = 0.5
    # Obstacle corridor (robot-local metres)
    robot_width: float = 0.3275
    corridor_margin: float = 0.10
    corridor_length: float = 0.60
    corridor_z_min: float = 0.05
    corridor_z_max: float = 1.70
    corridor_min_points: int = 30
    corridor_clear_time: float = 0.5
    person_exclusion_radius: float = 0.35
    # Robot's own body in the depth cloud: (f_min, f_max, l_min, l_max, z_min, z_max) boxes, from gate G0.
    self_mask: tuple[tuple[float, float, float, float, float, float], ...] = ()
    # Lock-on
    lock_window: float = 0.5
    lock_fraction: float = 0.8  # share of the window's frames the candidate must appear in
    lock_range_min: float = 0.5
    lock_range_max: float = 2.0
    lock_bearing_max: float = math.radians(30.0)
    lock_assoc_dist: float = 0.3
    # Tracking
    gate_sigma: float = 3.0
    hist_max_distance: float = 0.4
    hist_alpha: float = 0.05
    ambiguity_ratio: float = 0.10
    meas_sigma: float = 0.08
    accel_sigma: float = 1.0
    max_pos_sigma: float = 0.4
    lost_pos_sigma: float = 0.5


def clamp(value, low, high):
    return min(max(value, low), high)


def shrink(value, deadband):
    """Continuous deadband: zero inside +/-deadband, shifted linearly outside it."""
    if abs(value) <= deadband:
        return 0.0
    return math.copysign(abs(value) - deadband, value)


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    h: float = 0.0

    def integrate(self, v, omega, dt):
        mid = self.h + 0.5 * omega * dt
        self.x += v * math.cos(mid) * dt
        self.y += v * math.sin(mid) * dt
        self.h += omega * dt

    def to_odom(self, forward, left):
        c, s = math.cos(self.h), math.sin(self.h)
        return self.x + c * forward - s * left, self.y + s * forward + c * left

    def to_local(self, x, y):
        c, s = math.cos(self.h), math.sin(self.h)
        dx, dy = x - self.x, y - self.y
        return c * dx + s * dy, -s * dx + c * dy


@dataclass(frozen=True, eq=False)
class PersonObservation:
    forward: float
    left: float
    score: float = 1.0
    hist: np.ndarray | None = None  # optional appearance cue; depth-only perception leaves it None

    @property
    def range(self):
        return math.hypot(self.forward, self.left)

    @property
    def bearing(self):
        return math.atan2(self.left, self.forward)


@dataclass(frozen=True, eq=False)
class Perception:
    """One processed camera frame in robot-local coordinates."""

    t: float
    people: tuple[PersonObservation, ...]
    points: np.ndarray  # (N, 3) forward, left, up


@dataclass(frozen=True)
class Track:
    forward: float
    left: float
    range: float
    bearing: float
    v_radial: float  # person's own speed away from the robot (m/s)
    age: float  # seconds since the last accepted observation


def hist_distance(p, q):
    """Bhattacharyya distance of two L1-normalised histograms; 0 when either is missing.

    Dormant appearance hook (spec-sanctioned): depth-only perception never supplies
    a histogram, so ``p``/``q`` are always None here and this always returns 0.0.
    Do not mistake it for live behaviour.
    """
    if p is None or q is None:
        return 0.0
    overlap = float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))))
    return math.sqrt(max(0.0, 1.0 - overlap))


class ConstantVelocityKF:
    """(x, y, vx, vy) in the odometry frame, observed as (x, y)."""

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def __init__(self, x, y, t, cfg, vel_sigma=0.5):
        self.cfg = cfg
        self.s = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([cfg.meas_sigma**2] * 2 + [vel_sigma**2] * 2)
        self.R = np.eye(2) * cfg.meas_sigma**2
        self.t = t

    def predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt], [dt, 0.0], [0.0, dt]])
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + G @ G.T * self.cfg.accel_sigma**2
        self.t = t
        self._cap_position_variance()

    def _cap_position_variance(self):
        # Keeps the association gate bounded during long coasts.
        limit = self.cfg.max_pos_sigma**2
        for i in (0, 1):
            if self.P[i, i] > limit:
                k = math.sqrt(limit / self.P[i, i])
                self.P[i, :] *= k
                self.P[:, i] *= k

    def _innovation(self, z):
        y = np.asarray(z, dtype=float) - self.H @ self.s
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis(self, z):
        y, S = self._innovation(z)
        return float(math.sqrt(y @ np.linalg.solve(S, y)))

    def update(self, z):
        y, S = self._innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def forget_velocity(self, vel_sigma=0.5):
        pos_var = self.cfg.lost_pos_sigma**2
        self.s[2:] = 0.0
        self.P = np.diag([self.P[0, 0] + pos_var, self.P[1, 1] + pos_var, vel_sigma**2, vel_sigma**2])
        self._cap_position_variance()


class LockOn:
    """Chooses the single person who stays in the start zone for ``lock_window`` seconds."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.candidates = []
        self.frames = []  # times of the frames seen while searching

    def update(self, t, people, positions):
        """Returns ``(observation, odom_xy)`` of the locked person, or None."""
        cfg = self.cfg
        self.frames.append(t)
        used = set()
        for obs, xy in zip(people, positions):
            if not (cfg.lock_range_min <= obs.range <= cfg.lock_range_max):
                continue
            if abs(obs.bearing) > cfg.lock_bearing_max:
                continue
            best, best_d = None, cfg.lock_assoc_dist
            for i, cand in enumerate(self.candidates):
                d = math.dist(xy, cand["xy"])
                if i not in used and d <= best_d:
                    best, best_d = i, d
            if best is None:
                self.candidates.append({"xy": xy, "first_t": t, "seen": [], "obs": obs})
                best = len(self.candidates) - 1
            cand = self.candidates[best]
            cand["xy"], cand["obs"] = xy, obs
            cand["seen"].append(t)
            used.add(best)

        horizon = t - cfg.lock_window
        self.frames = [f for f in self.frames if f >= horizon]
        self.candidates = [c for c in self.candidates if c["seen"][-1] >= horizon]
        qualified = []
        for cand in self.candidates:
            cand["seen"] = [s for s in cand["seen"] if s >= horizon]
            if t - cand["first_t"] >= cfg.lock_window and len(cand["seen"]) >= cfg.lock_fraction * len(self.frames):
                qualified.append(cand)
        if len(qualified) != 1:
            return None  # nobody yet, or two people in the zone: keep waiting
        return qualified[0]["obs"], qualified[0]["xy"]


class Tracker:
    """Follows the locked person frame to frame, refusing to guess between look-alikes."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.kf = None
        self.ref_hist = None
        self.last_update = None

    @property
    def locked(self):
        return self.kf is not None

    def start(self, t, xy, hist):
        self.kf = ConstantVelocityKF(xy[0], xy[1], t, self.cfg)
        self.ref_hist = None if hist is None else np.asarray(hist, dtype=float)
        self.last_update = t

    def mark_lost(self):
        self.kf.forget_velocity()

    def age(self, t):
        return t - self.last_update

    def update(self, t, people, positions):
        """Returns "updated", "coasted" (no match), or "ambiguous" (two close matches)."""
        cfg = self.cfg
        self.kf.predict(t)
        candidates = []
        for obs, xy in zip(people, positions):
            d = self.kf.mahalanobis(xy)
            h = hist_distance(self.ref_hist, obs.hist)
            if d > cfg.gate_sigma or h > cfg.hist_max_distance:
                continue
            score = 0.5 * d / cfg.gate_sigma + 0.5 * h / cfg.hist_max_distance
            candidates.append((score, obs, xy))
        if not candidates:
            return "coasted"
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] <= candidates[0][0] * (1 + cfg.ambiguity_ratio):
            return "ambiguous"
        _, obs, xy = candidates[0]
        self.kf.update(xy)
        self.last_update = t
        if obs.hist is not None:
            hist = np.asarray(obs.hist, dtype=float)
            if self.ref_hist is None:
                self.ref_hist = hist
            else:
                blended = (1 - cfg.hist_alpha) * self.ref_hist + cfg.hist_alpha * hist
                self.ref_hist = blended / blended.sum()
        return "updated"

    def track(self, t, pose):
        """Robot-relative view of the person at time ``t``; does not modify the filter."""
        dt = max(0.0, t - self.kf.t)
        x = self.kf.s[0] + self.kf.s[2] * dt
        y = self.kf.s[1] + self.kf.s[3] * dt
        forward, left = pose.to_local(x, y)
        rng = math.hypot(forward, left)
        if rng > 1e-6:
            v_radial = (self.kf.s[2] * (x - pose.x) + self.kf.s[3] * (y - pose.y)) / rng
        else:
            v_radial = 0.0
        return Track(forward, left, rng, math.atan2(left, forward), float(v_radial), self.age(t))


def follow_command(track, gap, cfg):
    """(v, omega) that holds ``gap`` to the tracked person. v is never negative."""
    v = max(track.v_radial, 0.0) + cfg.k_r * shrink(track.range - gap, cfg.deadband_range)
    v *= math.cos(track.bearing)
    if abs(track.bearing) > cfg.turn_in_place_bearing:
        v = 0.0  # face the person before driving
    omega = cfg.k_theta * shrink(track.bearing, cfg.deadband_bearing)
    return clamp(v, 0.0, cfg.v_max), clamp(omega, -cfg.omega_max, cfg.omega_max)


class RateLimiter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.v = 0.0
        self.omega = 0.0

    def step(self, v, omega, dt):
        cfg = self.cfg
        self.v += clamp(v - self.v, -cfg.accel_down * dt, cfg.accel_up * dt)
        self.omega += clamp(omega - self.omega, -cfg.alpha_max * dt, cfg.alpha_max * dt)
        return self.v, self.omega


def corridor_count(points, person, cfg):
    """Points in the drive corridor that are not floor, the robot itself, or the person."""
    if points is None or len(points) == 0:
        return 0
    f, l, z = points[:, 0], points[:, 1], points[:, 2]
    half = cfg.robot_width / 2 + cfg.corridor_margin
    keep = (
        (f > 0) & (f <= cfg.corridor_length) & (np.abs(l) <= half)
        & (z >= cfg.corridor_z_min) & (z <= cfg.corridor_z_max)
    )
    if person is not None:
        keep &= np.hypot(f - person[0], l - person[1]) > cfg.person_exclusion_radius
    for f0, f1, l0, l1, z0, z1 in cfg.self_mask:
        keep &= ~((f >= f0) & (f <= f1) & (l >= l0) & (l <= l1) & (z >= z0) & (z <= z1))
    return int(np.count_nonzero(keep))


class CorridorGuard:
    """Blocks at once; clears only after ``corridor_clear_time`` below the threshold."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.blocked = False
        self._clear_since = None

    def update(self, t, count):
        if count >= self.cfg.corridor_min_points:
            self.blocked = True
            self._clear_since = None
        elif self.blocked:
            if self._clear_since is None:
                self._clear_since = t
            if t - self._clear_since >= self.cfg.corridor_clear_time:
                self.blocked = False
                self._clear_since = None
        return self.blocked


class OdometryCheck:
    """Trips when wheel feedback keeps moving opposite to what was sent (a sign bug)."""

    def __init__(self, cfg, *, v_sent_min=0.05, v_measured_min=0.03,
                 omega_sent_min=0.2, omega_measured_min=0.1):
        self.cfg = cfg
        self._since = None
        self.v_sent_min, self.v_measured_min = v_sent_min, v_measured_min
        self.omega_sent_min, self.omega_measured_min = omega_sent_min, omega_measured_min

    def update(self, t, v_sent, omega_sent, v_meas, omega_meas):
        wrong = (
            (abs(v_sent) >= self.v_sent_min and abs(v_meas) >= self.v_measured_min and v_meas * v_sent < 0)
            or (abs(omega_sent) >= self.omega_sent_min and abs(omega_meas) >= self.omega_measured_min and omega_meas * omega_sent < 0)
        )
        if not wrong:
            self._since = None
            return False
        if self._since is None:
            self._since = t
        return t - self._since >= self.cfg.odom_mismatch_time


@dataclass(frozen=True)
class Verdict:
    v: float
    omega: float
    rule: str
    exit: bool = False


def supervise(cfg, *, v, omega, stop_requested, heartbeat_age, roll_deg, pitch_deg,
              odom_mismatch, tracking, track_age, points_age, blocked, range_m):
    """Final say over every command. Exit rules first, then restrictions (all applied)."""
    if stop_requested:
        return Verdict(0.0, 0.0, "stop", True)
    if heartbeat_age > cfg.heartbeat_timeout:
        return Verdict(0.0, 0.0, "heartbeat", True)
    if abs(roll_deg) >= cfg.upright_deg or abs(pitch_deg) >= cfg.upright_deg:
        return Verdict(0.0, 0.0, "not-upright", True)
    if odom_mismatch:
        return Verdict(0.0, 0.0, "odometry-mismatch", True)

    fired = []
    if not tracking:
        v = omega = 0.0
        fired.append("no-track")
    if points_age > cfg.points_stale:
        v = 0.0  # obstacle state unknown: turning in place is still allowed
        fired.append("points-stale")
    if track_age is not None and track_age > cfg.perception_stale:
        v = omega = 0.0
        fired.append("track-stale")
    if blocked:
        v = 0.0
        fired.append("blocked")
    if range_m is not None and range_m < cfg.min_range:
        v = 0.0
        fired.append("min-range")
    return Verdict(
        clamp(v, 0.0, cfg.v_max),
        clamp(omega, -cfg.omega_max, cfg.omega_max),
        fired[0] if fired else "ok",
    )


def start_refusal(cfg, *, roll_deg, pitch_deg, voltage, low_battery_v, drive_writers, points_fresh):
    """Why the runner must not start, or None. ``voltage``/``low_battery_v`` may be None (unknown)."""
    if abs(roll_deg) >= cfg.upright_deg or abs(pitch_deg) >= cfg.upright_deg:
        return "robot is not upright"
    if not points_fresh:
        return "camera.points is not publishing; start the depth daemon"
    if drive_writers:
        return "another app is driving: " + "; ".join(drive_writers)
    if voltage is not None and low_battery_v is not None and voltage < low_battery_v:
        return f"battery low ({voltage:.1f} V < {low_battery_v:.1f} V)"
    return None


@dataclass(frozen=True, eq=False)
class TickInputs:
    t: float  # loop clock (s)
    heartbeat_age: float
    stop_requested: bool
    roll_deg: float
    pitch_deg: float
    measured_v: float  # from wheel feedback (m/s)
    measured_omega: float  # rad/s
    perception: Perception | None = None  # a new frame since the last tick, if any


@dataclass(frozen=True)
class TickOutput:
    t: float
    state: str
    v: float  # what to send
    omega: float
    v_cmd: float  # what the controller asked for
    omega_cmd: float
    rule: str
    exit: bool
    gap: float
    range: float | None
    bearing: float | None
    blocked: bool
    corridor_points: int
    track_age: float | None

    @property
    def error(self):
        return None if self.range is None else self.range - self.gap


class FollowLoop:
    """Composes lock-on, tracking, control, and supervision. One call per control tick."""

    def __init__(self, cfg=None, gap=None):
        self.cfg = cfg or FollowConfig()
        self.gap = self.cfg.gap_default
        if gap is not None:
            self.set_gap(gap)
        self.pose = Pose2D()
        self.lock_on = LockOn(self.cfg)
        self.tracker = Tracker(self.cfg)
        self.corridor = CorridorGuard(self.cfg)
        self.limiter = RateLimiter(self.cfg)
        self.odom_check = OdometryCheck(self.cfg)
        self.state = SEARCHING
        self.lost_since = None
        self.last_t = None
        self.last_points_t = None
        self.corridor_points = 0

    def set_gap(self, gap):
        self.gap = clamp(float(gap), self.cfg.gap_min, self.cfg.gap_max)
        return self.gap

    def tick(self, inp):
        cfg = self.cfg
        dt = 0.0 if self.last_t is None else max(0.0, inp.t - self.last_t)
        self.last_t = inp.t
        self.pose.integrate(inp.measured_v, inp.measured_omega, dt)
        mismatch = self.odom_check.update(
            inp.t, self.limiter.v, self.limiter.omega, inp.measured_v, inp.measured_omega
        )
        if inp.perception is not None:
            self._perceive(inp.perception)
        self._advance_state(inp.t)

        tracking = self.state in (FOLLOWING, BLOCKED)
        track = self.tracker.track(inp.t, self.pose) if tracking else None
        v_cmd, omega_cmd = follow_command(track, self.gap, cfg) if track else (0.0, 0.0)
        points_age = math.inf if self.last_points_t is None else inp.t - self.last_points_t
        verdict = supervise(
            cfg, v=v_cmd, omega=omega_cmd,
            stop_requested=inp.stop_requested, heartbeat_age=inp.heartbeat_age,
            roll_deg=inp.roll_deg, pitch_deg=inp.pitch_deg, odom_mismatch=mismatch,
            tracking=tracking, track_age=track.age if track else None,
            points_age=points_age, blocked=self.corridor.blocked,
            range_m=track.range if track else None,
        )
        if verdict.exit:
            self.limiter.reset()
            v, omega = 0.0, 0.0
        else:
            v, omega = self.limiter.step(verdict.v, verdict.omega, dt)
        return TickOutput(
            t=inp.t, state=self.state, v=v, omega=omega, v_cmd=v_cmd, omega_cmd=omega_cmd,
            rule=verdict.rule, exit=verdict.exit, gap=self.gap,
            range=track.range if track else None, bearing=track.bearing if track else None,
            blocked=self.corridor.blocked, corridor_points=self.corridor_points,
            track_age=track.age if track else None,
        )

    def _perceive(self, frame):
        positions = [self.pose.to_odom(o.forward, o.left) for o in frame.people]
        if self.state == SEARCHING:
            locked = self.lock_on.update(frame.t, frame.people, positions)
            if locked is not None:
                obs, xy = locked
                self.tracker.start(frame.t, xy, obs.hist)
                self.lock_on.reset()
                self.state = FOLLOWING
        elif self.tracker.update(frame.t, frame.people, positions) == "updated" and self.state == LOST:
            self.state = FOLLOWING
            self.lost_since = None
        person = None
        if self.tracker.locked:
            track = self.tracker.track(frame.t, self.pose)
            person = (track.forward, track.left)
        self.corridor_points = corridor_count(frame.points, person, self.cfg)
        self.corridor.update(frame.t, self.corridor_points)
        self.last_points_t = frame.t

    def _advance_state(self, t):
        if self.state in (FOLLOWING, BLOCKED):
            if self.tracker.age(t) > self.cfg.lost_after:
                self.state = LOST
                self.lost_since = t
                self.tracker.mark_lost()
            else:
                self.state = BLOCKED if self.corridor.blocked else FOLLOWING
        elif self.state == LOST and t - self.lost_since > self.cfg.lost_timeout:
            self.state = SEARCHING
            self.lost_since = None
            self.tracker.reset()
            self.lock_on.reset()


@dataclass(frozen=True)
class Command:
    kind: str  # "heartbeat" | "gap" | "stop"
    gap: float | None = None


def parse_command(line, cfg):
    """One dashboard stdin line -> Command, or None if malformed. Gaps are clamped."""
    try:
        message = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(message, dict):
        return None
    kind = message.get("type")
    if kind in ("heartbeat", "stop"):
        return Command(kind)
    if kind == "gap":
        value = message.get("m")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            finite = math.isfinite(value)
        except OverflowError:  # a bare JSON integer too large to convert to float
            return None
        if not finite:
            return None
        return Command("gap", clamp(float(value), cfg.gap_min, cfg.gap_max))
    return None


def status_line(out):
    def rounded(value, digits=3):
        return None if value is None else round(value, digits)

    payload = {
        "state": out.state,
        "range": rounded(out.range),
        "gap": round(out.gap, 2),
        "error": rounded(out.error),
        "bearing_deg": rounded(None if out.bearing is None else math.degrees(out.bearing), 1),
        "v": round(out.v, 3),
        "w": round(out.omega, 3),
        "blocked": out.blocked,
        "age_ms": None if out.track_age is None else round(out.track_age * 1000),
        "rule": out.rule,
    }
    return STATUS_PREFIX + json.dumps(payload, separators=(",", ":"))


STATE_LED = {
    SEARCHING: ((70, 125, 255), "pulse"),
    FOLLOWING: ((70, 220, 120), "solid"),
    BLOCKED: ((255, 160, 0), "solid"),
    LOST: ((255, 160, 0), "blink"),
}


def led_color(state, elapsed):
    """RGB for the state's LED pattern ``elapsed`` seconds into it (same shapes as robot_effect.py)."""
    rgb, pattern = STATE_LED[state]
    if pattern == "solid":
        scale = 1.0
    elif pattern == "blink":
        scale = 1.0 if int(elapsed * 3) % 2 == 0 else 0.08
    else:
        scale = 0.2 + 0.8 * (0.5 - 0.5 * math.cos(2 * math.pi * elapsed / 1.6))
    return tuple(round(c * scale) for c in rgb)


def timestamp_to_seconds(value):
    """BBOS timestamps have appeared in s, ms, us, and ns; normalise by magnitude."""
    value = float(value)
    if value > 1e17:
        return value / 1e9
    if value > 1e14:
        return value / 1e6
    if value > 1e11:
        return value / 1e3
    return value


def wheel_twist(turns_per_s, wheel_diam, robot_width, wheel_signs=(1.0, 1.0)):
    """drive.state.vel (turns/s, [left, right]) -> (v m/s, omega rad/s)."""
    circumference = math.pi * wheel_diam
    v_left = wheel_signs[0] * float(turns_per_s[0]) * circumference
    v_right = wheel_signs[1] * float(turns_per_s[1]) * circumference
    return (v_left + v_right) / 2.0, (v_right - v_left) / robot_width
