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
    lock_fraction: float = 0.8
    lock_range_min: float = 0.5
    lock_range_max: float = 2.5
    lock_bearing_max: float = math.radians(60.0)
    lock_assoc_dist: float = 0.3
    # Tracking
    gate_sigma: float = 3.0
    hist_max_distance: float = 0.4
    hist_alpha: float = 0.05
    ambiguity_ratio: float = 0.10
    meas_sigma: float = 0.08
    accel_sigma: float = 1.0
    max_pos_sigma: float = 1.0
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
    score: float
    hand_raised: bool
    hist: np.ndarray | None = None  # (64,) L1-normalised 4x4x4 HSV torso histogram

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
    """Bhattacharyya distance of two L1-normalised histograms; 0 when either is missing."""
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
    """Chooses the single person who keeps a hand raised for ``lock_window`` seconds."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.candidates = []

    def reset(self):
        self.candidates = []

    def update(self, t, people, positions):
        """Returns ``(observation, odom_xy)`` of the locked person, or None."""
        cfg = self.cfg
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
                self.candidates.append({"xy": xy, "first_t": t, "samples": [], "obs": obs})
                best = len(self.candidates) - 1
            cand = self.candidates[best]
            cand["xy"], cand["obs"] = xy, obs
            cand["samples"].append((t, obs.hand_raised))
            used.add(best)

        horizon = t - cfg.lock_window
        self.candidates = [c for c in self.candidates if c["samples"][-1][0] >= horizon]
        qualified = []
        for cand in self.candidates:
            cand["samples"] = [s for s in cand["samples"] if s[0] >= horizon]
            raised = [r for _, r in cand["samples"]]
            if t - cand["first_t"] >= cfg.lock_window and sum(raised) >= cfg.lock_fraction * len(raised):
                qualified.append(cand)
        if len(qualified) != 1:
            return None  # nobody yet, or two people at once: keep waiting
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
