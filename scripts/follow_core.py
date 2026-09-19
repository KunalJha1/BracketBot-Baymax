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
