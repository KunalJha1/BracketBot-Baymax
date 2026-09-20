"""Pure safety planning for recorded BracketBot arm gestures.

This module deliberately knows nothing about BBOS writers. It validates a
recording against fresh IMU, arm, and depth observations and returns the only
poses the motion thread is allowed to play.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence

import numpy as np


DOF = 8
SIDES = ("left", "right")
ACTIVE_SPAN_TURNS = 0.04
MAX_ENTRY_TURNS = 0.50
UPRIGHT_DEGREES = 25.0
MIN_VALID_DEPTH_POINTS = 150
MIN_BLOCKING_POINTS = 60
HAND_PATH_CLEARANCE_METRES = 0.06
# The real "the hand will touch this" boundary. Never widened, never narrowed,
# never profile-dependent: it is geometry, not caution.
HARD_COLLISION_RADIUS_METRES = 0.035
MIN_HARD_COLLISION_POINTS = 8
# Recordings are slowed for playback. Gestures that invite a person into reach
# keep the original slow pace; the dance keeps it because it is already the
# fastest recording. Everything else is a free-space motion.
DEFAULT_PLAYBACK_SPEED = 0.6
PLAYBACK_SPEEDS = {"wave": 0.75, "goodbye": 0.75, "salute": 0.75, "namaste": 0.75}
# Ease moves scale with distance so a 0.05 turn entry no longer takes as long
# as a 0.5 turn one. Smoothstep peaks at 1.5x the mean speed, so this bounds
# the peak at EASE_PEAK_TURNS_PER_SECOND; the longest allowed entry still takes
# about as long as it always did.
EASE_PEAK_TURNS_PER_SECOND = 0.30
MIN_EASE_SECONDS = 0.5
MAX_EASE_SECONDS = 3.0
IDLE_TURNS = 0.01
IDLE_MARGIN_SECONDS = 0.15
RETURNED_TURNS = 0.08


@dataclass(frozen=True)
class ClearanceProfile:
    """How much padding to keep beyond the hard collision boundary.

    Only the soft margins live here. The upright check and the hard collision
    radius are hazards rather than caution, so no profile can reach them.
    """

    name: str
    hand_path_clearance_m: float
    min_blocking_points: int
    min_valid_depth_points: int

    def __post_init__(self):
        if self.hand_path_clearance_m < HARD_COLLISION_RADIUS_METRES:
            raise ValueError(
                f"{self.name}: clearance {self.hand_path_clearance_m} m is inside the "
                f"hard collision radius {HARD_COLLISION_RADIUS_METRES} m"
            )


# "cautious" is the original behaviour. The looser profiles only shrink padding
# around the recorded hand path, where the hard collision gate still backstops
# every decision, so a smaller margin buys willingness and not contact.
PROFILES = {
    "cautious": ClearanceProfile("cautious", 0.060, 60, 150),
    "balanced": ClearanceProfile("balanced", 0.045, 90, 110),
    "bold": ClearanceProfile("bold", HARD_COLLISION_RADIUS_METRES, 120, 80),
}
DEFAULT_PROFILE = "balanced"
PROFILE_ENV_VAR = "BAYMAX_GESTURE_RISK"


def active_profile() -> ClearanceProfile:
    """The configured profile, falling back to the default when unset or unknown."""
    requested = os.environ.get(PROFILE_ENV_VAR, "").strip().lower()
    if not requested:
        return PROFILES[DEFAULT_PROFILE]
    profile = PROFILES.get(requested)
    if profile is None:
        print(
            f"[gesture-safety] Unknown {PROFILE_ENV_VAR}={requested!r}; "
            f"using {DEFAULT_PROFILE}. Choices: {', '.join(sorted(PROFILES))}.",
            flush=True,
        )
        return PROFILES[DEFAULT_PROFILE]
    return profile


# Conservative arm workspace in the base/IK frame: forward, left, height.
# The middle strip excludes the robot body. Floor points are below this volume.
SWEEP_X_RANGE = (0.08, 0.82)
SWEEP_Z_RANGE = (0.48, 1.52)
SWEEP_Y_RANGES = {
    "left": (0.05, 0.86),
    "right": (-0.86, -0.05),
}


def spoken_safety_refusal(action: str, error: object) -> str:
    """Turn detailed operator diagnostics into one short spoken explanation."""
    detail = str(error).lower()
    label = action.replace("point-left", "point").replace("point-right", "point")
    label = label.replace("goodbye", "wave goodbye")
    if "clearance" in detail or "inside the arm" in detail:
        return f"I can't {label}; there isn't enough clearance. Please step back a little."
    if "not upright" in detail:
        return f"I can't {label} while I'm not upright."
    if "depth" in detail or "surroundings" in detail:
        return f"I can't {label} because I can't check my surroundings."
    if "entry move" in detail or "arm state" in detail:
        return f"I can't {label} safely from this arm position."
    if "already running" in detail:
        return "I'm already doing another movement."
    return f"I can't {label} safely right now."


@dataclass(frozen=True)
class GesturePlan:
    times: np.ndarray
    poses: dict[str, np.ndarray]
    starts: dict[str, np.ndarray]
    sides: tuple[str, ...]
    clearance_points: dict[str, int]


def trajectory_arrays(frames: object) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if isinstance(frames, dict):
        frames = frames.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise RuntimeError("gesture trajectory must contain at least two frames")
    try:
        times = np.asarray([frame["t"] for frame in frames], dtype=np.float64)
        poses = {
            side: np.asarray([frame[side] for frame in frames], dtype=np.float32)
            for side in SIDES
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "gesture frames must contain numeric t, left, and right values"
        ) from exc
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise RuntimeError("gesture timestamps must be finite and strictly increasing")
    for side, trajectory in poses.items():
        if trajectory.shape != (len(frames), DOF):
            raise RuntimeError(
                f"{side} gesture poses must have shape ({len(frames)}, {DOF})"
            )
        if not np.isfinite(trajectory).all():
            raise RuntimeError(f"{side} gesture trajectory contains non-finite values")
    return times - times[0], poses


def playback_speed(name: str) -> float:
    return PLAYBACK_SPEEDS.get(name, DEFAULT_PLAYBACK_SPEED)


def ease_seconds(from_poses: Mapping[str, object], to_poses: Mapping[str, object]) -> float:
    """Shortest ease that keeps the fastest joint under the ease speed limit."""
    distance = max(
        float(np.max(np.abs(np.asarray(to_poses[side]) - np.asarray(from_poses[side]))))
        for side in to_poses
    )
    seconds = 1.5 * distance / EASE_PEAK_TURNS_PER_SECOND
    return float(np.clip(seconds, MIN_EASE_SECONDS, MAX_EASE_SECONDS))


def trim_idle(
    times: np.ndarray, poses: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Drop the motionless lead-in and lead-out of a recording.

    Only stillness is removed: a hold in the middle, or a final hold away from
    the first pose (namaste), is part of the gesture and stays.
    """
    joints = np.concatenate([poses[side][:, 1:7] for side in SIDES], axis=1)
    moved = np.flatnonzero(np.abs(joints - joints[0]).max(axis=1) > IDLE_TURNS)
    if not len(moved):
        return times, dict(poses)
    first = int(np.searchsorted(times, times[moved[0]] - IDLE_MARGIN_SECONDS))
    last = len(times) - 1
    if np.abs(joints[-1] - joints[0]).max() <= RETURNED_TURNS:
        settling = np.flatnonzero(
            np.abs(joints - joints[-1]).max(axis=1) > IDLE_TURNS
        )
        stop = times[settling[-1]] + IDLE_MARGIN_SECONDS
        last = min(last, int(np.searchsorted(times, stop)))
    if last - first < 1:
        return times, dict(poses)
    keep = slice(first, last + 1)
    return times[keep] - times[first], {side: poses[side][keep] for side in SIDES}


def active_sides(poses: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Ignore arms containing only lift-position or recording noise."""
    return tuple(
        side
        for side in SIDES
        if float(np.ptp(poses[side][:, 1:7], axis=0).max()) >= ACTIVE_SPAN_TURNS
    )


def depth_clearance(
    points: object,
    sides: Sequence[str],
    sweep_paths: Mapping[str, object] | None = None,
    profile: ClearanceProfile | None = None,
) -> dict[str, int]:
    """Reject dense geometry in the conservative volume swept by active arms."""
    profile = profile or active_profile()
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite_camera = raw[np.isfinite(raw).all(axis=1)]
    if len(finite_camera) < profile.min_valid_depth_points:
        raise RuntimeError(
            "surroundings check unavailable: not enough fresh depth points "
            f"({len(finite_camera)} < {profile.min_valid_depth_points})"
        )

    # camera.points is lateral/right, forward, height. Arm/base coordinates
    # are forward, lateral/left, height.
    base = np.column_stack(
        (finite_camera[:, 1], -finite_camera[:, 0], finite_camera[:, 2])
    )
    common = (
        (base[:, 0] >= SWEEP_X_RANGE[0])
        & (base[:, 0] <= SWEEP_X_RANGE[1])
        & (base[:, 2] >= SWEEP_Z_RANGE[0])
        & (base[:, 2] <= SWEEP_Z_RANGE[1])
    )
    counts = {}
    blocked = []
    for side in sides:
        if sweep_paths is not None:
            path = np.asarray(sweep_paths.get(side), dtype=np.float64).reshape(-1, 3)
            if not len(path) or not np.isfinite(path).all():
                raise RuntimeError(f"{side} arm sweep path is unavailable")
            # Follow the actual recorded hand path so a table elsewhere in the
            # arm's large theoretical workspace does not cause a false block.
            minimum_distance_squared = np.full(len(base), np.inf)
            radius_squared = profile.hand_path_clearance_m**2
            for waypoint in path:
                minimum_distance_squared = np.minimum(
                    minimum_distance_squared,
                    np.sum((base - waypoint) ** 2, axis=1),
                )
            near_path = minimum_distance_squared <= radius_squared
            count = int(np.count_nonzero(near_path))
            hard_count = int(
                np.count_nonzero(
                    minimum_distance_squared <= HARD_COLLISION_RADIUS_METRES**2
                )
            )
            detail = (
                f"path_min={np.round(path.min(axis=0), 2).tolist()} "
                f"path_max={np.round(path.max(axis=0), 2).tolist()} "
                f"near_8cm={int(np.count_nonzero(minimum_distance_squared <= 0.08**2))} "
                f"near_10cm={int(np.count_nonzero(minimum_distance_squared <= 0.10**2))} "
                f"near_12cm={int(np.count_nonzero(minimum_distance_squared <= 0.12**2))}"
            )
        else:
            y_min, y_max = SWEEP_Y_RANGES[side]
            count = int(
                np.count_nonzero(
                    common
                    & (base[:, 1] >= y_min)
                    & (base[:, 1] <= y_max)
                )
            )
            detail = "broad-workspace fallback"
        counts[side] = count
        # The soft point budget only loosens where a real hand path exists,
        # because only there does the hard collision gate below backstop it.
        # The broad-workspace fallback keeps the original conservative budget.
        soft_limit = (
            profile.min_blocking_points if sweep_paths is not None else MIN_BLOCKING_POINTS
        )
        if count >= soft_limit or (
            sweep_paths is not None and hard_count >= MIN_HARD_COLLISION_POINTS
        ):
            blocked.append(f"{side} arm ({count} depth points; {detail})")
    if blocked:
        raise RuntimeError(
            "gesture blocked because something is inside the arm clearance zone: "
            + ", ".join(blocked)
        )
    return counts


def plan_recorded_gesture(
    frames: object,
    starts: Mapping[str, object],
    rpy_degrees: object,
    depth_points: object | None,
    profile: ClearanceProfile | None = None,
) -> GesturePlan:
    """Validate observations and make a lift-preserving playback plan."""
    times, poses = trajectory_arrays(frames)
    sides = active_sides(poses)
    if not sides:
        raise RuntimeError("gesture recording has no intentional arm movement")
    times, poses = trim_idle(times, poses)

    rpy = np.asarray(rpy_degrees, dtype=np.float64).reshape(-1)
    if len(rpy) < 2 or not np.isfinite(rpy[:2]).all():
        raise RuntimeError("upright check unavailable: IMU orientation is invalid")
    if abs(rpy[0]) >= UPRIGHT_DEGREES or abs(rpy[1]) >= UPRIGHT_DEGREES:
        raise RuntimeError(
            "gesture blocked because the robot is not upright; "
            f"roll={rpy[0]:.1f}°, pitch={rpy[1]:.1f}°"
        )

    safe_starts = {}
    safe_poses = {}
    for side in sides:
        if side not in starts:
            raise RuntimeError(f"fresh {side} arm state is unavailable")
        start = np.asarray(starts[side], dtype=np.float32).reshape(-1)
        if start.shape != (DOF,) or not np.isfinite(start).all():
            raise RuntimeError(f"fresh {side} arm state is invalid")
        trajectory = poses[side].copy()
        # Lift depends on how the robot is parked. Never inherit the recording's
        # lift height and unexpectedly raise or lower the whole arm.
        trajectory[:, 0] = start[0]
        entry = float(np.max(np.abs(trajectory[0] - start)))
        if entry > MAX_ENTRY_TURNS:
            raise RuntimeError(
                f"gesture blocked because the {side} arm entry move is "
                f"{entry:.3f} turns (limit {MAX_ENTRY_TURNS:.2f})"
            )
        safe_starts[side] = start.copy()
        safe_poses[side] = trajectory

    clearance = (
        depth_clearance(depth_points, sides, profile=profile)
        if depth_points is not None
        else {}
    )
    return GesturePlan(times, safe_poses, safe_starts, sides, clearance)
