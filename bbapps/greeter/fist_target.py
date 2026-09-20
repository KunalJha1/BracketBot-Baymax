"""Aim the recorded fist bump at the fist a person is actually holding out.

Pure planning, like ``gesture_safety``: nothing here touches BBOS. The runtime
hands in a depth cloud and a forward-kinematics callable and gets back either
a bent copy of the recording or a reason to play the recording unchanged.

The recording is kept as the shape of the motion. Only the hand position is
moved, by an offset that grows from nothing at the first frame to the full
correction at the apex and back to nothing at the last frame, so the arm still
leaves from and returns to the recorded rest pose.

Three things share the correction. The base turns part of the way toward the
fist (``body_turn_deg``; the runtime does the turning and then looks again),
the lift takes as much of the height as its travel and speed allow, and the
arm bends for what is left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# Where an offered fist can be, in the base frame (forward, left, height).
# The right arm bumps, so the box leans to the robot's right.
FIST_X_RANGE = (0.25, 0.80)
FIST_Y_RANGE = (-0.45, 0.20)
FIST_Z_RANGE = (0.80, 1.50)
# The nearest thing in that box is the candidate. A fist held out is a small
# blob well in front of the body; a chest, a wall or a table edge is wide.
FRONT_SLAB_METRES = 0.06
FIST_RADIUS_METRES = 0.08
MAX_FIST_SPREAD_METRES = 0.075
MIN_FIST_POINTS = 25
MIN_CLUSTER_SHARE = 0.7
# Two looks must agree before the arm is aimed at anything.
STABLE_FIST_METRES = 0.05

# Stop the hand just short of the knuckles; the person closes the gap.
STANDOFF_METRES = 0.02
# How far the apex may move from the recording (forward, left, height). Wider
# than this and the recorded arm shape stops being a good starting point.
MIN_OFFSET = np.array([-0.08, -0.15, -0.20])
MAX_OFFSET = np.array([0.12, 0.15, 0.28])
# Depth points this close to the target are the fist itself, not an obstacle.
TARGET_EXCLUSION_METRES = 0.09

# The base turns this much of the way from where the recording points to where
# the fist is; the arm bends for the rest. All body looks stiff, all arm leaves
# the robot bumping sideways across its own chest.
BODY_TURN_SHARE = 0.6
MIN_BODY_TURN_DEG = 5.0           # the base cannot place a smaller turn reliably
MAX_BODY_TURN_DEG = 30.0

# Height comes from the lift first, because it keeps the recorded arm shape, and
# from bending the arm for whatever is left. The lift rises with the reach and
# sinks with the return, so it must cover its travel in the time the recording
# gives it without going faster than any other eased arm move.
LIFT_PEAK_TURNS_PER_SECOND = 0.30

KEYFRAME_STRIDE = 4
MAX_JOINT_DELTA_TURNS = 0.15
MAX_APEX_ERROR_METRES = 0.02
URDF_JOINT_LIMIT_RAD = 2.0
JOINT_LIMIT_MARGIN_RAD = 0.15
ARM_JOINTS = slice(1, 7)          # motor 0 is the lift, motor 7 the gripper


def camera_to_base(points: object) -> np.ndarray:
    """camera.points is right, forward, height; the arm frame is forward, left, height."""
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    raw = raw[np.isfinite(raw).all(axis=1)]
    return np.column_stack((raw[:, 1], -raw[:, 0], raw[:, 2]))


def find_offered_fist(points: object) -> np.ndarray | None:
    """Front surface of a fist held out toward the robot, or None.

    ``points`` is the raw camera.points cloud. The answer is in the base frame.
    """
    return examine_offered_fist(points)[0]


def examine_offered_fist(points: object) -> tuple[np.ndarray | None, str]:
    """``find_offered_fist`` plus a short reason, for the operator log."""
    base = camera_to_base(points)
    box = base[
        (base[:, 0] >= FIST_X_RANGE[0]) & (base[:, 0] <= FIST_X_RANGE[1])
        & (base[:, 1] >= FIST_Y_RANGE[0]) & (base[:, 1] <= FIST_Y_RANGE[1])
        & (base[:, 2] >= FIST_Z_RANGE[0]) & (base[:, 2] <= FIST_Z_RANGE[1])
    ]
    if len(box) < MIN_FIST_POINTS:
        return None, f"nothing in reach ({len(box)} depth points)"
    nearest = np.percentile(box[:, 0], 2)
    slab = box[box[:, 0] <= nearest + FRONT_SLAB_METRES]
    where = f"nearest thing is {nearest:.2f} m ahead"
    if len(slab) < MIN_FIST_POINTS:
        return None, f"{where} but too sparse ({len(slab)} points)"
    centre = np.median(slab[:, 1:], axis=0)
    for _ in range(2):
        near = np.linalg.norm(slab[:, 1:] - centre, axis=1) <= FIST_RADIUS_METRES
        if np.count_nonzero(near) < MIN_FIST_POINTS:
            return None, f"{where} with no fist-sized blob in it"
        centre = np.median(slab[near, 1:], axis=0)
    cluster = slab[near]
    # Something else at the same depth (a second hand, a chest) means the
    # nearest blob is not clearly "the fist"; leave the recording alone.
    share = len(cluster) / len(slab)
    if share < MIN_CLUSTER_SHARE:
        return None, f"{where} but only {share:.0%} of it is one fist-sized blob"
    spread = np.percentile(np.linalg.norm(cluster[:, 1:] - centre, axis=1), 90)
    if spread > MAX_FIST_SPREAD_METRES:
        return None, f"{where} but it is {2 * spread:.2f} m wide, too big for a fist"
    fist = np.array([np.percentile(cluster[:, 0], 5), centre[0], centre[1]])
    return fist, f"fist at {np.round(fist, 2).tolist()}"


def stable_fist(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    """The agreed fist position when two consecutive looks match."""
    if first is None or second is None:
        return None
    if np.linalg.norm(first - second) > STABLE_FIST_METRES:
        return None
    return (first + second) / 2.0


def without_target(points: object, target: np.ndarray) -> np.ndarray:
    """Camera-frame cloud with the fist being aimed at removed.

    Everything else, including the forearm behind the fist, still counts as an
    obstacle for the clearance check.
    """
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    raw = raw[np.isfinite(raw).all(axis=1)]
    base = np.column_stack((raw[:, 1], -raw[:, 0], raw[:, 2]))
    keep = np.linalg.norm(base - np.asarray(target), axis=1) > TARGET_EXCLUSION_METRES
    return raw[keep]


def body_turn_deg(fist: np.ndarray, apex: np.ndarray) -> float:
    """How far the base should turn (degrees, positive left) before the bump.

    ``apex`` is where the recording peaks. A full turn would put the fist dead
    ahead of the recorded reach; only ``BODY_TURN_SHARE`` of it is asked for.
    """
    fist_bearing = np.degrees(np.arctan2(fist[1], fist[0]))
    apex_bearing = np.degrees(np.arctan2(apex[1], apex[0]))
    turn = BODY_TURN_SHARE * (fist_bearing - apex_bearing)
    if abs(turn) < MIN_BODY_TURN_DEG:
        return 0.0
    return float(np.clip(turn, -MAX_BODY_TURN_DEG, MAX_BODY_TURN_DEG))


def recorded_apex(hand_position, trajectory) -> tuple[int, np.ndarray]:
    """Frame index and hand positions of a recording; the apex is the furthest forward."""
    recorded = np.stack([hand_position(pose) for pose in np.asarray(trajectory, dtype=np.float64)])
    return int(np.argmax(recorded[:, 0])), recorded


def _lift_plan(hand_position, pose, wanted_height, lift_range, max_turns):
    """Lift change (turns) toward ``wanted_height`` metres, and the hand shift it gives."""
    nudged = pose.copy()
    nudged[0] += 0.05
    per_turn = (hand_position(nudged) - hand_position(pose)) / 0.05
    if abs(per_turn[2]) < 1e-3:
        return 0.0, np.zeros(3)
    # A lift already parked outside its range is left where it is, not pulled in.
    low = min(lift_range[0] - pose[0], 0.0)
    high = max(lift_range[1] - pose[0], 0.0)
    turns = float(np.clip(wanted_height / per_turn[2], max(low, -max_turns), min(high, max_turns)))
    return turns, turns * per_turn


def _position_solve(position_of, joints, goal, iterations=12, damping=0.02):
    """Damped least squares on hand position only; smallest joint change wins."""
    joints = joints.copy()
    for _ in range(iterations):
        here = position_of(joints)
        error = goal - here
        if np.linalg.norm(error) < 0.001:
            break
        jacobian = np.empty((3, len(joints)))
        for index in range(len(joints)):
            nudged = joints.copy()
            nudged[index] += 1e-4
            jacobian[:, index] = (position_of(nudged) - here) / 1e-4
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(3), error
        )
        # Motor turns: keep each iteration small so the linearisation holds.
        joints += step * min(1.0, 0.03 / max(float(np.max(np.abs(step))), 1e-9))
    return joints


@dataclass(frozen=True)
class Retarget:
    trajectory: np.ndarray
    apex_index: int
    apex: np.ndarray              # where the hand now peaks, base frame
    offset: np.ndarray            # applied apex correction after clamping
    apex_error_m: float
    max_joint_delta_turns: float
    lift_turns: float = 0.0       # lift change at the apex; zero at both ends
    lift_metres: float = 0.0


def retarget_trajectory(
    hand_position, times, trajectory, fist, urdf_joints=None, lift_range=None, playback_speed=1.0
) -> Retarget:
    """Aim at ``fist``, going only part of the way when the full reach is refused.

    A fist at the edge of the arm's comfortable range still gets a bump that
    visibly heads toward it. RuntimeError means even a quarter of the way was
    refused; the caller then plays the recording as it is.
    """
    refusal = None
    for share in (1.0, 0.75, 0.5, 0.25):
        try:
            return _retarget(
                hand_position, times, trajectory, fist, urdf_joints, share,
                lift_range, playback_speed,
            )
        except RuntimeError as exc:
            refusal = refusal or exc
    raise refusal


def _retarget(
    hand_position: Callable[[np.ndarray], np.ndarray],
    times: np.ndarray,
    trajectory: np.ndarray,
    fist: np.ndarray,
    urdf_joints: Callable[[np.ndarray], np.ndarray] | None = None,
    share: float = 1.0,
    lift_range: tuple[float, float] | None = None,
    playback_speed: float = 1.0,
) -> Retarget:
    """Bend ``trajectory`` (motor turns, N x 8) so its apex meets ``fist``.

    ``hand_position`` maps one motor pose to the hand's base-frame position.
    ``lift_range`` is the lift travel (motor turns) the bump may use; None keeps
    the lift where it is parked. ``share`` scales the arm bend only, since the
    lift does not strain the recorded arm shape.
    Raises RuntimeError when the bent motion is not trustworthy; the caller
    then plays the recording as it is.
    """
    original = np.asarray(trajectory, dtype=np.float64)
    apex_index, recorded = recorded_apex(hand_position, original)
    if apex_index in (0, len(original) - 1):
        raise RuntimeError("fist bump recording has no forward apex")
    goal = np.asarray(fist, dtype=np.float64) - np.array([STANDOFF_METRES, 0.0, 0.0])
    wanted = goal - recorded[apex_index]

    # 0 at both ends, 1 at the apex, smooth in between.
    rise = np.clip(times / times[apex_index], 0.0, 1.0)
    fall = np.clip((times[-1] - times) / (times[-1] - times[apex_index]), 0.0, 1.0)
    weight = np.minimum(rise, fall)
    weight = weight * weight * (3.0 - 2.0 * weight)

    lift_turns, lift_shift = 0.0, np.zeros(3)
    if lift_range is not None:
        # Smoothstep peaks at 1.5x its mean speed.
        shortest = min(times[apex_index], times[-1] - times[apex_index]) / playback_speed
        lift_turns, lift_shift = _lift_plan(
            hand_position, original[apex_index], wanted[2], lift_range,
            LIFT_PEAK_TURNS_PER_SECOND * shortest / 1.5,
        )
    # The lifted recording is the new starting shape; the arm bends from there.
    trajectory = original.copy()
    trajectory[:, 0] += weight * lift_turns
    recorded = recorded + np.outer(weight, lift_shift)
    offset = share * np.clip(wanted - lift_shift, MIN_OFFSET, MAX_OFFSET)

    keyframes = sorted(
        set(range(0, len(trajectory), KEYFRAME_STRIDE)) | {apex_index, len(trajectory) - 1}
    )
    deltas = np.zeros((len(keyframes), 6))
    previous = np.zeros(6)
    for row, index in enumerate(keyframes):
        if weight[index] == 0.0:
            previous = np.zeros(6)
            continue
        pose = trajectory[index]

        def position_of(arm, pose=pose):
            moved = pose.copy()
            moved[ARM_JOINTS] = arm
            return hand_position(moved)

        solved = _position_solve(
            position_of, pose[ARM_JOINTS] + previous, recorded[index] + weight[index] * offset
        )
        previous = deltas[row] = solved - pose[ARM_JOINTS]

    bent = trajectory.copy()
    for joint in range(6):
        bent[:, 1 + joint] += np.interp(
            np.arange(len(trajectory)), keyframes, deltas[:, joint]
        )

    max_delta = float(np.max(np.abs(bent - trajectory)))
    if max_delta > MAX_JOINT_DELTA_TURNS:
        raise RuntimeError(
            f"aimed fist bump bends a joint {max_delta:.3f} turns from the recording "
            f"(limit {MAX_JOINT_DELTA_TURNS:.2f})"
        )
    if urdf_joints is not None:
        def worst_angle(poses):
            return max(
                float(np.max(np.abs(np.asarray(urdf_joints(pose.copy()))[1:7])))
                for pose in poses
            )

        # The recording was played by hand on the real arm, so the angles it
        # reaches are proven; the aimed copy may not go meaningfully past them.
        worst = worst_angle(bent)
        allowed = max(URDF_JOINT_LIMIT_RAD, worst_angle(trajectory) + JOINT_LIMIT_MARGIN_RAD)
        if worst > allowed:
            raise RuntimeError(
                f"aimed fist bump reaches a joint limit ({worst:.2f} rad, allowed {allowed:.2f})"
            )
    apex = hand_position(bent[apex_index])
    apex_error = float(np.linalg.norm(apex - (recorded[apex_index] + offset)))
    if apex_error > MAX_APEX_ERROR_METRES:
        raise RuntimeError(f"aimed fist bump misses its target by {apex_error * 100:.1f} cm")
    return Retarget(
        bent.astype(np.float32), apex_index, apex, offset, apex_error, max_delta,
        lift_turns, float(lift_shift[2]),
    )
