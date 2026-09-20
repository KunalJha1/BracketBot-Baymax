"""Pick one depth-detected tabletop object with one arm, lift it, put it back.

Runs on the robot next to ``table_rest.py``, ``tabletop_scene.py`` and
``camera_geometry.py`` in ``/tmp``. Plan-only is the default: it scans depth,
selects an object, plans and validates the complete IK path, and never opens
an arm writer. ``--execute`` is required for motion.

Motion sequence (first-milestone behaviour, nothing is carried anywhere):

  live pose -> raise behind table -> home -> pregrasp (gripper opens)
  -> grasp -> close until contact -> lift -> hold -> lower -> open -> retrace

A grasp approaches from behind-above along the shoulder-to-object bearing,
pitched down, with the jaws closing horizontally. Several pitches are tried and
the first fully validated path wins. Stop before contact retraces the path;
Stop while holding lowers the object back to where it was, opens, and retraces.
Torque is never cut while an object is held.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_rest as tr  # noqa: E402
from tabletop_scene import (  # noqa: E402
    TableObject,
    find_box,
    find_objects,
    fit_table_plane,
    object_near,
    points_to_arm,
    select_graspable,
)


UPRIGHT_DEGREES = 25.0
SHOULDER_LATERAL_METRES = 0.0975
# Flatter approaches first: they let the jaws reach the thick middle of an
# object without the fingertips dropping into the table clearance band.
GRASP_PITCHES_DEGREES = (25.0, 35.0, 45.0, 20.0, 55.0)
# Close to the body the arm cannot stretch out flat, so come down steeply.
NEAR_OBJECT_METRES = 0.34
NEAR_GRASP_PITCHES_DEGREES = (60.0, 75.0, 45.0, 85.0, 35.0)
GRASP_HEIGHT_FRACTION = 0.5      # aim for the object's mid-height (its thickest part)
GRASP_BELOW_TOP_METRES = 0.02    # but never above this much below the top
MIN_GRASP_ABOVE_TABLE = 0.065
PREGRASP_BACKOFF_METRES = 0.10
# The IK end-effector point is at the fingertips; the jaws pivot 9 cm behind
# it and the pads meet ~4 cm behind it (URDF). Push the tips this far past
# the object's axis so the pad centre, not the tips, lands on the object.
GRASP_TIP_PAST_AXIS_METRES = 0.045  # past the axis so the pads, not the tips, hold it
PREGRASP_RAISE_METRES = 0.03
LIFT_METRES = 0.10
GRIPPER_OPEN_RADIANS = 0.80
GRIPPER_CLOSED_RADIANS = -0.15
GRIPPER_HOLD_SQUEEZE_RADIANS = 0.06
# Force grip: the arm daemon relieves current on a stalled position target, so a
# held grasp fades. Teleop's full-trigger grasp is 0.70 Nm (~3.4 A at kt=0.204)
# and its comments cap a held grasp near 2.0 Nm.
GRIP_CLOSE_TORQUE_NM = 0.90
GRIP_OPEN_TORQUE_NM = -0.45
MAX_GRIP_TORQUE_NM = 1.50
GRIP_SETTLE_SECONDS = 0.8
CONTACT_CURRENT_AMPS = 0.8
HOLDING_MIN_RADIANS = 0.10
CLOSE_SECONDS = 2.0
APPROACH_SECONDS = 6.0
DESCEND_SECONDS = 3.0
LIFT_SECONDS = 3.0
HOLD_SECONDS = 2.0
RETREAT_SECONDS = 8.0
# Phase durations scale with how far the joints actually travel: the values
# above are ceilings (never slower than before), these are floors, and in
# between a phase runs at the smoothstep peak speeds below.
PEAK_ROTARY_TURNS_PER_SECOND = 0.15
PEAK_LIFT_TURNS_PER_SECOND = 0.30
MIN_PHASE_SECONDS = {"approach": 3.0, "descend": 1.5, "lift": 1.5, "lower": 2.0,
                     "ascend": 1.2, "carry": 2.5, "retreat": 3.0}
GRIPPER_INDEX = 7
# Automatic retry after a miss: nudges (cm, arm frame forward/left/up) tried
# in order from the hover pose, stopping at the first successful grasp. They
# cover the ~1-2 cm scatter of single-frame stereo depth on small objects.
RETRY_NUDGES_CM = ((0, 0, 0), (2, 0, 0), (4, 0, 0), (0, 2, 0), (0, -2, 0), (2, 0, -1.5))
MAX_ADJUST_METRES = 0.10
ADJUST_IK_SAMPLES = 32
# The table edge/apron hazard band: from this far below the surface up to the
# hand-clearance height. Deeper than this the hand is simply under the table
# (where it already hangs at rest), so the edge guard does not apply there.
APRON_DEPTH_METRES = 0.20
# Raising the arm from its hanging rest pose happens right at the table edge
# (bracketbot-184 parks ~0.06 m from it -- the robot's own calibrated startup
# waypoints pass within 7 mm of the strict guard). For that phase the rule is
# just: stay behind the measured edge, or 3 cm clear above the surface. Moves
# out over the table keep the full HAND_CLEARANCE_METRES.
RAISE_EDGE_MARGIN_METRES = 0.03   # the hand is wider than its FK point; 0 let it clip the edge
RAISE_BEHIND_EDGE_METRES = 0.07   # climb this far behind the measured edge
MIN_TABLE_EDGE_METRES = 0.13      # closer than this the raise is cramped
# Automatic spacing: back away from the table in short verified steps.
SPACE_TARGET_EDGE_METRES = 0.17
SPACE_SPEED_MPS = 0.10
SPACE_PULSE_SECONDS = 2.0
SPACE_MAX_PULSES = 5
SPACE_MIN_OBJECT_METRES = 0.34   # nearer than this the arm cannot fold to grasp
SPACE_GOAL_OBJECT_METRES = 0.40
DRIVE_PERIOD_SECONDS = 0.01
RAISE_CLEARANCE_METRES = 0.03
# Hanging rest pose: joints straight, prismatic lift all the way down. The
# lift sign differs per arm (docs/robot-facts.md).
REST_LIFT_TURNS = 1.0
REST_SECONDS = 6.0
# A healthy gripper reads within its URDF range (0..1 rad, a little slack each
# side). bracketbot-184's right gripper reads ~2.26 rad, so its grasp feedback
# is meaningless and that arm must not be trusted to report a hold.
GRIPPER_VALID_RADIANS = (-0.30, 1.20)
GRIPPER_FIX_SECONDS = 6.0
GRIPPER_FIX_MAX_AMPS = 1.5
# Leaning the torso forward buys reach: the shoulder sits ~1.27 m above the
# wheel axis, so each degree of lean moves it ~22 mm further out. BBOS clamps
# a lean request to 1-15 deg and expires it after 0.25 s, so it must be
# republished continuously (docs/robot-facts.md).
SHOULDER_ABOVE_AXLE_METRES = 1.265
MAX_LEAN_DEGREES = 10.0
STABILITY_LEAN_DEGREES = 4.0
LEAN_SETTLE_SECONDS = 3.0
LEAN_PERIOD_SECONDS = 0.05
LEAN_MODE, BALANCE_MODE = 1, 0
# Measured on bracketbot-184: grasps solve to 2-6 mm out to ~0.45 m from the
# shoulder and fail beyond ~0.52 m, where tipping the ~14 cm fingers down
# costs horizontal reach.
IK_CONVERGED_METRES = 0.008
MAX_REACH_METRES = 0.48      # empirical guide only: the IK decides
HARD_REACH_METRES = 0.68     # beyond this no lean can help, so do not try
LEAN_STEP_DEGREES = 2.0
# Placing into a container: hold the object this far above its rim before
# opening, so the jaws clear the walls and the drop is short.
PLACE_ABOVE_RIM_METRES = 0.07
CARRY_SECONDS = 5.0

SERVE_IDLE_SECONDS = 1800

cancel_event = threading.Event()
shutdown_event = threading.Event()


def log(stage, message):
    print(f"[pick][{stage}] {message}", flush=True)


def fmt(values):
    return tr.format_values(values)


def rotation_to_quaternion(matrix):
    """xyzw quaternion for a proper rotation matrix."""

    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def grasp_orientation(pitch_degrees, yaw):
    """Gripper pose: fingertips (local Z) along the approach, jaws (local Y) level.

    At the calibrated home pose the fingertips point forward and the jaws close
    along the robot's lateral axis; this keeps that relationship while pitching
    the fingertips down and turning toward the object.
    """

    pitch = math.radians(pitch_degrees)
    z_axis = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    ])
    y_axis = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    x_axis = np.cross(y_axis, z_axis)
    return rotation_to_quaternion(np.column_stack((x_axis, y_axis, z_axis))), z_axis


def grasp_waypoints(item, side, pitch_degrees, table_height_at, yaw_fraction=1.0):
    """Pregrasp, grasp and lift positions plus the grasp quaternion.

    ``yaw_fraction`` scales the approach heading between straight ahead (0)
    and the full shoulder-to-object bearing (1); round objects accept either,
    and off-to-the-side objects often need a straighter wrist.
    """

    shoulder_y = SHOULDER_LATERAL_METRES if side == "left" else -SHOULDER_LATERAL_METRES
    cx, cy = item.center[0], item.center[1]
    yaw = yaw_fraction * math.atan2(cy - shoulder_y, cx)
    quaternion, approach = grasp_orientation(pitch_degrees, yaw)
    table_z = table_height_at(cx, cy)
    # Reaching GRASP_TIP_PAST_AXIS_METRES past the object's axis also lowers the
    # fingertips by this much, so the grasp height has to allow for it.
    tip_drop = GRASP_TIP_PAST_AXIS_METRES * math.sin(math.radians(pitch_degrees))
    grasp_height = float(np.clip(
        GRASP_HEIGHT_FRACTION * item.top,
        MIN_GRASP_ABOVE_TABLE + tip_drop,
        max(MIN_GRASP_ABOVE_TABLE + tip_drop, item.top - GRASP_BELOW_TOP_METRES),
    ))
    grasp = np.array([cx, cy, table_z + grasp_height]) + GRASP_TIP_PAST_AXIS_METRES * approach
    pregrasp = grasp - PREGRASP_BACKOFF_METRES * approach + np.array([0.0, 0.0, PREGRASP_RAISE_METRES])
    lift = grasp + np.array([0.0, 0.0, LIFT_METRES])
    return pregrasp, grasp, lift, quaternion


def gripper_turns(cfg, pose, radians):
    urdf = np.asarray(cfg.q2urdf(np.asarray(pose, dtype=np.float64).copy()), dtype=np.float64)
    urdf[GRIPPER_INDEX] = radians
    return float(np.asarray(cfg.urdf2q(urdf), dtype=np.float64)[GRIPPER_INDEX])


def gripper_radians(cfg, pose):
    return float(np.asarray(cfg.q2urdf(np.asarray(pose, dtype=np.float64).copy()))[GRIPPER_INDEX])


def densify(path):
    """Subdivide so no command step exceeds table_rest's playback limits."""

    dense = [path[0].copy()]
    for before, after in zip(path[:-1], path[1:]):
        delta = np.abs(after - before)
        steps = max(
            1,
            int(math.ceil(float(delta[0]) / tr.MAX_LIFT_COMMAND_STEP_TURNS)),
            int(math.ceil(float(np.max(delta[1:7])) / tr.MAX_ROTARY_COMMAND_STEP_TURNS)),
        )
        for step in range(1, steps + 1):
            alpha = step / steps
            dense.append((1.0 - alpha) * before + alpha * after)
    return dense


def phase_seconds(poses, ceiling, floor):
    """Playback time for a path: joint travel at the peak speeds, within bounds.

    smoothstep peaks at 1.5x the mean speed, hence the factor.
    """

    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) < 2:
        return float(floor)
    travel = np.sum(np.abs(np.diff(poses[:, :GRIPPER_INDEX], axis=0)), axis=0)
    needed = 1.5 * max(float(travel[0]) / PEAK_LIFT_TURNS_PER_SECOND,
                       float(np.max(travel[1:])) / PEAK_ROTARY_TURNS_PER_SECOND)
    return float(min(ceiling, max(floor, needed)))


def pose_at(poses, fraction):
    """Pose at a 0..1 fraction of a path, interpolated between neighbours.

    Returns the pose and the index of the last path pose already passed, so a
    cancelled move can retrace from there.
    """

    scaled = float(np.clip(fraction, 0.0, 1.0)) * (len(poses) - 1)
    index = min(int(scaled), len(poses) - 1)
    if index >= len(poses) - 1:
        return np.asarray(poses[-1], dtype=np.float64).copy(), len(poses) - 1
    blend = scaled - index
    return ((1.0 - blend) * np.asarray(poses[index], dtype=np.float64)
            + blend * np.asarray(poses[index + 1], dtype=np.float64)), index


def solve_config(cfg, position, quaternion, seeds, template, iterations=20):
    """Converged IK configuration (urdf) for a pose, trying several seeds.

    Returns the first seed's converged solution whose FK lands within
    ``IK_CONVERGED_METRES``, or ``None``. Used as the nominal target for a segment so the
    solver moves toward a known-good branch instead of snapping near the end.
    """

    for seed in seeds:
        seed = np.asarray(seed, dtype=np.float64)
        cfg.ik.reset(list(seed[:7]))
        solution = None
        for _ in range(iterations):
            solution = cfg.ik.solve_with_nominal(list(position), list(quaternion), list(seed[:7]))
            if solution is None:
                break
        if solution is None or len(solution) < 7:
            continue
        candidate = np.asarray(template, dtype=np.float64).copy()
        candidate[:7] = np.asarray(solution[:7], dtype=np.float64)
        reached, _ = cfg.ik.fk(list(candidate[:7]))
        error = float(np.linalg.norm(np.asarray(reached) - np.asarray(position)))
        if error > IK_CONVERGED_METRES:
            continue
        # Seeds are ordered by preference, so the first converged one wins.
        return candidate
    return None


def lift_seeds(template, drops=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
    """Seeds that differ only in how far the prismatic lift (j0) is lowered."""

    seeds = []
    for drop in drops:
        seed = np.asarray(template, dtype=np.float64).copy()
        seed[0] = max(-1.0, seed[0] - drop)
        seeds.append(seed)
    return seeds


def joint_space_raise(cfg, path, escape, quaternions, observation, home_urdf, live_urdf,
                      samples=96):
    """Fallback raise: blend joints to a solved escape pose, checking every step.

    Cartesian IK from some hanging poses flips the elbow part-way up. A joint
    blend cannot branch-jump; each blended pose is checked with FK so the
    hand never passes the table edge below the safe height.
    """

    start = path[-1].copy()
    config = None
    for quaternion in quaternions:
        config = solve_config(cfg, escape, quaternion,
                              [live_urdf, home_urdf, *lift_seeds(home_urdf)], home_urdf)
        if config is not None:
            break
    if config is None:
        raise RuntimeError(f"no converged IK configuration for escape {fmt(escape)}")
    config[GRIPPER_INDEX] = np.asarray(cfg.q2urdf(start.copy()))[GRIPPER_INDEX]
    goal = np.asarray(cfg.urdf2q(config), dtype=np.float64)
    goal[GRIPPER_INDEX] = start[GRIPPER_INDEX]
    edge_guard_x = observation["near_edge"] - 0.03
    minimum_z = observation["height"] + tr.HAND_CLEARANCE_METRES
    apron_z = observation["height"] - APRON_DEPTH_METRES
    for index in range(1, samples + 1):
        alpha = tr.smoothstep(index / samples)
        pose = (1.0 - alpha) * start + alpha * goal
        xyz, _ = cfg.ik.fk(list(np.asarray(cfg.q2urdf(pose.copy()), dtype=np.float64)[:7]))
        if xyz[0] >= edge_guard_x and apron_z <= xyz[2] < minimum_z:
            raise RuntimeError(
                f"joint-space raise crosses the table edge at step {index}: xyz={fmt(xyz)}")
        path.append(pose)
    log("plan", f"raise-behind-table used the joint-space fallback to {fmt(escape)}")


def blend_to(cfg, path, goal, observation, samples=64, label="blend"):
    """Joint-space blend to a motor pose, FK-checking the table edge each step."""

    start = path[-1].copy()
    goal = np.asarray(goal, dtype=np.float64).copy()
    goal[GRIPPER_INDEX] = start[GRIPPER_INDEX]
    edge_guard_x = observation["near_edge"] - RAISE_EDGE_MARGIN_METRES
    minimum_z = observation["height"] + RAISE_CLEARANCE_METRES
    apron_z = observation["height"] - APRON_DEPTH_METRES
    for index in range(1, samples + 1):
        alpha = tr.smoothstep(index / samples)
        pose = (1.0 - alpha) * start + alpha * goal
        xyz, _ = cfg.ik.fk(list(np.asarray(cfg.q2urdf(pose.copy()), dtype=np.float64)[:7]))
        if xyz[0] >= edge_guard_x and apron_z <= xyz[2] < minimum_z:
            raise RuntimeError(f"{label} crosses the table edge at step {index}: xyz={fmt(xyz)}")
        path.append(pose)


def ladder_raise(cfg, path, side, observation, live_urdf, home_urdf, live_quaternion,
                 home_quaternion):
    """Climb behind the table edge: tuck back, rise through solved rungs, clear.

    Each rung is an IK solution seeded from the one below, so neighbouring
    rungs share a branch; joints are blended between rungs and every blended
    pose is FK-checked against the edge. The first rung only pulls the hand
    back, far below the tabletop where there is nothing to hit.
    """

    start_urdf = np.asarray(cfg.q2urdf(path[-1].copy()), dtype=np.float64)
    start_xyz, _ = cfg.ik.fk(list(start_urdf[:7]))
    lateral = SHOULDER_LATERAL_METRES if side == "left" else -SHOULDER_LATERAL_METRES
    climb_x = min(float(start_xyz[0]), observation["near_edge"] - RAISE_BEHIND_EDGE_METRES)
    top = observation["height"] + 0.10
    heights = [float(start_xyz[2])]
    while heights[-1] < top - 1e-6:
        heights.append(min(top, heights[-1] + 0.08))
    seed = start_urdf
    for number, z in enumerate(heights):
        target = np.array([climb_x, lateral, z])
        # Fingertips-down beside the body is an impossible fold near table
        # height, so turn toward the home orientation on the way up.
        fraction = number / max(len(heights) - 1, 1)
        config = None
        for quaternion in (tr.quaternion_slerp(live_quaternion, home_quaternion, fraction),
                           home_quaternion, live_quaternion):
            config = solve_config(cfg, target, quaternion, [seed, live_urdf, home_urdf], home_urdf)
            if config is not None:
                break
        if config is None:
            raise RuntimeError(f"ladder raise: no IK solution at rung {number} {fmt(target)}")
        config[GRIPPER_INDEX] = start_urdf[GRIPPER_INDEX]
        blend_to(cfg, path, np.asarray(cfg.urdf2q(config), dtype=np.float64), observation,
                 samples=24, label=f"ladder rung {number}")
        seed = config
    log("plan", f"ladder raise: {len(heights)} rungs at x={climb_x:.3f} up to z={top:.3f}")


def startup_raise(cfg, path, observation, quiet=[False]):
    """Lift the arm to home along the robot's own calibrated startup waypoints.

    These are the vendor waypoints quest_teleop homes through: they rise close
    to the body (behind the table edge) before reaching out, which Cartesian
    IK from a hanging pose cannot reliably reproduce.
    """

    waypoints = [np.asarray(w, dtype=np.float64) for w in cfg.startup_waypoints]
    waypoints.append(np.asarray(cfg.home, dtype=np.float64))
    for number, waypoint in enumerate(waypoints, start=1):
        blend_to(cfg, path, waypoint, observation, label=f"startup waypoint {number}")
    if not quiet[0]:
        log("plan", f"raised to home along {len(waypoints)} calibrated startup waypoints")
        quiet[0] = True


def raise_to_home(cfg, start, side, observation, live_urdf, live_quaternion, home_quaternion):
    """Motor path from the live pose up to home, solved once per scene.

    The raise does not depend on the grasp pitch or yaw, so the result (or its
    failure) is cached and reused for every option pick_with() tries.
    """

    key = (side, np.asarray(start, dtype=np.float64).tobytes(),
           round(observation["height"], 4), round(observation["near_edge"], 4))
    cached = raise_to_home.cache.get(key)
    if isinstance(cached, str):
        raise RuntimeError(cached)
    if cached is not None:
        return [pose.copy() for pose in cached]
    path = [start.copy()]
    home_cfg = np.asarray(cfg.q2urdf(np.asarray(cfg.home, dtype=np.float64).copy()),
                          dtype=np.float64)
    try:
        try:
            ladder_raise(cfg, path, side, observation, live_urdf, home_cfg, live_quaternion,
                         home_quaternion)
            blend_to(cfg, path, np.asarray(cfg.home, dtype=np.float64), observation,
                     label="ladder to home")
        except RuntimeError as ladder_exc:
            log("plan", f"ladder raise unavailable ({ladder_exc}); trying startup waypoints")
            path = [start.copy()]
            startup_raise(cfg, path, observation)
    except RuntimeError as exc:
        raise_to_home.cache[key] = str(exc)
        raise
    raise_to_home.cache[key] = [pose.copy() for pose in path]
    return path


raise_to_home.cache = {}


def plan_place(cfg, side, lift_pose, place_point, quaternion, observation, low, high):
    """Validated carry phase from the lift pose to above the container."""

    home = np.asarray(cfg.home, dtype=np.float64).copy()
    home[GRIPPER_INDEX] = lift_pose[GRIPPER_INDEX]
    home_urdf = np.asarray(cfg.q2urdf(home.copy()), dtype=np.float64)
    lift_urdf = np.asarray(cfg.q2urdf(np.asarray(lift_pose).copy()), dtype=np.float64)
    config = solve_config(cfg, place_point, quaternion,
                          [lift_urdf, *lift_seeds(home_urdf)], home_urdf)
    if config is None:
        raise RuntimeError(f"no converged IK configuration above the box {fmt(place_point)}")
    path = [np.asarray(lift_pose, dtype=np.float64).copy()]
    position, orientation = (np.asarray(v, dtype=np.float64)
                             for v in cfg.ik.fk(list(lift_urdf[:7])))
    cfg.ik.reset(list(lift_urdf[:7]))
    nominal = config.copy()
    nominal[GRIPPER_INDEX] = lift_urdf[GRIPPER_INDEX]
    tr._append_segment(
        cfg, nominal, path, position, orientation, place_point, quaternion,
        f"{side}:carry-to-box", logger=lambda stage, message: None,
        samples=ADJUST_IK_SAMPLES, endpoint_error_limit=tr.MAX_IK_ENDPOINT_ERROR_METRES,
        clearance_observation=observation,
    )
    carry = np.asarray(densify(list(path)), dtype=np.float64)
    carry[:, GRIPPER_INDEX] = lift_pose[GRIPPER_INDEX]
    tr.validate_calibration(carry, low, high, f"{side}:carry", logger=lambda *_: None)
    tr.validate_playback_clearance(carry, cfg, observation, f"{side}:carry", logger=lambda *_: None)
    return carry


def place_point_for(box, plane, item):
    """Where the gripper should hold the object before opening over the box."""

    x, y = box.center[0], box.center[1]
    rim = plane.height_at(x, y) + box.top
    return np.array([x, y, rim + PLACE_ABOVE_RIM_METRES + item.top * 0.5])


def arm_config(Config, side):
    """One Config per arm for the life of the process (a warm server keeps it)."""

    if side not in arm_config.cache:
        arm_config.cache[side] = Config(f"arm_{side}")
    return arm_config.cache[side]


arm_config.cache = {}


def ready_ik(cfg):
    """Initialise a Config's IK solver once, not once per planning option."""

    if not getattr(cfg, "_pick_ik_ready", False):
        cfg.ik.init()
        try:
            cfg._pick_ik_ready = True
        except AttributeError:  # a Config that refuses new attributes: init each time
            pass


def ordered_options(pitches, remembered=None):
    """Every (pitch, yaw_fraction) to try, the last accepted one first."""

    options = [(pitch, fraction) for fraction in (1.0, 0.5, 0.0) for pitch in pitches]
    if remembered in options:
        options.remove(remembered)
        options.insert(0, remembered)
    return options


def plan_pick(cfg, start, side, item, plane, pitch, yaw_fraction=1.0):
    """Full validated motor path plus the indices where each phase ends."""

    lateral = SHOULDER_LATERAL_METRES if side == "left" else -SHOULDER_LATERAL_METRES
    observation = {
        # Conservative: the plane rises with forward distance, so use the
        # surface height just beyond the object for every clearance check.
        "height": plane.height_at(item.center[0] + 0.03, item.center[1]),
        "near_edge": plane.near_edge_at(item.center[1]),
    }
    # The arm rises at its own lateral offset, where a round table's edge sits
    # further from the robot than it does straight ahead.
    raise_observation = dict(observation, near_edge=plane.near_edge_at(lateral))
    pregrasp, grasp, lift, grasp_quaternion = grasp_waypoints(
        item, side, pitch, plane.height_at, yaw_fraction)
    log(
        "plan",
        f"side={side} raise_edge={raise_observation['near_edge']:.3f} "
        f"pitch={pitch:.0f}deg yaw_fraction={yaw_fraction:.1f} pregrasp={fmt(pregrasp)} grasp={fmt(grasp)} "
        f"lift={fmt(lift)} clearance_height={observation['height']:.3f} "
        f"near_edge={observation['near_edge']:.3f}",
    )
    live_urdf = np.asarray(cfg.q2urdf(start.copy()), dtype=np.float64)
    ready_ik(cfg)
    cfg.ik.reset(list(live_urdf[:7]))
    live_position, live_quaternion = (np.asarray(v, dtype=np.float64) for v in cfg.ik.fk(list(live_urdf[:7])))
    home = np.asarray(cfg.home, dtype=np.float64).copy()
    home[GRIPPER_INDEX] = start[GRIPPER_INDEX]
    home_position, home_quaternion = (
        np.asarray(v, dtype=np.float64)
        for v in cfg.ik.fk(list(np.asarray(cfg.q2urdf(home.copy()), dtype=np.float64)[:7]))
    )
    escape = np.array([min(0.13, plane.near_edge - 0.05), lateral, observation["height"] + 0.08])

    path = [start.copy()]
    marks = {}
    startup_failure = None
    try:
        path = raise_to_home(cfg, start, side, raise_observation, live_urdf, live_quaternion,
                             home_quaternion)
        marks["raise-behind-table"] = marks["move-home"] = len(path) - 1
        home_reached = np.asarray(cfg.q2urdf(path[-1].copy()), dtype=np.float64)
        cfg.ik.reset(list(home_reached[:7]))
        live_position, live_quaternion = (
            np.asarray(v, dtype=np.float64) for v in cfg.ik.fk(list(home_reached[:7])))
    except RuntimeError as exc:
        startup_failure = str(exc)
        if not getattr(plan_pick, "_warned_startup", False):
            log("plan", f"startup-waypoint raise unavailable ({exc}); using Cartesian escape")
            plan_pick._warned_startup = True
        path = [start.copy()]
        marks = {}
    # Same choice as table_rest: the right arm cannot raise behind the table
    # while keeping its hanging (fingertips-down) orientation.
    escape_quaternion = home_quaternion if side == "right" else live_quaternion
    segments = (
        ("raise-behind-table", escape, escape_quaternion, 0.03),
        ("move-home", home_position, home_quaternion, tr.MAX_IK_ENDPOINT_ERROR_METRES),
        ("pregrasp", pregrasp, grasp_quaternion, tr.MAX_IK_ENDPOINT_ERROR_METRES),
        ("grasp", grasp, grasp_quaternion, tr.MAX_IK_ENDPOINT_ERROR_METRES),
        ("lift", lift, grasp_quaternion, tr.MAX_IK_ENDPOINT_ERROR_METRES),
    )
    # Solve the manipulation poses first (grasp, then pregrasp and lift seeded
    # from it) so the lift lowers gradually on the approach instead of
    # jumping in the last few centimetres.
    home_urdf = np.asarray(cfg.q2urdf(home.copy()), dtype=np.float64)
    grasp_config = solve_config(cfg, grasp, grasp_quaternion, lift_seeds(home_urdf), home_urdf)
    if grasp_config is None:
        raise RuntimeError(f"no converged IK configuration for grasp {fmt(grasp)}")
    targets = {}
    for name, point in (("pregrasp", pregrasp), ("lift", lift)):
        targets[name] = solve_config(
            cfg, point, grasp_quaternion, [grasp_config, *lift_seeds(home_urdf)], home_urdf)
        if targets[name] is None:
            raise RuntimeError(f"no converged IK configuration for {name} {fmt(point)}")
    targets["grasp"] = grasp_config
    log("plan", f"grasp_config={fmt(grasp_config)} lift_j0={grasp_config[0]:.3f}m "
                f"home_j0={home_urdf[0]:.3f}m")
    cfg.ik.reset(list(live_urdf[:7]))

    if startup_failure is None:
        segments = tuple(item for item in segments
                         if item[0] not in {"raise-behind-table", "move-home"})
    position, orientation = live_position, live_quaternion
    for name, target, target_quaternion, endpoint_limit in segments:
        current_urdf = np.asarray(cfg.q2urdf(path[-1].copy()), dtype=np.float64)
        nominal = targets.get(name, current_urdf).copy()
        nominal[GRIPPER_INDEX] = current_urdf[GRIPPER_INDEX]
        # The raise can fail with one wrist orientation and succeed with the
        # other (it differs per arm), so try both before giving up.
        options = [target_quaternion]
        if name == "raise-behind-table":
            other = live_quaternion if target_quaternion is home_quaternion else home_quaternion
            options.append(other)
        errors = []
        for option in options:
            attempt = list(path)
            cfg.ik.reset(list(current_urdf[:7]))
            try:
                tr._append_segment(
                    cfg, nominal, attempt, position, orientation, target, option,
                    f"{side}:{name}", logger=lambda stage, message: log(stage, message),
                    endpoint_error_limit=endpoint_limit, clearance_observation=observation,
                )
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            path = attempt
            break
        else:
            if name != "raise-behind-table":
                raise RuntimeError("; ".join(errors))
            attempt = list(path)
            joint_space_raise(cfg, attempt, target, options, raise_observation,
                              np.asarray(cfg.q2urdf(home.copy()), dtype=np.float64), live_urdf)
            path = attempt
        reached_urdf = np.asarray(cfg.q2urdf(path[-1].copy()), dtype=np.float64)
        position, orientation = (np.asarray(v, dtype=np.float64) for v in cfg.ik.fk(list(reached_urdf[:7])))
        marks[name] = len(path) - 1

    path = np.asarray(path, dtype=np.float64)
    total = float(np.max(np.abs(path - start)))
    if total > tr.MAX_TOTAL_MOVE_TURNS:
        raise RuntimeError(f"pick needs a {total:.3f}-turn move, above the safe bound")

    # Open the gripper while travelling to pregrasp, keep it open to grasp.
    open_turns = gripper_turns(cfg, start, GRIPPER_OPEN_RADIANS)
    for index in range(len(path)):
        if index <= marks["pregrasp"]:
            alpha = index / max(marks["pregrasp"], 1)
            path[index, GRIPPER_INDEX] = (1.0 - alpha) * start[GRIPPER_INDEX] + alpha * open_turns
        else:
            path[index, GRIPPER_INDEX] = open_turns
    return path, marks, observation


def split_dense(path, marks):
    """Densify each phase separately so phase boundaries stay addressable."""

    phases = {}
    # "raise" (rest pose up to home, right at the table edge) is validated by
    # its own raise-phase rule while planning; the strict over-table clearance
    # rule applies from home onward.
    edges = [("raise", 0, marks["move-home"]), ("approach", marks["move-home"], marks["pregrasp"]),
             ("descend", marks["pregrasp"], marks["grasp"]), ("lift", marks["grasp"], marks["lift"])]
    for name, first, last in edges:
        phases[name] = np.asarray(densify(list(path[first:last + 1])), dtype=np.float64)
    return phases


def measure_frame(camera_points, near=None, virtual=None, box_side=None, report=log):
    """Table, target and container from one ``camera.points`` cloud.

    Pure numpy: the robot scan and the offline replay (pick_replay.py) share
    it, so detection changes can be tried on recorded frames in seconds.
    Returns ``(plane, item, box)``; ``item`` is None when nothing is graspable.
    """

    arm = points_to_arm(np.asarray(camera_points, dtype=np.float64))
    plane = fit_table_plane(arm)
    objects = find_objects(arm, plane)
    item = select_graspable(objects, near=near, max_reach=1.0)
    if near is not None and (
        item is None
        or math.hypot(item.center[0] - near[0], item.center[1] - near[1]) > 0.08
    ):
        item = object_near(arm, plane, near)
        if item is not None:
            report("scan", "hinted object was merged with a neighbour; measured it locally")
    if item is not None and virtual is not None:
        x, y, top = virtual
        item = TableObject((x, y, plane.height_at(x, y)), top, 0.066, 0.066, 0.0, 0)
    report(
        "scan",
        f"plane tilt={plane.tilt_degrees:.1f}deg inliers={plane.inliers} "
        f"near_edge={plane.near_edge:.3f} objects={len(objects)} "
        f"selected={None if item is None else fmt(item.center)} "
        f"top={0 if item is None else item.top:.3f} "
        f"width={0 if item is None else item.width:.3f}",
    )
    box = None if item is None else find_box(arm, plane, objects, exclude=item, side_of=box_side)
    return plane, item, box


def combine_picks(picks, report=log):
    """One steady target from several per-frame ``(plane, item)`` measurements."""

    centres = np.asarray([item.center for _, item in picks])
    spread = float(np.max(np.ptp(centres[:, :2], axis=0)))
    if spread > 0.05:
        raise RuntimeError(f"object position unstable across frames ({spread:.3f} m)")
    # Single stereo frames jitter by a few centimetres on small shiny objects;
    # the per-axis median across frames is far steadier than any one frame.
    plane, last = picks[-1]
    median = np.median(centres, axis=0)
    item = TableObject(
        (float(median[0]), float(median[1]), float(plane.height_at(median[0], median[1]))),
        float(np.median([item.top for _, item in picks])),
        float(np.median([item.length for _, item in picks])),
        float(np.median([item.width for _, item in picks])),
        last.yaw,
        last.points,
    )
    report("scan", f"median of {len(picks)} frames centre={fmt(item.center)} top={item.top:.3f} "
                   f"spread={spread:.3f}m")
    return plane, item


def scan(Reader, frames=5, timeout=8.0):
    """Fit the table and select a graspable object on fresh, consistent frames."""

    picks = []
    near = scan.near
    last_stamp = None
    deadline = time.monotonic() + timeout
    with tr.nonsuppressing(Reader("camera.points", keeptime=False)) as reader:
        while len(picks) < frames and time.monotonic() < deadline:
            if cancel_event.is_set():
                raise RuntimeError("cancelled during scan")
            if not reader.ready():
                time.sleep(0.02)
                continue
            stamp = str(reader.data["timestamp"])
            if stamp == last_stamp:
                time.sleep(0.02)
                continue
            last_stamp = stamp
            count = int(reader.data["num_points"])
            cloud = np.asarray(reader.data["points"])[:count]
            if scan.record_dir is not None:
                scan.record_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(scan.record_dir / f"frame_{scan.recorded:03d}.npz",
                                    points=cloud.astype(np.float32))
                scan.recorded += 1
            try:
                plane, item, box = measure_frame(cloud, near, scan.virtual, scan.box_side)
            except RuntimeError as exc:
                log("scan", f"frame rejected: {exc}")
                continue
            if item is not None:
                if near is None:
                    near = (item.center[0], item.center[1])  # lock on: no flip-flopping
                picks.append((plane, item))
                scan.last_box = box
    if len(picks) < frames:
        raise RuntimeError(f"no graspable object seen on {frames} fresh depth frames")
    return combine_picks(picks)


scan.near = None
scan.virtual = None
scan.last_box = None
scan.box_side = None
scan.record_dir = None
scan.recorded = 0


def check_reach(item):
    shoulder_y = SHOULDER_LATERAL_METRES if item.center[1] >= 0 else -SHOULDER_LATERAL_METRES
    reach = math.hypot(item.center[0], item.center[1] - shoulder_y)
    log("reach", f"object {reach:.3f} m from the {'left' if shoulder_y > 0 else 'right'} "
                 f"shoulder (limit {MAX_REACH_METRES:.2f})")
    if reach > HARD_REACH_METRES:
        raise RuntimeError(
            f"object is {reach:.2f} m from the shoulder, past anything leaning can reach "
            f"({HARD_REACH_METRES:.2f} m); move it closer"
        )
    if reach > MAX_REACH_METRES:
        log("reach", "beyond the usual limit; letting IK decide, leaning further if it fails")


def plan_adjusted(cfg, side, pregrasp_pose, grasp, lift, quaternion, observation, low, high):
    """Validated descend and lift phases from the hover pose to a nudged grasp."""

    home = np.asarray(cfg.home, dtype=np.float64).copy()
    home[GRIPPER_INDEX] = pregrasp_pose[GRIPPER_INDEX]
    home_urdf = np.asarray(cfg.q2urdf(home.copy()), dtype=np.float64)
    hover_urdf = np.asarray(cfg.q2urdf(np.asarray(pregrasp_pose).copy()), dtype=np.float64)
    grasp_config = solve_config(cfg, grasp, quaternion, [hover_urdf, *lift_seeds(home_urdf)], home_urdf)
    if grasp_config is None:
        raise RuntimeError(f"no converged IK configuration for adjusted grasp {fmt(grasp)}")
    lift_config = solve_config(cfg, lift, quaternion, [grasp_config, *lift_seeds(home_urdf)], home_urdf)
    if lift_config is None:
        raise RuntimeError(f"no converged IK configuration for adjusted lift {fmt(lift)}")
    path = [np.asarray(pregrasp_pose, dtype=np.float64).copy()]
    position, orientation = (np.asarray(v, dtype=np.float64) for v in cfg.ik.fk(list(hover_urdf[:7])))
    marks = {}
    for name, target, config in (("grasp", grasp, grasp_config), ("lift", lift, lift_config)):
        current_urdf = np.asarray(cfg.q2urdf(path[-1].copy()), dtype=np.float64)
        cfg.ik.reset(list(current_urdf[:7]))
        nominal = config.copy()
        nominal[GRIPPER_INDEX] = current_urdf[GRIPPER_INDEX]
        tr._append_segment(
            cfg, nominal, path, position, orientation, target, quaternion,
            f"{side}:adjust-{name}", logger=lambda stage, message: None,
            samples=ADJUST_IK_SAMPLES, endpoint_error_limit=tr.MAX_IK_ENDPOINT_ERROR_METRES,
            clearance_observation=observation,
        )
        reached = np.asarray(cfg.q2urdf(path[-1].copy()), dtype=np.float64)
        position, orientation = (np.asarray(v, dtype=np.float64) for v in cfg.ik.fk(list(reached[:7])))
        marks[name] = len(path) - 1
    path = np.asarray(path, dtype=np.float64)
    path[:, GRIPPER_INDEX] = pregrasp_pose[GRIPPER_INDEX]
    descend = np.asarray(densify(list(path[: marks["grasp"] + 1])), dtype=np.float64)
    lifted = np.asarray(densify(list(path[marks["grasp"]: marks["lift"] + 1])), dtype=np.float64)
    for name, phase in (("adjust-descend", descend), ("adjust-lift", lifted)):
        tr.validate_calibration(phase, low, high, f"{side}:{name}", logger=lambda *_: None)
        tr.validate_playback_clearance(phase, cfg, observation, f"{side}:{name}", logger=lambda *_: None)
    return descend, lifted


def lean_settled(pitches, initial, degrees, window=10, tolerance=0.4):
    """True once the pitch has moved toward the lean and then stopped moving.

    ``pitches`` are IMU pitch samples (deg) since the lean was requested. The
    move must be at least half the requested lean, so a base that has not
    started leaning yet is never mistaken for a settled one.
    """

    if len(pitches) < window:
        return False
    recent = np.asarray(pitches[-window:], dtype=np.float64)
    moved = abs(float(np.median(recent)) - initial) >= 0.5 * degrees
    return bool(moved and float(np.ptp(recent)) <= tolerance)


class LeanHold:
    """Hold a bounded forward lean for as long as the pick needs it.

    The base-mode request expires after ~0.25 s, so a thread republishes it;
    balance is restored on stop, and by expiry if this process dies.
    """

    def __init__(self, Type, Writer, degrees, Reader=None):
        self.Type, self.Writer, self.Reader = Type, Writer, Reader
        self.degrees = float(degrees)
        self.stop = threading.Event()
        self.thread = None

    def _run(self):
        with tr.nonsuppressing(self.Writer("base.mode", self.Type("base_mode"), keeptime=False)) as base:
            while not self.stop.is_set():
                with base.buf() as frame:
                    frame["mode"] = np.uint8(LEAN_MODE)
                    frame["lean_angle_deg"] = np.float32(self.degrees)
                time.sleep(LEAN_PERIOD_SECONDS)
            for _ in range(8):
                with base.buf() as frame:
                    frame["mode"] = np.uint8(BALANCE_MODE)
                    frame["lean_angle_deg"] = np.float32(0.0)
                time.sleep(LEAN_PERIOD_SECONDS)
        log("lean", "balance mode restored")

    def __enter__(self):
        log("lean", f"holding {self.degrees:.1f} deg lean to brace the base")
        self.thread = threading.Thread(target=self._run, name="lean-hold", daemon=True)
        self.thread.start()
        self._settle()
        return self

    def _settle(self):
        """Wait for the lean to arrive and steady; LEAN_SETTLE_SECONDS at most."""
        began = time.monotonic()
        try:
            if self.Reader is None:
                raise RuntimeError("no IMU reader")
            with tr.nonsuppressing(self.Reader("imu.orientation", keeptime=False)) as imu:
                initial = float(np.asarray(tr.fresh(imu)["rpy"])[1])
                pitches = []
                while time.monotonic() - began < LEAN_SETTLE_SECONDS:
                    time.sleep(LEAN_PERIOD_SECONDS)
                    if imu.ready():
                        pitches.append(float(np.asarray(imu.data["rpy"])[1]))
                    if lean_settled(pitches, initial, self.degrees):
                        break
        except Exception as exc:  # noqa: BLE001 - any IMU trouble: use the fixed wait
            log("lean", f"timed settle ({exc})")
            time.sleep(max(0.0, LEAN_SETTLE_SECONDS - (time.monotonic() - began)))
        log("lean", f"settled after {time.monotonic() - began:.1f}s")

    def __exit__(self, *_):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=4.0)
        return False


def lean_for_reach(reach):
    """Starting lean (deg) for an object near or beyond the usual reach limit.

    Trigonometry alone does not predict the gain (tilting the torso also tilts
    the arm frame), so this is only a starting point: execute() leans further
    in LEAN_STEP_DEGREES steps whenever planning still fails.
    """

    if reach <= MAX_REACH_METRES - 0.02:
        return 0.0
    return min(MAX_LEAN_DEGREES, 4.0)


def fix_gripper(Config, Reader, Type, Writer, side):
    """Drive a gripper joint back inside its URDF range, watching current."""

    cfg = Config(f"arm_{side}")
    with tr.nonsuppressing(Reader(f"arm_{side}.state", keeptime=False)) as reader:
        start = np.asarray(tr.fresh(reader)["pos"], dtype=np.float64).copy()
        radians = gripper_radians(cfg, start)
        log("gripper", f"{side} reads {radians:.3f} rad ({start[GRIPPER_INDEX]:.3f} turns)")
        goal = start.copy()
        goal[GRIPPER_INDEX] = float(np.asarray(cfg.home, dtype=np.float64)[GRIPPER_INDEX])
        log("gripper", f"{side} moving gripper to its home {goal[GRIPPER_INDEX]:.3f} turns "
                       f"over {GRIPPER_FIX_SECONDS:.0f}s, stopping above {GRIPPER_FIX_MAX_AMPS} A")
        with ExitStack() as stack:
            control = stack.enter_context(tr.nonsuppressing(
                Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)))
            torque = stack.enter_context(tr.nonsuppressing(
                Writer(f"arm_{side}.torque", Type("arm_torque"), keeptime=False)))

            def command(pose):
                with control.buf() as frame:
                    frame["pos"][:] = np.asarray(pose, dtype=np.float32)
                    frame["vel"][:] = 0
                    frame["tau"][:] = 0
                    frame["alpha"] = 0.0

            def set_torque(enabled):
                with torque.buf() as frame:
                    frame["enable"][:] = enabled
                    frame["tau_mode"][:] = False
                    frame["compliance_mode"] = False

            for _ in range(8):
                command(start)
                time.sleep(tr.TICK_SECONDS)
            set_torque(True)
            began = time.monotonic()
            pose = start.copy()
            try:
                while True:
                    alpha = min((time.monotonic() - began) / GRIPPER_FIX_SECONDS, 1.0)
                    pose[GRIPPER_INDEX] = ((1.0 - alpha) * start[GRIPPER_INDEX]
                                           + alpha * goal[GRIPPER_INDEX])
                    command(pose)
                    time.sleep(tr.TICK_SECONDS)
                    data = reader.data
                    current = abs(float(np.asarray(data["current"])[GRIPPER_INDEX]))
                    if current >= GRIPPER_FIX_MAX_AMPS:
                        log("gripper", f"{side} stopped at {current:.2f} A "
                                       f"({gripper_radians(cfg, np.asarray(data['pos'])):.3f} rad)")
                        break
                    if alpha >= 1.0 or cancel_event.is_set():
                        break
            finally:
                time.sleep(0.3)
                set_torque(False)
        final = gripper_radians(cfg, np.asarray(tr.fresh(reader)["pos"], dtype=np.float64))
    healthy = GRIPPER_VALID_RADIANS[0] <= final <= GRIPPER_VALID_RADIANS[1]
    log("gripper", f"{side} now reads {final:.3f} rad; in range={healthy}")
    return healthy


def measure_edge(Reader, frames=3):
    """Median table near-edge distance over a few fresh depth frames."""

    edges, last = [], None
    deadline = time.monotonic() + 6.0
    with tr.nonsuppressing(Reader("camera.points", keeptime=False)) as reader:
        while len(edges) < frames and time.monotonic() < deadline:
            if not reader.ready():
                time.sleep(0.02)
                continue
            stamp = str(reader.data["timestamp"])
            if stamp == last:
                time.sleep(0.02)
                continue
            last = stamp
            count = int(reader.data["num_points"])
            arm = points_to_arm(np.asarray(reader.data["points"])[:count].astype(np.float64))
            try:
                edges.append(fit_table_plane(arm).near_edge)
            except RuntimeError:
                continue
    if not edges:
        raise RuntimeError("cannot see the table to measure spacing")
    return float(np.median(edges))


def nearest_target_distance(Reader):
    """Forward distance of the object the pick would choose, or None."""

    with tr.nonsuppressing(Reader("camera.points", keeptime=False)) as reader:
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if reader.ready():
                count = int(reader.data["num_points"])
                arm = points_to_arm(np.asarray(reader.data["points"])[:count].astype(np.float64))
                try:
                    plane = fit_table_plane(arm)
                except RuntimeError:
                    return None
                objects = find_objects(arm, plane)
                item = select_graspable(objects, near=scan.near, max_reach=1.0)
                if item is None and scan.near is not None:
                    item = object_near(arm, plane, scan.near)
                return None if item is None else float(item.center[0])
            time.sleep(0.02)
    return None


def auto_space(Reader, Type, Writer):
    """Back the base up until the target sits at a comfortable grasp distance.

    Depth is the ground truth: drive a continuous pulse (a balancing base only
    wiggles on short ones), stop, let it settle, measure the object again, and
    repeat. Backward only, so it can never drive into the table.
    """

    distance = nearest_target_distance(Reader)
    if distance is None:
        log("space", "no target visible; skipping spacing")
        return
    log("space", f"target is {distance:.3f} m ahead (want >= {SPACE_MIN_OBJECT_METRES:.2f})")
    if distance >= SPACE_MIN_OBJECT_METRES:
        return
    with tr.nonsuppressing(Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False)) as drive:

        def twist(v):
            with drive.buf() as frame:
                frame["twist"] = np.array([v, 0.0], dtype=np.float32)

        try:
            for pulse in range(1, SPACE_MAX_PULSES + 1):
                if cancel_event.is_set() or distance >= SPACE_GOAL_OBJECT_METRES:
                    break
                began = time.monotonic()
                while time.monotonic() - began < SPACE_PULSE_SECONDS and not cancel_event.is_set():
                    twist(-SPACE_SPEED_MPS)
                    time.sleep(DRIVE_PERIOD_SECONDS)
                for _ in range(60):
                    twist(0.0)
                    time.sleep(DRIVE_PERIOD_SECONDS)
                time.sleep(1.5)  # let the balancer settle before trusting depth
                measured = nearest_target_distance(Reader)
                if measured is None:
                    log("space", "lost sight of the target; stopping the spacing")
                    break
                log("space", f"pulse {pulse}: target {distance:.3f} -> {measured:.3f} m")
                distance = measured
        finally:
            for _ in range(30):
                twist(0.0)
                time.sleep(DRIVE_PERIOD_SECONDS)
    log("space", f"spacing done: target at {distance:.3f} m")


def rest_arms(Config, Reader, Type, Writer, sides=("left", "right")):
    """Lower the arms back to their hanging rest pose, then release torque."""

    with tr.nonsuppressing(Reader("camera.points", keeptime=False)) as points:
        deadline = time.monotonic() + 4.0
        arm = None
        while time.monotonic() < deadline:
            if points.ready():
                count = int(points.data["num_points"])
                arm = points_to_arm(np.asarray(points.data["points"])[:count].astype(np.float64))
                break
            time.sleep(0.02)
    plane = fit_table_plane(arm) if arm is not None else None
    observation = ({"height": plane.height_at(0.3, 0.0), "near_edge": plane.near_edge}
                   if plane is not None else {"height": -10.0, "near_edge": 10.0})
    log("rest", f"table height={observation['height']:.3f} near_edge={observation['near_edge']:.3f}")

    for side in sides:
        cfg = Config(f"arm_{side}")
        cfg.ik.init()
        with tr.nonsuppressing(Reader(f"arm_{side}.state", keeptime=False)) as reader:
            start = np.asarray(tr.fresh(reader)["pos"], dtype=np.float64).copy()
        lift_sign = 1.0 if side == "left" else -1.0
        straight = start.copy()
        straight[1:7] = 0.0
        hanging = straight.copy()
        hanging[0] = lift_sign * REST_LIFT_TURNS
        path = [start.copy()]
        try:
            blend_to(cfg, path, straight, observation, label=f"{side} straighten")
            blend_to(cfg, path, hanging, observation, label=f"{side} lower")
        except RuntimeError as exc:
            log("rest", f"{side} rest path rejected: {exc}")
            continue
        xyz, _ = cfg.ik.fk(list(np.asarray(cfg.q2urdf(path[-1].copy()), dtype=np.float64)[:7]))
        log("rest", f"{side} start={fmt(start)} -> rest={fmt(hanging)} hand_xyz={fmt(xyz)}")
        with ExitStack() as stack:
            control = stack.enter_context(tr.nonsuppressing(
                Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)))
            torque = stack.enter_context(tr.nonsuppressing(
                Writer(f"arm_{side}.torque", Type("arm_torque"), keeptime=False)))

            def command(pose):
                with control.buf() as frame:
                    frame["pos"][:] = np.asarray(pose, dtype=np.float32)
                    frame["vel"][:] = 0
                    frame["tau"][:] = 0
                    frame["alpha"] = 0.0

            def set_torque(enabled):
                with torque.buf() as frame:
                    frame["enable"][:] = enabled
                    frame["tau_mode"][:] = False
                    frame["compliance_mode"] = False

            for _ in range(8):
                command(start)
                time.sleep(tr.TICK_SECONDS)
            set_torque(True)
            began = time.monotonic()
            while True:
                alpha = min((time.monotonic() - began) / REST_SECONDS, 1.0)
                index = min(int(tr.smoothstep(alpha) * (len(path) - 1)), len(path) - 1)
                command(path[index])
                if alpha >= 1.0 or cancel_event.is_set():
                    break
                time.sleep(tr.TICK_SECONDS)
            time.sleep(0.3)
            set_torque(False)
        log("rest", f"{side} arm is at its rest pose; torque off")


def execute(plan_only=True, pid_file=None, stop_at=None, adjust=False,
            grip_torque=GRIP_CLOSE_TORQUE_NM, allow_lean=True, place=False,
            auto_space_enabled=False):
    bbos, Config, Reader, Type, Writer = tr._load_bbos()
    TIMING.mark("bbos ready")
    with tr.nonsuppressing(Reader("imu.orientation", keeptime=False)) as imu:
        rpy = np.asarray(tr.fresh(imu)["rpy"], dtype=np.float64)
    log("preflight", f"imu_rpy={fmt(rpy)}")
    if abs(rpy[0]) >= UPRIGHT_DEGREES or abs(rpy[1]) >= UPRIGHT_DEGREES:
        raise RuntimeError("robot is not upright")

    def space_if_needed():
        # Must run with lean already held: entering lean mode shifts the base
        # forward ~17 cm (measured), which would undo any spacing done before.
        if not auto_space_enabled:
            return
        try:
            auto_space(Reader, Type, Writer)
        except RuntimeError as exc:
            log("space", f"continuing without spacing: {exc}")

    # A balancing base rocks as the arm extends, which moves the hand away from
    # where depth measured the object. Lean mode braces the base, so hold it for
    # the whole pick (both successful early picks ran with lean on; the miss
    # without it). Skip our own hold if another process already publishes lean.
    external = False
    if allow_lean:
        try:
            with tr.nonsuppressing(Reader("base.mode", keeptime=False)) as mode_reader:
                deadline = time.monotonic() + 0.6
                while time.monotonic() < deadline and not external:
                    if mode_reader.ready() and int(mode_reader.data["mode"]) == LEAN_MODE:
                        external = True
                    time.sleep(0.05)
        except Exception:  # noqa: BLE001 - topic absent means nobody holds lean
            external = False
    if not allow_lean or external:
        log("lean", "lean already held elsewhere" if external else "lean disabled by flag")
        space_if_needed()
        plane, item = scan(Reader)
        return pick_with(bbos, Config, Reader, Type, Writer, plane, item,
                         plan_only, stop_at, adjust, grip_torque, box_for(plane, item, place))
    with LeanHold(Type, Writer, STABILITY_LEAN_DEGREES, Reader):
        TIMING.mark("lean settled")
        space_if_needed()
        plane, item = scan(Reader)
        return pick_with(bbos, Config, Reader, Type, Writer, plane, item,
                         plan_only, stop_at, adjust, grip_torque, box_for(plane, item, place))


def box_for(plane, item, place):
    """Locate the container to place into, or None to put the object back."""

    if not place:
        return None
    box = scan.last_box
    if box is None:
        log("place", "no container found near the object; it will be put back instead")
    return box


def object_reach(item):
    shoulder_y = SHOULDER_LATERAL_METRES if item.center[1] >= 0 else -SHOULDER_LATERAL_METRES
    return math.hypot(item.center[0], item.center[1] - shoulder_y)


def pick_with(bbos, Config, Reader, Type, Writer, plane, item,
              plan_only, stop_at, adjust, grip_torque, place_box=None):
    if plane.near_edge < MIN_TABLE_EDGE_METRES:
        log("plan", f"table edge is only {plane.near_edge * 100:.0f} cm from the arm base; "
                    "the raise will climb behind it")
    check_reach(item)
    preferred = "left" if item.center[1] >= 0.0 else "right"
    rejected = []
    for side in (preferred, "right" if preferred == "left" else "left"):
        cfg = arm_config(Config, side)
        with tr.nonsuppressing(Reader(f"arm_{side}.state", keeptime=False)) as state_reader:
            start = np.asarray(tr.fresh(state_reader)["pos"], dtype=np.float64).copy()
        radians = gripper_radians(cfg, start)
        log("state", f"side={side} start={fmt(start)} gripper={radians:.3f}rad")
        if not GRIPPER_VALID_RADIANS[0] <= radians <= GRIPPER_VALID_RADIANS[1]:
            rejected.append(f"{side} gripper reads {radians:.2f} rad, outside "
                            f"{GRIPPER_VALID_RADIANS}; its grasp feedback cannot be trusted")
            log("state", f"side={side} rejected: {rejected[-1]}")
            continue
        shoulder_y = SHOULDER_LATERAL_METRES if side == "left" else -SHOULDER_LATERAL_METRES
        reach = math.hypot(item.center[0], item.center[1] - shoulder_y)
        log("state", f"side={side} object reach {reach:.3f} m")
        if reach > HARD_REACH_METRES:
            rejected.append(f"{side} arm would need {reach:.2f} m of reach")
            log("state", f"side={side} rejected: {rejected[-1]}")
            continue
        break
    else:
        raise RuntimeError("no usable arm: " + "; ".join(rejected))

    failures = []
    pitches = (NEAR_GRASP_PITCHES_DEGREES if object_reach(item) < NEAR_OBJECT_METRES
               else GRASP_PITCHES_DEGREES)
    # The object rarely moves between tests, so the option that validated last
    # time is tried first instead of re-rejecting the ones before it.
    options = ordered_options(pitches, pick_with.accepted.get(side))
    TIMING.mark("scan done, planning")
    for pitch, yaw_fraction in options:
        try:
            path, marks, observation = plan_pick(cfg, start, side, item, plane, pitch, yaw_fraction)
            phases = split_dense(path, marks)
            low, high = tr.calibration_limits(bbos, side, logger=log)
            for name, phase in phases.items():
                tr.validate_calibration(phase, low, high, f"{side}:{name}", logger=log)
                if name != "raise":
                    tr.validate_playback_clearance(phase, cfg, observation, f"{side}:{name}",
                                                   logger=log)
            phases["approach"] = np.vstack([phases.pop("raise"), phases["approach"][1:]])
            break
        except RuntimeError as exc:
            failures.append(f"pitch {pitch:.0f} yaw x{yaw_fraction:.1f}: {exc}")
            log("plan", f"pitch={pitch:.0f}deg yaw_fraction={yaw_fraction:.1f} rejected: {exc}")
    else:
        raise RuntimeError("no validated grasp: " + " | ".join(failures))
    grasp_urdf = np.asarray(cfg.q2urdf(phases["descend"][-1].copy()), dtype=np.float64)
    reached_xyz, _ = cfg.ik.fk(list(grasp_urdf[:7]))
    pick_with.accepted[side] = (pitch, yaw_fraction)
    TIMING.mark(f"plan validated after {len(failures) + 1} option(s)")
    log("plan", f"grasp pose reaches {fmt(reached_xyz)}")
    log("plan", f"accepted pitch={pitch:.0f}deg yaw_fraction={yaw_fraction:.1f} phases=" + ",".join(
        f"{name}:{len(phase)}" for name, phase in phases.items()))
    if plan_only:
        log("complete", "plan-only succeeded; no arm writers were opened")
        return

    base_pregrasp, base_grasp, base_lift, grasp_quaternion = grasp_waypoints(
        item, side, pitch, plane.height_at, yaw_fraction)

    def plan_extras():
        # Runs while the arm plays the approach: that phase only streams poses
        # and never touches cfg.ik, and run_motion() collects this before it
        # descends, so the solver is never shared between threads.
        attempts = None
        if adjust:
            attempts = []
            for nudge_cm in RETRY_NUDGES_CM:
                offset = np.asarray(nudge_cm, dtype=np.float64) / 100.0
                if np.any(np.abs(offset) > MAX_ADJUST_METRES):
                    continue
                if not np.any(offset):
                    attempts.append((offset, phases["descend"], phases["lift"]))
                    continue
                try:
                    descend, lifted = plan_adjusted(
                        cfg, side, phases["approach"][-1], base_grasp + offset,
                        base_lift + offset, grasp_quaternion, observation, low, high)
                except RuntimeError as exc:
                    log("retry", f"nudge_cm={fmt(offset * 100)} rejected: {exc}")
                    continue
                attempts.append((offset, descend, lifted))
            log("retry", f"{len(attempts)} validated attempts planned during the approach")
        carry = None
        if place_box is not None and attempts:
            target = place_point_for(place_box, plane, item)
            try:
                carry = plan_place(cfg, side, phases["lift"][-1], target, grasp_quaternion,
                                   observation, low, high)
                log("place", f"box at {fmt(place_box.center)} rim={place_box.top:.3f}; "
                             f"release above {fmt(target)} ({len(carry)} poses)")
            except RuntimeError as exc:
                log("place", f"cannot reach above the box ({exc}); the object will be put back")
        return attempts, carry

    TIMING.mark("motion starts")
    run_motion(Config, Reader, Type, Writer, cfg, side, start, phases, stop_at,
               grip_torque, Deferred(plan_extras))


class Deferred:
    """Run ``work`` on a thread now; ``result()`` waits for it and re-raises."""

    def __init__(self, work):
        self._work, self._value, self._error = work, None, None
        self._thread = threading.Thread(target=self._run, name="deferred-plan", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._value = self._work()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller of result()
            self._error = exc

    def wait(self):
        self._thread.join()

    def result(self):
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._value


class Timing:
    """Seconds since the job began, logged at each startup milestone."""

    def __init__(self):
        self.began = self.last = time.monotonic()

    def restart(self):
        self.began = self.last = time.monotonic()

    def mark(self, label):
        now = time.monotonic()
        log("timing", f"{label}: +{now - self.last:.2f}s (t={now - self.began:.2f}s)")
        self.last = now


TIMING = Timing()


pick_with.accepted = {}


def run_motion(Config, Reader, Type, Writer, cfg, side, start, phases, stop_at=None,
               grip_torque=GRIP_CLOSE_TORQUE_NM, extras=None):
    attempts = carry = None
    with ExitStack() as stack:
        state = stack.enter_context(tr.nonsuppressing(Reader(f"arm_{side}.state", keeptime=False)))
        control = stack.enter_context(tr.nonsuppressing(
            Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)))
        torque = stack.enter_context(tr.nonsuppressing(
            Writer(f"arm_{side}.torque", Type("arm_torque"), keeptime=False)))

        grip = {"tau": 0.0, "force_mode": False}

        def command(pose):
            with control.buf() as frame:
                frame["pos"][:] = np.asarray(pose, dtype=np.float32)
                frame["vel"][:] = 0
                frame["tau"][:] = 0
                # In force mode the daemon ignores pos for the gripper joint and
                # applies this torque, so the squeeze cannot fade as it would
                # against a stalled position target.
                frame["tau"][GRIPPER_INDEX] = cfg.gripper_sign * grip["tau"]
                frame["alpha"] = 0.0

        def set_torque(enabled, force_grip=False):
            with torque.buf() as frame:
                frame["enable"][:] = enabled
                frame["tau_mode"][:] = False
                frame["tau_mode"][GRIPPER_INDEX] = bool(enabled and force_grip)
                frame["compliance_mode"] = False
            grip["force_mode"] = bool(enabled and force_grip)
            log("torque", f"enabled={enabled} gripper_force_mode={grip['force_mode']}")

        def hold_with_force(pose, newtons_metre):
            """Switch the gripper to a constant squeeze; report if it holds."""
            grip["tau"] = newtons_metre
            set_torque(True, force_grip=True)
            began = time.monotonic()
            while time.monotonic() - began < GRIP_SETTLE_SECONDS:
                command(pose)
                time.sleep(tr.TICK_SECONDS)
            data = tr.fresh(state)
            radians = gripper_radians(cfg, np.asarray(data["pos"]))
            current = abs(float(np.asarray(data["current"])[GRIPPER_INDEX]))
            holding = radians >= HOLDING_MIN_RADIANS
            log("grip", f"force={newtons_metre:.2f}Nm gripper={radians:.3f}rad "
                        f"current={current:.2f}A holding={holding}")
            return holding

        def release(pose):
            """Push the jaws open under torque, then return to position control."""
            if grip["force_mode"]:
                grip["tau"] = GRIP_OPEN_TORQUE_NM
                began = time.monotonic()
                while time.monotonic() - began < GRIP_SETTLE_SECONDS:
                    command(pose)
                    time.sleep(tr.TICK_SECONDS)
                    # Stop pushing once the jaws are open: a blind 0.8 s shove
                    # can drive the joint past its range (see GRIPPER_VALID_RADIANS).
                    if state.ready() and gripper_radians(
                            cfg, np.asarray(state.data["pos"])) >= GRIPPER_OPEN_RADIANS:
                        break
            grip["tau"] = 0.0
            set_torque(True, force_grip=False)
            command(pose)
            time.sleep(0.4)

        def play(poses, ceiling, cancellable, stage, pace=None):
            # ``pace`` names the MIN_PHASE_SECONDS floor; short moves then run
            # faster than the ceiling instead of crawling through a fixed time.
            seconds = ceiling if pace is None else phase_seconds(
                poses, ceiling, MIN_PHASE_SECONDS[pace])
            began = time.monotonic()
            index = 0
            while True:
                alpha = min((time.monotonic() - began) / seconds, 1.0)
                # Interpolate between path poses: stepping whole indices made
                # the arm advance in stair-steps whenever ticks outnumber poses.
                pose, index = pose_at(poses, tr.smoothstep(alpha))
                command(pose)
                if alpha >= 1.0 or (cancellable and cancel_event.is_set()):
                    log("motion", f"stage={stage} end index={index}/{len(poses) - 1} "
                                  f"seconds={seconds:.1f} cancelled={cancel_event.is_set()}")
                    return index
                time.sleep(tr.TICK_SECONDS)

        def with_gripper(poses, turns):
            poses = np.array(poses, dtype=np.float64)
            poses[:, GRIPPER_INDEX] = turns
            return poses

        def close_until_contact(pose):
            """Close in position mode; stop at current spike or stall."""
            open_turns = pose[GRIPPER_INDEX]
            closed_turns = gripper_turns(cfg, pose, GRIPPER_CLOSED_RADIANS)
            began = time.monotonic()
            target = pose.copy()
            while True:
                alpha = min((time.monotonic() - began) / CLOSE_SECONDS, 1.0)
                target[GRIPPER_INDEX] = (1.0 - alpha) * open_turns + alpha * closed_turns
                command(target)
                time.sleep(tr.TICK_SECONDS)
                data = tr.fresh(state)
                current = abs(float(np.asarray(data["current"])[GRIPPER_INDEX]))
                measured = float(np.asarray(data["pos"])[GRIPPER_INDEX])
                if current >= CONTACT_CURRENT_AMPS or alpha >= 1.0:
                    break
            measured_pose = np.asarray(tr.fresh(state)["pos"], dtype=np.float64)
            measured_rad = gripper_radians(cfg, measured_pose)
            hold_turns = gripper_turns(cfg, pose, measured_rad - GRIPPER_HOLD_SQUEEZE_RADIANS)
            target[GRIPPER_INDEX] = hold_turns
            command(target)
            holding = measured_rad >= HOLDING_MIN_RADIANS
            log("grip", f"contact_current={current:.2f}A measured={measured_rad:.3f}rad "
                        f"raw_turns={measured:.3f} hold_turns={hold_turns:.3f} holding={holding}")
            return holding, hold_turns

        approach, descend, lift = phases["approach"], phases["descend"], phases["lift"]
        open_turns = descend[-1][GRIPPER_INDEX]
        enabled = False
        stage = "approach"
        reached = 0
        holding = False
        try:
            for _ in range(8):
                command(start)
                time.sleep(tr.TICK_SECONDS)
            set_torque(True)
            enabled = True
            reached = play(approach, APPROACH_SECONDS, True, "approach", pace="approach")
            if extras is not None:
                attempts, carry = extras.result()
            if cancel_event.is_set():
                return
            if stop_at == "pregrasp":
                log("staged", "hovering at pregrasp; returning without descending")
                cancel_event.wait(HOLD_SECONDS)
                return
            if attempts:
                for number, (offset, descend, lift) in enumerate(attempts, start=1):
                    if cancel_event.is_set():
                        break
                    log("retry", f"attempt={number}/{len(attempts)} nudge_cm={fmt(offset * 100)}")
                    stage = "descend"
                    reached = play(descend, DESCEND_SECONDS, True, "descend", pace="descend")
                    if cancel_event.is_set():
                        break
                    stage = "grip"
                    holding, hold_turns = close_until_contact(descend[-1].copy())
                    if holding:
                        holding = hold_with_force(descend[-1].copy(), grip_torque)
                    result = "PICKED" if holding else "MISS"
                    if holding:
                        stage = "lift"
                        reached = play(with_gripper(lift, hold_turns), LIFT_SECONDS, False,
                                       "lift", pace="lift")
                        data = tr.fresh(state)
                        still = gripper_radians(cfg, np.asarray(data["pos"])) >= HOLDING_MIN_RADIANS
                        result = "PICKED" if still else "SLIPPED"
                        log("evidence", f"attempt={number} after lift holding={still} "
                                        f"current={float(np.asarray(data['current'])[GRIPPER_INDEX]):.2f}A")
                        cancel_event.wait(HOLD_SECONDS)
                        if carry is not None and still:
                            # Carrying is deliberately not cancellable: a Stop
                            # mid-carry finishes over the box and releases there
                            # rather than dropping the object anywhere.
                            stage = "carry"
                            play(with_gripper(carry, hold_turns), CARRY_SECONDS, False, "carry",
                                 pace="carry")
                            release(with_gripper([carry[-1]], hold_turns)[0])
                            holding = False
                            result = "PLACED"
                            log("place", "released above the box")
                            play(with_gripper(carry[::-1], open_turns), CARRY_SECONDS, False,
                                 "leave-box", pace="carry")
                            play(with_gripper(lift[::-1], open_turns), LIFT_SECONDS, False,
                                 "lower-empty", pace="ascend")
                        else:
                            stage = "put-back"
                            play(with_gripper(lift[::-1], hold_turns), LIFT_SECONDS, False,
                                 "lower", pace="lower")
                            holding = False
                    open_pose = descend[-1].copy()
                    open_pose[GRIPPER_INDEX] = open_turns
                    release(open_pose)
                    command(open_pose)
                    time.sleep(0.6)
                    play(with_gripper(descend[::-1], open_turns), DESCEND_SECONDS, False, "ascend",
                         pace="ascend")
                    stage = "approach"
                    reached = len(approach) - 1
                    log("retry", f"attempt={number} result={result}")
                    if result in {"PICKED", "PLACED"}:
                        break
                return
            stage = "descend"
            reached = play(descend, DESCEND_SECONDS, True, "descend", pace="descend")
            if cancel_event.is_set():
                return
            if stop_at == "grasp":
                time.sleep(0.6)
                measured = np.asarray(tr.fresh(state)["pos"], dtype=np.float64)
                measured_xyz, _ = cfg.ik.fk(list(np.asarray(cfg.q2urdf(measured.copy()))[:7]))
                planned_xyz, _ = cfg.ik.fk(list(np.asarray(cfg.q2urdf(descend[-1].copy()))[:7]))
                log("staged", f"planned tip {fmt(planned_xyz)} measured tip {fmt(measured_xyz)} "
                              f"error_mm={fmt((np.asarray(measured_xyz) - np.asarray(planned_xyz)) * 1000)} "
                              f"joint_error_turns={fmt(measured - descend[-1])}")
                log("staged", "gripper around the object, open; returning without closing")
                cancel_event.wait(4.0 * HOLD_SECONDS)
                return
            stage = "grip"
            holding, hold_turns = close_until_contact(descend[-1].copy())
            if holding:
                holding = hold_with_force(descend[-1].copy(), grip_torque)
            if not holding:
                log("grip", "gripper closed without an object; opening and retreating")
                return
            stage = "lift"
            lift_holding = with_gripper(lift, hold_turns)
            reached = play(lift_holding, LIFT_SECONDS, False, "lift", pace="lift")
            data = tr.fresh(state)
            still = gripper_radians(cfg, np.asarray(data["pos"])) >= HOLDING_MIN_RADIANS
            log("evidence", f"after lift holding={still} gripper_current="
                            f"{float(np.asarray(data['current'])[GRIPPER_INDEX]):.2f}A")
            cancel_event.wait(HOLD_SECONDS)
            stage = "put-back"
            play(with_gripper(lift[::-1], hold_turns), LIFT_SECONDS, False, "lower", pace="lower")
            holding = False
        finally:
            if extras is not None:
                extras.wait()  # never leave the planner running on the shared solver
            if enabled:
                if holding:
                    # Stop arrived mid-grip or mid-lift: set the object back down first.
                    log("cleanup", f"holding during {stage}; lowering object before release")
                    play(with_gripper(lift[: reached + 1][::-1], hold_turns), LIFT_SECONDS, False,
                         "emergency-lower", pace="lower")
                if stage in {"grip", "lift", "put-back"}:
                    grasp_pose = descend[-1].copy()
                    grasp_pose[GRIPPER_INDEX] = open_turns
                    release(grasp_pose)
                    time.sleep(0.6)
                    log("cleanup", "gripper opened at the grasp pose")
                    reached = len(descend) - 1
                    stage = "descend"
                back = []
                if stage == "descend":
                    back.extend(with_gripper(descend[: reached + 1][::-1], open_turns))
                    reached = len(approach) - 1
                back.extend(approach[: reached + 1][::-1])
                if back:
                    play(back, RETREAT_SECONDS, False, "retreat", pace="retreat")
                command(start)
                time.sleep(0.2)
                set_torque(False)
                log("cleanup", "arm retraced to its start pose; torque off")
    log("complete", "pick attempt finished")


def build_parser():
    parser = argparse.ArgumentParser(description="Pick one tabletop object, lift, put back")
    parser.add_argument("--execute", action="store_true", help="open arm writers and move")
    parser.add_argument("--stop-at", choices=("pregrasp", "grasp"),
                        help="staged test: go only to hover (pregrasp) or around the object "
                             "with the gripper open (grasp), pause, then retrace")
    parser.add_argument("--adjust", action="store_true",
                        help="after a miss, retry from the hover pose with the pre-planned "
                             "RETRY_NUDGES_CM offsets until one grasp holds")
    parser.add_argument("--near", type=float, nargs=2, metavar=("X", "Y"),
                        help="arm-frame hint: pick the candidate closest to this point")
    parser.add_argument("--virtual", type=float, nargs=3, metavar=("X", "Y", "TOP"),
                        help="plan-only: plan against a pretend object at arm-frame X,Y "
                             "with this height, on the live table plane")
    parser.add_argument("--auto-space", action="store_true",
                        help="back the base away from the table first if it is parked too close")
    parser.add_argument("--space", action="store_true",
                        help="only do the automatic spacing, then stop")
    parser.add_argument("--place", action="store_true",
                        help="carry the object to the container found beside it and release, "
                             "instead of putting it back down")
    parser.add_argument("--box-side", choices=("left", "right"),
                        help="which side of the object the container is on")
    parser.add_argument("--fix-gripper", choices=("left", "right"),
                        help="drive that gripper back into its valid range and stop")
    parser.add_argument("--no-lean", action="store_true",
                        help="never request extra forward lean for out-of-reach objects")
    parser.add_argument("--rest", action="store_true",
                        help="lower both arms back to their hanging rest pose and stop")
    parser.add_argument("--grip-torque", type=float, default=GRIP_CLOSE_TORQUE_NM,
                        help=f"constant gripper squeeze in Nm once contact is made "
                             f"(default {GRIP_CLOSE_TORQUE_NM}, max {MAX_GRIP_TORQUE_NM})")
    parser.add_argument("--record", type=Path, metavar="DIR",
                        help="save every scanned depth frame here for offline replay "
                             "with scripts/pick_replay.py")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--serve", type=Path, metavar="DIR",
                        help="stay warm (bbos imported, IK initialised) and run the jobs "
                             "pick_lab.sh drops into DIR/jobs; exits after "
                             f"{SERVE_IDLE_SECONDS // 60} idle minutes")
    return parser


def run_job(parser, args):
    """One complete command; returns the process exit code."""

    TIMING.restart()
    if not 0.0 < args.grip_torque <= MAX_GRIP_TORQUE_NM:
        parser.error(f"--grip-torque must be between 0 and {MAX_GRIP_TORQUE_NM} Nm")
    if args.virtual and args.execute:
        parser.error("--virtual is for plan-only checks")
    scan.near = tuple(args.near) if args.near else None
    scan.box_side = {"left": 1.0, "right": -1.0}.get(args.box_side)
    scan.virtual = tuple(args.virtual) if args.virtual else None
    scan.record_dir = args.record
    scan.recorded = 0
    scan.last_box = None
    cancel_event.clear()
    raise_to_home.cache.clear()  # keyed on the live pose; stale entries only cost memory
    if args.pid_file:
        args.pid_file.write_text(f"{os.getpid()}\n")
    try:
        if args.space:
            _, Config, Reader, Type, Writer = tr._load_bbos()
            auto_space(Reader, Type, Writer)
            return 0
        if args.fix_gripper:
            _, Config, Reader, Type, Writer = tr._load_bbos()
            return 0 if fix_gripper(Config, Reader, Type, Writer, args.fix_gripper) else 1
        if args.rest:
            _, Config, Reader, Type, Writer = tr._load_bbos()
            rest_arms(Config, Reader, Type, Writer)
            return 0
        execute(plan_only=not args.execute, stop_at=args.stop_at, adjust=args.adjust,
                grip_torque=args.grip_torque, allow_lean=not args.no_lean,
                place=args.place, auto_space_enabled=args.auto_space)
        return 0
    except Exception as exc:  # noqa: BLE001 - report every failure as NOT SAFE
        log("fatal", f"NOT SAFE TO RUN: {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().splitlines():
            log("trace", line)
        return 1
    finally:
        if args.pid_file:
            try:
                args.pid_file.unlink()
            except OSError:
                pass


def serve(parser, directory):
    """Run queued jobs in this one warm process.

    A job is a file of command-line arguments in ``DIR/jobs``; its output goes
    to ``DIR/pick.log`` and ``DIR/pick.pid`` exists while it runs, exactly as
    for a one-shot run, so pick_lab.sh follows and stops both the same way.
    """

    import shlex

    jobs = directory / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    for stale in jobs.glob("*.job"):
        stale.unlink()
    (directory / "server.pid").write_text(f"{os.getpid()}\n")
    tr._load_bbos()  # pay for the import now, not when a job arrives
    console = sys.stdout
    print(f"[pick][serve] warm and waiting in {jobs}", flush=True)
    idle_since = time.monotonic()
    try:
        # SIGTERM asks the server to leave; it only does so between jobs, and a
        # job it interrupts still makes its safe return first.
        while (time.monotonic() - idle_since < SERVE_IDLE_SECONDS
               and not shutdown_event.is_set()):
            queued = sorted(jobs.glob("*.job"))
            if not queued:
                cancel_event.clear()  # a Stop with nothing running is a no-op
                time.sleep(0.05)
                continue
            words = shlex.split(queued[0].read_text())
            queued[0].unlink()
            with open(directory / "pick.log", "w", buffering=1) as output:
                sys.stdout = sys.stderr = output
                try:
                    args = parser.parse_args(
                        words + ["--pid-file", str(directory / "pick.pid")])
                    run_job(parser, args)
                except SystemExit:  # argparse rejects bad arguments by exiting
                    log("fatal", f"bad job arguments: {words}")
                finally:
                    sys.stdout, sys.stderr = console, sys.__stderr__
            idle_since = time.monotonic()
    finally:
        (directory / "server.pid").unlink(missing_ok=True)
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    def on_signal(signum, _frame):
        log("signal", f"received={signum}; requesting safe return")
        cancel_event.set()
        if signum == signal.SIGTERM:
            shutdown_event.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    if args.serve:
        return serve(parser, args.serve)
    return run_job(parser, args)


if __name__ == "__main__":
    raise SystemExit(main())
