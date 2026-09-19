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
GRASP_HEIGHT_FRACTION = 0.5      # aim for the object's mid-height (its thickest part)
GRASP_BELOW_TOP_METRES = 0.02    # but never above this much below the top
MIN_GRASP_ABOVE_TABLE = 0.065
PREGRASP_BACKOFF_METRES = 0.10
# The IK end-effector point is at the fingertips; the jaws pivot 9 cm behind
# it and the pads meet ~4 cm behind it (URDF). Push the tips this far past
# the object's axis so the pad centre, not the tips, lands on the object.
GRASP_TIP_PAST_AXIS_METRES = 0.04
PREGRASP_RAISE_METRES = 0.03
LIFT_METRES = 0.10
GRIPPER_OPEN_RADIANS = 0.80
GRIPPER_CLOSED_RADIANS = -0.15
GRIPPER_HOLD_SQUEEZE_RADIANS = 0.06
# Force grip: the arm daemon relieves current on a stalled position target, so a
# held grasp fades. Teleop's full-trigger grasp is 0.70 Nm (~3.4 A at kt=0.204)
# and its comments cap a held grasp near 2.0 Nm.
GRIP_CLOSE_TORQUE_NM = 0.70
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
RAISE_EDGE_MARGIN_METRES = 0.0
RAISE_CLEARANCE_METRES = 0.03
# Measured on bracketbot-184: grasps solve to 2-6 mm out to ~0.45 m from the
# shoulder and fail beyond ~0.52 m, where tipping the ~14 cm fingers down
# costs horizontal reach.
IK_CONVERGED_METRES = 0.008
MAX_REACH_METRES = 0.48

cancel_event = threading.Event()


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


def startup_raise(cfg, path, observation):
    """Lift the arm to home along the robot's own calibrated startup waypoints.

    These are the vendor waypoints quest_teleop homes through: they rise close
    to the body (behind the table edge) before reaching out, which Cartesian
    IK from a hanging pose cannot reliably reproduce.
    """

    waypoints = [np.asarray(w, dtype=np.float64) for w in cfg.startup_waypoints]
    waypoints.append(np.asarray(cfg.home, dtype=np.float64))
    for number, waypoint in enumerate(waypoints, start=1):
        blend_to(cfg, path, waypoint, observation, label=f"startup waypoint {number}")
    log("plan", f"raised to home along {len(waypoints)} calibrated startup waypoints")


def plan_pick(cfg, start, side, item, plane, pitch, yaw_fraction=1.0):
    """Full validated motor path plus the indices where each phase ends."""

    observation = {
        # Conservative: the plane rises with forward distance, so use the
        # surface height just beyond the object for every clearance check.
        "height": plane.height_at(item.center[0] + 0.03, item.center[1]),
        "near_edge": plane.near_edge,
    }
    pregrasp, grasp, lift, grasp_quaternion = grasp_waypoints(
        item, side, pitch, plane.height_at, yaw_fraction)
    log(
        "plan",
        f"side={side} pitch={pitch:.0f}deg yaw_fraction={yaw_fraction:.1f} pregrasp={fmt(pregrasp)} grasp={fmt(grasp)} "
        f"lift={fmt(lift)} clearance_height={observation['height']:.3f} "
        f"near_edge={observation['near_edge']:.3f}",
    )
    live_urdf = np.asarray(cfg.q2urdf(start.copy()), dtype=np.float64)
    cfg.ik.init()
    cfg.ik.reset(list(live_urdf[:7]))
    live_position, live_quaternion = (np.asarray(v, dtype=np.float64) for v in cfg.ik.fk(list(live_urdf[:7])))
    home = np.asarray(cfg.home, dtype=np.float64).copy()
    home[GRIPPER_INDEX] = start[GRIPPER_INDEX]
    home_position, home_quaternion = (
        np.asarray(v, dtype=np.float64)
        for v in cfg.ik.fk(list(np.asarray(cfg.q2urdf(home.copy()), dtype=np.float64)[:7]))
    )
    lateral = SHOULDER_LATERAL_METRES if side == "left" else -SHOULDER_LATERAL_METRES
    escape = np.array([min(0.13, plane.near_edge - 0.05), lateral, observation["height"] + 0.08])

    path = [start.copy()]
    marks = {}
    startup_failure = None
    try:
        startup_raise(cfg, path, observation)
        marks["raise-behind-table"] = marks["move-home"] = len(path) - 1
        home_reached = np.asarray(cfg.q2urdf(path[-1].copy()), dtype=np.float64)
        cfg.ik.reset(list(home_reached[:7]))
        live_position, live_quaternion = (
            np.asarray(v, dtype=np.float64) for v in cfg.ik.fk(list(home_reached[:7])))
    except RuntimeError as exc:
        startup_failure = str(exc)
        log("plan", f"startup-waypoint raise unavailable ({exc}); using Cartesian escape")
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
            joint_space_raise(cfg, attempt, target, options, observation,
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
    edges = [("approach", 0, marks["pregrasp"]), ("descend", marks["pregrasp"], marks["grasp"]),
             ("lift", marks["grasp"], marks["lift"])]
    for name, first, last in edges:
        phases[name] = np.asarray(densify(list(path[first:last + 1])), dtype=np.float64)
    return phases


def scan(Reader, frames=5, timeout=8.0):
    """Fit the table and select a graspable object on fresh, consistent frames."""

    picks = []
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
            arm = points_to_arm(np.asarray(reader.data["points"])[:count].astype(np.float64))
            try:
                plane = fit_table_plane(arm)
                objects = find_objects(arm, plane)
            except RuntimeError as exc:
                log("scan", f"frame rejected: {exc}")
                continue
            item = select_graspable(objects, near=scan.near, max_reach=1.0)
            if scan.near is not None and (
                item is None
                or math.hypot(item.center[0] - scan.near[0], item.center[1] - scan.near[1]) > 0.08
            ):
                item = object_near(arm, plane, scan.near)
                if item is not None:
                    log("scan", "hinted object was merged with a neighbour; measured it locally")
            if item is not None and scan.virtual is not None:
                x, y, top = scan.virtual
                item = TableObject((x, y, plane.height_at(x, y)), top, 0.066, 0.066, 0.0, 0)
            log(
                "scan",
                f"plane tilt={plane.tilt_degrees:.1f}deg inliers={plane.inliers} "
                f"near_edge={plane.near_edge:.3f} objects={len(objects)} "
                f"selected={None if item is None else fmt(item.center)} "
                f"top={0 if item is None else item.top:.3f} "
                f"width={0 if item is None else item.width:.3f}",
            )
            if item is not None:
                picks.append((plane, item))
    if len(picks) < frames:
        raise RuntimeError(f"no graspable object seen on {frames} fresh depth frames")
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
    log("scan", f"median of {len(picks)} frames centre={fmt(item.center)} top={item.top:.3f} "
                f"spread={spread:.3f}m")
    return plane, item


scan.near = None
scan.virtual = None


def check_reach(item):
    shoulder_y = SHOULDER_LATERAL_METRES if item.center[1] >= 0 else -SHOULDER_LATERAL_METRES
    reach = math.hypot(item.center[0], item.center[1] - shoulder_y)
    log("reach", f"object {reach:.3f} m from the {'left' if shoulder_y > 0 else 'right'} "
                 f"shoulder (limit {MAX_REACH_METRES:.2f})")
    if reach > MAX_REACH_METRES:
        raise RuntimeError(
            f"object is out of reach: {reach:.2f} m from the shoulder; move it to within "
            f"{MAX_REACH_METRES:.2f} m (about {item.center[0] - (reach - MAX_REACH_METRES) - 0.03:.2f} m forward)"
        )


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


def execute(plan_only=True, pid_file=None, stop_at=None, adjust=False,
            grip_torque=GRIP_CLOSE_TORQUE_NM):
    bbos, Config, Reader, Type, Writer = tr._load_bbos()
    with tr.nonsuppressing(Reader("imu.orientation", keeptime=False)) as imu:
        rpy = np.asarray(tr.fresh(imu)["rpy"], dtype=np.float64)
    log("preflight", f"imu_rpy={fmt(rpy)}")
    if abs(rpy[0]) >= UPRIGHT_DEGREES or abs(rpy[1]) >= UPRIGHT_DEGREES:
        raise RuntimeError("robot is not upright")

    plane, item = scan(Reader)
    check_reach(item)
    side = "left" if item.center[1] >= 0.0 else "right"
    cfg = Config(f"arm_{side}")
    with tr.nonsuppressing(Reader(f"arm_{side}.state", keeptime=False)) as state_reader:
        start = np.asarray(tr.fresh(state_reader)["pos"], dtype=np.float64).copy()
    log("state", f"side={side} start={fmt(start)} gripper={gripper_radians(cfg, start):.3f}rad")

    failures = []
    options = [(pitch, fraction) for fraction in (1.0, 0.5, 0.0) for pitch in GRASP_PITCHES_DEGREES]
    for pitch, yaw_fraction in options:
        try:
            path, marks, observation = plan_pick(cfg, start, side, item, plane, pitch, yaw_fraction)
            phases = split_dense(path, marks)
            low, high = tr.calibration_limits(bbos, side, logger=log)
            for name, phase in phases.items():
                tr.validate_calibration(phase, low, high, f"{side}:{name}", logger=log)
                tr.validate_playback_clearance(phase, cfg, observation, f"{side}:{name}", logger=log)
            break
        except RuntimeError as exc:
            failures.append(f"pitch {pitch:.0f} yaw x{yaw_fraction:.1f}: {exc}")
            log("plan", f"pitch={pitch:.0f}deg yaw_fraction={yaw_fraction:.1f} rejected: {exc}")
    else:
        raise RuntimeError("no validated grasp: " + " | ".join(failures))
    log("plan", f"accepted pitch={pitch:.0f}deg yaw_fraction={yaw_fraction:.1f} phases=" + ",".join(
        f"{name}:{len(phase)}" for name, phase in phases.items()))
    if plan_only:
        log("complete", "plan-only succeeded; no arm writers were opened")
        return

    base_pregrasp, base_grasp, base_lift, grasp_quaternion = grasp_waypoints(
        item, side, pitch, plane.height_at, yaw_fraction)

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
                    cfg, side, phases["approach"][-1], base_grasp + offset, base_lift + offset,
                    grasp_quaternion, observation, low, high)
            except RuntimeError as exc:
                log("retry", f"nudge_cm={fmt(offset * 100)} rejected: {exc}")
                continue
            attempts.append((offset, descend, lifted))
        log("retry", f"{len(attempts)} validated attempts planned before moving")

    run_motion(Config, Reader, Type, Writer, cfg, side, start, phases, stop_at, attempts,
               grip_torque)


def run_motion(Config, Reader, Type, Writer, cfg, side, start, phases, stop_at=None,
               attempts=None, grip_torque=GRIP_CLOSE_TORQUE_NM):
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
            grip["tau"] = 0.0
            set_torque(True, force_grip=False)
            command(pose)
            time.sleep(0.4)

        def play(poses, seconds, cancellable, stage):
            began = time.monotonic()
            index = 0
            while True:
                alpha = min((time.monotonic() - began) / seconds, 1.0)
                index = min(int(tr.smoothstep(alpha) * (len(poses) - 1)), len(poses) - 1)
                command(poses[index])
                if alpha >= 1.0 or (cancellable and cancel_event.is_set()):
                    log("motion", f"stage={stage} end index={index}/{len(poses) - 1} "
                                  f"cancelled={cancel_event.is_set()}")
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
            reached = play(approach, APPROACH_SECONDS, True, "approach")
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
                    reached = play(descend, DESCEND_SECONDS, True, "descend")
                    if cancel_event.is_set():
                        break
                    stage = "grip"
                    holding, hold_turns = close_until_contact(descend[-1].copy())
                    if holding:
                        holding = hold_with_force(descend[-1].copy(), grip_torque)
                    result = "PICKED" if holding else "MISS"
                    if holding:
                        stage = "lift"
                        reached = play(with_gripper(lift, hold_turns), LIFT_SECONDS, False, "lift")
                        data = tr.fresh(state)
                        still = gripper_radians(cfg, np.asarray(data["pos"])) >= HOLDING_MIN_RADIANS
                        result = "PICKED" if still else "SLIPPED"
                        log("evidence", f"attempt={number} after lift holding={still} "
                                        f"current={float(np.asarray(data['current'])[GRIPPER_INDEX]):.2f}A")
                        cancel_event.wait(HOLD_SECONDS)
                        stage = "put-back"
                        play(with_gripper(lift[::-1], hold_turns), LIFT_SECONDS, False, "lower")
                        holding = False
                    open_pose = descend[-1].copy()
                    open_pose[GRIPPER_INDEX] = open_turns
                    release(open_pose)
                    command(open_pose)
                    time.sleep(0.6)
                    play(with_gripper(descend[::-1], open_turns), DESCEND_SECONDS, False, "ascend")
                    stage = "approach"
                    reached = len(approach) - 1
                    log("retry", f"attempt={number} result={result}")
                    if result == "PICKED":
                        break
                return
            stage = "descend"
            reached = play(descend, DESCEND_SECONDS, True, "descend")
            if cancel_event.is_set():
                return
            if stop_at == "grasp":
                log("staged", "gripper around the object, open; returning without closing")
                cancel_event.wait(2.0 * HOLD_SECONDS)
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
            reached = play(lift_holding, LIFT_SECONDS, False, "lift")
            data = tr.fresh(state)
            still = gripper_radians(cfg, np.asarray(data["pos"])) >= HOLDING_MIN_RADIANS
            log("evidence", f"after lift holding={still} gripper_current="
                            f"{float(np.asarray(data['current'])[GRIPPER_INDEX]):.2f}A")
            cancel_event.wait(HOLD_SECONDS)
            stage = "put-back"
            play(with_gripper(lift[::-1], hold_turns), LIFT_SECONDS, False, "lower")
            holding = False
        finally:
            if enabled:
                if holding:
                    # Stop arrived mid-grip or mid-lift: set the object back down first.
                    log("cleanup", f"holding during {stage}; lowering object before release")
                    play(with_gripper(lift[: reached + 1][::-1], hold_turns), LIFT_SECONDS, False,
                         "emergency-lower")
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
                    play(back, RETREAT_SECONDS, False, "retreat")
                command(start)
                time.sleep(0.2)
                set_torque(False)
                log("cleanup", "arm retraced to its start pose; torque off")
    log("complete", "pick attempt finished")


def main():
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
    parser.add_argument("--grip-torque", type=float, default=GRIP_CLOSE_TORQUE_NM,
                        help=f"constant gripper squeeze in Nm once contact is made "
                             f"(default {GRIP_CLOSE_TORQUE_NM}, max {MAX_GRIP_TORQUE_NM})")
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.grip_torque <= MAX_GRIP_TORQUE_NM:
        parser.error(f"--grip-torque must be between 0 and {MAX_GRIP_TORQUE_NM} Nm")
    if args.virtual and args.execute:
        parser.error("--virtual is for plan-only checks")
    scan.near = tuple(args.near) if args.near else None
    scan.virtual = tuple(args.virtual) if args.virtual else None

    def on_signal(signum, _frame):
        log("signal", f"received={signum}; requesting safe return")
        cancel_event.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    if args.pid_file:
        args.pid_file.write_text(f"{os.getpid()}\n")
    try:
        execute(plan_only=not args.execute, stop_at=args.stop_at, adjust=args.adjust,
                grip_torque=args.grip_torque)
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


if __name__ == "__main__":
    raise SystemExit(main())
