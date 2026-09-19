"""Place both BracketBot arms on a depth-detected tabletop. Runs on the robot.

The action deliberately has no fixed-height fallback. It requires fresh
``camera.points`` frames, finds a broad horizontal surface in the bounded arm
workspace, plans every end-effector waypoint through the robot's IK solver,
and validates every motor target against this robot's calibrated limits before
enabling torque. The arms hold the measured tabletop pose until interrupted,
then retrace the path and become limp.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import numpy as np


SIDES = ("left", "right")
DOF = 8
TICK_SECONDS = 0.015
DEPTH_TIMEOUT_SECONDS = 6.0
DEPTH_SAMPLES = 3
UPRIGHT_DEGREES = 25.0
TABLE_Z_RANGE = (0.52, 1.02)
TABLE_X_RANGE = (0.18, 0.82)
TABLE_Y_LIMIT = 0.62
PLANE_BIN_METRES = 0.015
PLANE_HALF_WIDTH_METRES = 0.012
MIN_PLANE_POINTS = 120
MIN_HAND_SUPPORT_POINTS = 10
HAND_LATERAL_METRES = 0.20
HAND_CLEARANCE_METRES = 0.055
APPROACH_CLEARANCE_METRES = 0.16
MAX_TABLE_VARIATION_METRES = 0.04
IK_SAMPLES_PER_SEGMENT = 96
MAX_ROTARY_IK_STEP_TURNS = 0.08
MAX_LIFT_IK_STEP_TURNS = 0.25
LIFT_METRES_PER_TURN = 0.0465
MAX_TOTAL_MOVE_TURNS = 1.10
CALIBRATION_MARGIN = 0.02
MOVE_SECONDS = 5.0
RETURN_SECONDS = 4.0

cancel_event = threading.Event()


def table_log(stage, message):
    """Emit one immediately visible, grep-friendly diagnostic line."""
    print(f"[table][{stage}] {message}", flush=True)


def format_values(values, decimals=3):
    return np.array2string(
        np.asarray(values),
        precision=decimals,
        suppress_small=True,
        separator=",",
        max_line_width=240,
    )


@contextmanager
def nonsuppressing(manager):
    """Use a BBOS context manager without allowing it to hide runner errors."""
    value = manager.__enter__()
    try:
        yield value
    except BaseException:
        # Some BBOS managers return True for exceptions, which would hide the
        # failure. Close normally here, then re-raise it ourselves.
        manager.__exit__(None, None, None)
        raise
    else:
        manager.__exit__(None, None, None)


def _load_bbos():
    """Import BBOS only on the robot so perception helpers stay unit-testable."""
    bbos_root = Path.home() / "bbos"
    venv_packages = bbos_root / ".venv/lib/python3.10/site-packages"
    for path in (bbos_root, venv_packages):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    import bbos
    from bbos import Config, Reader, Type, Writer

    return bbos, Config, Reader, Type, Writer


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def quaternion_slerp(start, end, alpha):
    q0 = np.asarray(start, dtype=np.float64)
    q1 = np.asarray(end, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    return (
        math.sin((1.0 - alpha) * angle) * q0
        + math.sin(alpha * angle) * q1
    ) / math.sin(angle)


def quaternion_from_z(direction):
    """Return xyzw rotation whose gripper-forward Z axis faces ``direction``."""
    target = np.asarray(direction, dtype=np.float64)
    target /= np.linalg.norm(target)
    source = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot < -1.0 + 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    cross = np.cross(source, target)
    result = np.array([*cross, 1.0 + dot])
    return result / np.linalg.norm(result)


def detect_table_plane(points, logger=None):
    """Find a broad, two-hand tabletop in a base-frame XYZ point cloud.

    Returns a small observation dictionary, or raises ``RuntimeError`` when a
    surface cannot be distinguished safely. A height histogram is intentionally
    used instead of an unconstrained plane fit: only nearly-horizontal surfaces
    inside the known reachable workspace are eligible.
    """
    emit = logger or (lambda _stage, _message: None)
    raw = np.asarray(points)
    camera_cloud = np.asarray(raw, dtype=np.float64).reshape(-1, 3)
    finite_camera = camera_cloud[np.isfinite(camera_cloud).all(axis=1)]
    emit(
        "depth",
        f"cloud shape={raw.shape} dtype={raw.dtype} total={len(camera_cloud)} "
        f"finite={len(finite_camera)} frame=camera.points(lateral,forward,height)",
    )
    if len(finite_camera):
        emit(
            "depth",
            "camera_xyz min=" + format_values(np.min(finite_camera, axis=0))
            + " p50=" + format_values(np.median(finite_camera, axis=0))
            + " max=" + format_values(np.max(finite_camera, axis=0)),
        )
    # camera.points uses lateral/forward/height, while each arm IK target uses
    # forward/left/height. The camera's positive lateral axis points right.
    finite = np.column_stack(
        (finite_camera[:, 1], -finite_camera[:, 0], finite_camera[:, 2])
    )
    if len(finite):
        emit(
            "transform",
            "arm_xyz=(camera_y,-camera_x,camera_z) min="
            + format_values(np.min(finite, axis=0))
            + " p50=" + format_values(np.median(finite, axis=0))
            + " max=" + format_values(np.max(finite, axis=0)),
        )
    usable = finite[
        (finite[:, 0] >= TABLE_X_RANGE[0])
        & (finite[:, 0] <= TABLE_X_RANGE[1])
        & (np.abs(finite[:, 1]) <= TABLE_Y_LIMIT)
        & (finite[:, 2] >= TABLE_Z_RANGE[0])
        & (finite[:, 2] <= TABLE_Z_RANGE[1])
    ]
    emit(
        "filter",
        f"workspace x={TABLE_X_RANGE} |y|<={TABLE_Y_LIMIT:.2f} "
        f"z={TABLE_Z_RANGE}: kept={len(usable)}/{len(finite)} "
        f"required={MIN_PLANE_POINTS}",
    )
    if len(usable) < MIN_PLANE_POINTS:
        raise RuntimeError(
            f"not enough depth points in the reachable table area "
            f"({len(usable)} < {MIN_PLANE_POINTS})"
        )

    bin_ids = np.floor(usable[:, 2] / PLANE_BIN_METRES).astype(np.int64)
    unique, counts = np.unique(bin_ids, return_counts=True)
    ranked = unique[np.argsort(counts)[::-1]]
    count_by_bin = dict(zip(unique.tolist(), counts.tolist()))
    emit(
        "bins",
        "top=" + ", ".join(
            f"z~{(float(bin_id) + 0.5) * PLANE_BIN_METRES:.3f}:{count_by_bin[int(bin_id)]}"
            for bin_id in ranked[:12]
        ),
    )
    candidates = []
    rejection_counts = {
        "few-points": 0,
        "small-span": 0,
        "shallow-depth": 0,
        "hand-support": 0,
    }
    for rank, bin_id in enumerate(ranked[:12], start=1):
        center = (float(bin_id) + 0.5) * PLANE_BIN_METRES
        plane = usable[np.abs(usable[:, 2] - center) <= PLANE_HALF_WIDTH_METRES]
        if len(plane) < MIN_PLANE_POINTS:
            rejection_counts["few-points"] += 1
            emit(
                "candidate",
                f"#{rank} z={center:.3f} points={len(plane)} reject=few-points "
                f"required={MIN_PLANE_POINTS}",
            )
            continue
        x10, x90 = np.quantile(plane[:, 0], (0.10, 0.90))
        y10, y90 = np.quantile(plane[:, 1], (0.10, 0.90))
        if x90 - x10 < 0.22 or y90 - y10 < 0.42:
            rejection_counts["small-span"] += 1
            emit(
                "candidate",
                f"#{rank} z={center:.3f} points={len(plane)} "
                f"x10..90={x10:.3f}..{x90:.3f} span={x90-x10:.3f} "
                f"y10..90={y10:.3f}..{y90:.3f} span={y90-y10:.3f} "
                "reject=small-span required_x=0.220 required_y=0.420",
            )
            continue

        near_edge = float(np.quantile(plane[:, 0], 0.05))
        far_edge = float(np.quantile(plane[:, 0], 0.95))
        hand_x = float(np.clip(near_edge + 0.16, 0.30, 0.48))
        if hand_x > far_edge - 0.05:
            rejection_counts["shallow-depth"] += 1
            emit(
                "candidate",
                f"#{rank} z={center:.3f} points={len(plane)} edge={near_edge:.3f}..{far_edge:.3f} "
                f"hand_x={hand_x:.3f} reject=shallow-depth require_far>={hand_x+0.05:.3f}",
            )
            continue

        hand_support = []
        for hand_y in (HAND_LATERAL_METRES, -HAND_LATERAL_METRES):
            supported = plane[
                (np.abs(plane[:, 0] - hand_x) <= 0.13)
                & (np.abs(plane[:, 1] - hand_y) <= 0.12)
            ]
            hand_support.append(len(supported))
        if min(hand_support) < MIN_HAND_SUPPORT_POINTS:
            rejection_counts["hand-support"] += 1
            emit(
                "candidate",
                f"#{rank} z={center:.3f} points={len(plane)} "
                f"x_span={x90-x10:.3f} y_span={y90-y10:.3f} "
                f"edge={near_edge:.3f}..{far_edge:.3f} hand_x={hand_x:.3f} "
                f"support_left={hand_support[0]} support_right={hand_support[1]} "
                f"reject=hand-support required_each={MIN_HAND_SUPPORT_POINTS}",
            )
            continue

        height = float(np.median(plane[:, 2]))
        flatness = float(np.median(np.abs(plane[:, 2] - height)))
        candidates.append(
            (
                min(hand_support),
                len(plane),
                {
                    "height": height,
                    "hand_x": hand_x,
                    "points": int(len(plane)),
                    "hand_support": tuple(int(value) for value in hand_support),
                    "flatness": flatness,
                },
            )
        )
        emit(
            "candidate",
            f"#{rank} z={height:.3f} points={len(plane)} "
            f"x_span={x90-x10:.3f} y_span={y90-y10:.3f} "
            f"edge={near_edge:.3f}..{far_edge:.3f} hand_x={hand_x:.3f} "
            f"support_left={hand_support[0]} support_right={hand_support[1]} "
            f"flatness={flatness*1000:.1f}mm accept",
        )

    if not candidates:
        summary = ",".join(f"{key}={value}" for key, value in rejection_counts.items())
        raise RuntimeError(
            "no broad horizontal surface supports both hand positions "
            f"(top_bins_checked={min(len(ranked), 12)}; {summary})"
        )
    selected = max(candidates, key=lambda item: (item[0], item[1]))[2]
    emit(
        "select",
        f"height={selected['height']:.3f} hand_x={selected['hand_x']:.3f} "
        f"points={selected['points']} support={selected['hand_support']} "
        f"flatness={selected['flatness']*1000:.1f}mm",
    )
    return selected


def fresh(reader, timeout=2.0):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started >= timeout:
            raise RuntimeError("no fresh robot state is available")
        time.sleep(0.005)
    return reader.data


def observe_table(reader, logger=table_log):
    """Require three consistent, newly published table observations."""
    deadline = time.monotonic() + DEPTH_TIMEOUT_SECONDS
    observations = []
    last_timestamp = None
    last_error = "camera.points did not publish"
    frames_seen = 0
    rejected = 0
    last_wait_log = 0.0
    logger(
        "scan",
        f"deadline={DEPTH_TIMEOUT_SECONDS:.1f}s required_good_frames={DEPTH_SAMPLES}",
    )
    while time.monotonic() < deadline and len(observations) < DEPTH_SAMPLES:
        if cancel_event.is_set():
            raise RuntimeError("table placement cancelled")
        if not reader.ready():
            now = time.monotonic()
            if now - last_wait_log >= 1.0:
                logger(
                    "scan",
                    f"waiting camera.points elapsed={DEPTH_TIMEOUT_SECONDS-(deadline-now):.1f}s "
                    f"frames={frames_seen} accepted={len(observations)} rejected={rejected}",
                )
                last_wait_log = now
            time.sleep(0.02)
            continue
        data = reader.data
        try:
            timestamp = float(np.asarray(data["timestamp"]).reshape(-1)[0])
        except (KeyError, TypeError, ValueError, IndexError):
            last_error = "camera.points frames have no timestamp"
            logger("frame", f"reject={last_error}")
            time.sleep(0.02)
            continue
        if timestamp == last_timestamp:
            time.sleep(0.02)
            continue
        last_timestamp = timestamp
        count = int(data["num_points"])
        frames_seen += 1
        points = np.asarray(data["points"])
        logger(
            "frame",
            f"#{frames_seen} timestamp={timestamp:.6f} declared_points={count} "
            f"buffer_shape={points.shape} buffer_dtype={points.dtype}",
        )
        try:
            verbose = frames_seen <= 3 or frames_seen % 10 == 0
            observation = detect_table_plane(
                points[:count], logger=logger if verbose else None
            )
            observations.append(observation)
            logger(
                "frame",
                f"#{frames_seen} accepted={len(observations)}/{DEPTH_SAMPLES} "
                f"z={observation['height']:.3f} x={observation['hand_x']:.3f}",
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            last_error = str(exc)
            rejected += 1
            logger("frame", f"#{frames_seen} rejected: {last_error}")
        time.sleep(0.04)

    if len(observations) < DEPTH_SAMPLES:
        raise RuntimeError(
            "adaptive table placement needs fresh camera.points depth data: "
            f"frames={frames_seen} accepted={len(observations)} rejected={rejected}; "
            f"last={last_error}"
        )
    heights = np.asarray([item["height"] for item in observations])
    logger(
        "stability",
        f"heights={format_values(heights)} spread={float(np.ptp(heights)):.3f}m "
        f"limit={MAX_TABLE_VARIATION_METRES:.3f}m",
    )
    if float(np.ptp(heights)) > MAX_TABLE_VARIATION_METRES:
        raise RuntimeError("table height was not stable across fresh depth frames")
    return {
        "height": float(np.median(heights)),
        "hand_x": float(np.median([item["hand_x"] for item in observations])),
        "points": min(item["points"] for item in observations),
        "flatness": max(item["flatness"] for item in observations),
    }


def _append_segment(cfg, current_urdf, path, start_position, start_quaternion,
                    end_position, end_quaternion, segment, logger=table_log,
                    samples=IK_SAMPLES_PER_SEGMENT):
    logger(
        "ik",
        f"segment={segment} samples={samples} start_xyz={format_values(start_position)} "
        f"goal_xyz={format_values(end_position)}",
    )
    for index in range(1, samples + 1):
        alpha = index / samples
        position = (1.0 - alpha) * start_position + alpha * end_position
        orientation = quaternion_slerp(start_quaternion, end_quaternion, alpha)
        nominal = list(np.asarray(current_urdf[:7], dtype=np.float64))
        if hasattr(cfg.ik, "solve_with_nominal"):
            solution = cfg.ik.solve_with_nominal(
                position.tolist(), orientation.tolist(), nominal
            )
        else:
            solution = cfg.ik.solve(position.tolist(), orientation.tolist())
        if solution is None or len(solution) < 7:
            logger(
                "ik",
                f"segment={segment} waypoint={index}/{samples} reject=no-solution "
                f"xyz={format_values(position)} quat={format_values(orientation)}",
            )
            raise RuntimeError(
                f"IK missed {segment} waypoint {index}/{samples} at xyz={format_values(position)}"
            )
        solved = current_urdf.copy()
        solved[:7] = np.asarray(solution[:7], dtype=np.float64)
        pose = np.asarray(cfg.urdf2q(solved), dtype=np.float64)
        pose[7] = path[0][7]
        delta = np.abs(pose - path[-1])
        step = float(np.max(delta))
        lift_step = float(delta[0])
        rotary_step = float(np.max(delta[1:7]))
        if not np.isfinite(pose).all():
            logger(
                "ik",
                f"segment={segment} waypoint={index}/{samples} reject=non-finite "
                f"pose={format_values(pose)}",
            )
            raise RuntimeError(
                f"IK returned non-finite values in {segment} waypoint {index}/{samples}"
            )
        if (
            lift_step > MAX_LIFT_IK_STEP_TURNS
            or rotary_step > MAX_ROTARY_IK_STEP_TURNS
        ):
            joint = int(np.argmax(delta))
            logger(
                "ik",
                f"segment={segment} waypoint={index}/{samples} reject=branch-jump "
                f"joint={joint} max_step={step:.4f} lift_step={lift_step:.4f} "
                f"lift_mm={lift_step * LIFT_METRES_PER_TURN * 1000:.1f} "
                f"lift_limit={MAX_LIFT_IK_STEP_TURNS:.4f} rotary_step={rotary_step:.4f} "
                f"rotary_limit={MAX_ROTARY_IK_STEP_TURNS:.4f} "
                f"previous={format_values(path[-1])} pose={format_values(pose)} "
                f"delta={format_values(delta)}",
            )
            raise RuntimeError(
                f"IK branch jump in {segment} waypoint {index}/{samples}: "
                f"joint={joint} lift={lift_step:.3f} turns "
                f"rotary={rotary_step:.3f} turns"
            )
        path.append(pose)
        if index == 1 or index % 12 == 0 or index == samples:
            logger(
                "ik",
                f"segment={segment} waypoint={index}/{samples} max_step={step:.4f} "
                f"lift_step={lift_step:.4f} ({lift_step * LIFT_METRES_PER_TURN * 1000:.1f}mm) "
                f"rotary_step={rotary_step:.4f} "
                f"motor={format_values(pose)}",
            )
    logger("ik", f"segment={segment} complete path_points={len(path)}")
    return end_position, end_quaternion


def plan_table_path(cfg, start, side, observation, logger=table_log):
    current_urdf = np.asarray(cfg.q2urdf(start.copy()), dtype=np.float64)
    cfg.ik.init()
    cfg.ik.reset(list(current_urdf[:7]))
    start_position, start_quaternion = cfg.ik.fk(list(current_urdf[:7]))
    position = np.asarray(start_position, dtype=np.float64)
    orientation = np.asarray(start_quaternion, dtype=np.float64)
    lateral = HAND_LATERAL_METRES if side == "left" else -HAND_LATERAL_METRES
    rest_z = observation["height"] + HAND_CLEARANCE_METRES
    # Rotate toward a shallow, table-facing wrist pose during the clearance
    # lift. Preserving the hanging pose forces wrist joint 6 beyond this
    # robot's calibrated range once the hand reaches tabletop height.
    rest_orientation = quaternion_from_z([1.0, 0.0, 0.36])
    orient_height = min(0.70, rest_z - 0.10)
    orient = np.array([
        float(np.clip(position[0], 0.08, 0.14)),
        lateral,
        max(float(position[2]), orient_height),
    ])
    clear = np.array([
        float(np.clip(position[0], 0.25, 0.34)),
        lateral,
        max(float(position[2]), rest_z + APPROACH_CLEARANCE_METRES),
    ])
    above = np.array([
        observation["hand_x"],
        lateral,
        rest_z + APPROACH_CLEARANCE_METRES,
    ])
    rest = np.array([observation["hand_x"], lateral, rest_z])

    logger(
        "plan",
        f"side={side} start_motor={format_values(start)} "
        f"start_urdf={format_values(current_urdf)} start_xyz={format_values(position)} "
        f"start_quat={format_values(orientation)}",
    )
    logger(
        "plan",
        f"side={side} table_z={observation['height']:.3f} rest_z={rest_z:.3f} "
        f"orient={format_values(orient)} clear={format_values(clear)} "
        f"above={format_values(above)} rest={format_values(rest)} "
        f"orientation=table-facing quat={format_values(rest_orientation)}",
    )

    path = [np.asarray(start, dtype=np.float64).copy()]
    for segment, destination, destination_orientation in (
        ("raise-for-orientation", orient, orientation),
        ("orient-in-clear-space", orient, rest_orientation),
        ("raise-clear", clear, rest_orientation),
        ("move-above", above, rest_orientation),
        ("lower-rest", rest, rest_orientation),
    ):
        position, orientation = _append_segment(
            cfg,
            current_urdf,
            path,
            position,
            orientation,
            destination,
            destination_orientation,
            f"{side}:{segment}",
            logger=logger,
        )
    total = float(np.max(np.abs(np.asarray(path) - start)))
    joint = int(np.argmax(np.max(np.abs(np.asarray(path) - start), axis=0)))
    logger(
        "plan",
        f"side={side} path_points={len(path)} largest_move={total:.4f} "
        f"joint={joint} limit={MAX_TOTAL_MOVE_TURNS:.3f}",
    )
    if total > MAX_TOTAL_MOVE_TURNS:
        raise RuntimeError(f"{side} arm needs a {total:.3f}-turn move, above the safe bound")
    return np.asarray(path, dtype=np.float64)


def calibration_limits(bbos_module, side, logger=table_log):
    path = (
        Path(bbos_module.__file__).parent
        / "daemons"
        / f"arm_{side}"
        / "ranges.calibration.json"
    )
    try:
        values = json.loads(path.read_text())
        low = np.minimum(values["cal_min"], values["cal_max"]).astype(np.float64)
        high = np.maximum(values["cal_min"], values["cal_max"]).astype(np.float64)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot read {side} arm calibration limits") from exc
    logger(
        "calibration",
        f"side={side} file={path} low={format_values(low)} high={format_values(high)}",
    )
    return low, high


def validate_calibration(path, low, high, side, logger=table_log):
    margin = CALIBRATION_MARGIN * (high - low)
    path_low = np.min(path, axis=0)
    path_high = np.max(path, axis=0)
    below = path_low < low - margin
    above = path_high > high + margin
    logger(
        "calibration",
        f"side={side} path_low={format_values(path_low)} path_high={format_values(path_high)} "
        f"margin={format_values(margin)}",
    )
    if np.any(below) or np.any(above):
        joints = np.flatnonzero(below | above).tolist()
        logger(
            "calibration",
            f"side={side} reject joints={joints} below={np.flatnonzero(below).tolist()} "
            f"above={np.flatnonzero(above).tolist()}",
        )
        raise RuntimeError(
            f"{side} table path leaves calibrated joint limits at joints={joints}"
        )
    logger("calibration", f"side={side} accepted all {path.shape[0]} path poses")


def write_pid_file(path):
    if path is not None:
        path.write_text(f"{os.getpid()}\n")


def remove_pid_file(path):
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def execute(pid_file=None, scan_only=False, plan_only=False):
    table_log(
        "boot",
        f"pid={os.getpid()} pid_file={pid_file} scan_only={scan_only} plan_only={plan_only} "
        f"tick={TICK_SECONDS:.3f}s table_x={TABLE_X_RANGE} table_z={TABLE_Z_RANGE} "
        f"hand_y=±{HAND_LATERAL_METRES:.3f} hand_clearance={HAND_CLEARANCE_METRES:.3f}",
    )
    bbos, Config, Reader, Type, Writer = _load_bbos()
    table_log("boot", f"bbos={Path(bbos.__file__).resolve()}")
    with ExitStack() as stack:
        imu = stack.enter_context(
            nonsuppressing(Reader("imu.orientation", keeptime=False))
        )
        points = stack.enter_context(
            nonsuppressing(Reader("camera.points", keeptime=False))
        )
        table_log("readers", "opened imu.orientation and camera.points")
        rpy = np.asarray(fresh(imu)["rpy"], dtype=np.float64)
        table_log(
            "preflight",
            f"imu_rpy_deg={format_values(rpy)} upright_limit=±{UPRIGHT_DEGREES:.1f}",
        )
        if abs(rpy[0]) >= UPRIGHT_DEGREES or abs(rpy[1]) >= UPRIGHT_DEGREES:
            raise RuntimeError(f"robot is not upright: rpy={format_values(rpy)}")

        table_log("scan", "scanning reachable space with live depth")
        observation = observe_table(points, logger=table_log)
        table_log(
            "scan",
            f"selected surface z={observation['height']:.3f}m "
            f"x={observation['hand_x']:.3f}m points={observation['points']} "
            f"flatness={observation['flatness'] * 1000:.1f}mm",
        )
        if scan_only:
            table_log("complete", "scan-only succeeded; no arm writers were opened")
            return observation

        states = {
            side: stack.enter_context(
                nonsuppressing(Reader(f"arm_{side}.state", keeptime=False))
            )
            for side in SIDES
        }
        table_log("readers", "opened arm_left.state and arm_right.state")
        starts = {
            side: np.asarray(fresh(states[side])["pos"], dtype=np.float64).copy()
            for side in SIDES
        }
        for side in SIDES:
            state = states[side].data
            table_log(
                "state",
                f"side={side} pos={format_values(starts[side])} "
                f"vel={format_values(state['vel'])} current={format_values(state['current'])}",
            )
        configs = {side: Config(f"arm_{side}") for side in SIDES}
        table_log("plan", "initialized per-arm configs; starting full-path IK")
        paths = {}
        for side in SIDES:
            paths[side] = plan_table_path(
                configs[side], starts[side], side, observation, logger=table_log
            )
            validate_calibration(
                paths[side],
                *calibration_limits(bbos, side, logger=table_log),
                side,
                logger=table_log,
            )
        if len(paths["left"]) != len(paths["right"]):
            raise RuntimeError(
                f"arm path lengths differ: left={len(paths['left'])} right={len(paths['right'])}"
            )
        table_log("plan", f"both paths accepted points={len(paths['left'])}")
        if plan_only:
            table_log(
                "complete",
                "plan-only succeeded; arm state was read but no arm writers were opened",
            )
            return observation

        controls = {
            side: stack.enter_context(
                nonsuppressing(
                    Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)
                )
            )
            for side in SIDES
        }
        torques = {
            side: stack.enter_context(
                nonsuppressing(
                    Writer(f"arm_{side}.torque", Type("arm_torque"), keeptime=False)
                )
            )
            for side in SIDES
        }
        table_log("writers", "opened both arm ctrl and torque writers after all checks passed")

        def command(poses):
            for side in SIDES:
                with controls[side].buf() as frame:
                    frame["pos"][:] = poses[side].astype(np.float32)
                    frame["vel"][:] = 0
                    frame["tau"][:] = 0
                    frame["alpha"] = 0.0

        def torque(enabled):
            table_log("torque", f"request enabled={enabled} position_mode=True compliance=False")
            for side in SIDES:
                with torques[side].buf() as frame:
                    frame["enable"][:] = enabled
                    frame["tau_mode"][:] = False
                    frame["compliance_mode"] = False
            table_log("torque", f"published enabled={enabled} to both arms")

        def play(indices, seconds, cancellable, stage):
            began = time.monotonic()
            last_index = indices[0]
            logged_quarters = set()
            table_log(
                "motion",
                f"stage={stage} begin seconds={seconds:.1f} samples={len(indices)} "
                f"index={indices[0]}->{indices[-1]} cancellable={cancellable}",
            )
            while True:
                alpha = min((time.monotonic() - began) / seconds, 1.0)
                offset = min(int(smoothstep(alpha) * (len(indices) - 1)), len(indices) - 1)
                last_index = indices[offset]
                command({side: paths[side][last_index] for side in SIDES})
                quarter = min(int(alpha * 4), 4)
                if quarter not in logged_quarters:
                    logged_quarters.add(quarter)
                    table_log(
                        "motion",
                        f"stage={stage} progress={alpha*100:.0f}% path_index={last_index} "
                        f"left={format_values(paths['left'][last_index])} "
                        f"right={format_values(paths['right'][last_index])}",
                    )
                if alpha >= 1.0 or (cancellable and cancel_event.is_set()):
                    table_log(
                        "motion",
                        f"stage={stage} end path_index={last_index} "
                        f"cancelled={cancel_event.is_set()}",
                    )
                    return last_index
                time.sleep(TICK_SECONDS)

        enabled = False
        last_index = 0
        try:
            table_log("motion", "seeding both controllers at measured start poses")
            for _ in range(8):
                command(starts)
                time.sleep(TICK_SECONDS)
            table_log("motion", "start poses flushed for 8 control ticks")
            torque(True)
            enabled = True
            table_log("motion", "torque on; approaching above the detected surface")
            last_index = play(
                list(range(len(paths["left"]))), MOVE_SECONDS, True, "approach"
            )
            if not cancel_event.is_set():
                measured = {
                    side: np.asarray(fresh(states[side])["pos"], dtype=np.float64)
                    for side in SIDES
                }
                errors = {
                    side: np.abs(measured[side] - paths[side][-1]) for side in SIDES
                }
                for side in SIDES:
                    table_log(
                        "arrival",
                        f"side={side} target={format_values(paths[side][-1])} "
                        f"measured={format_values(measured[side])} "
                        f"abs_error={format_values(errors[side])} "
                        f"worst={float(np.max(errors[side])):.4f}",
                    )
                worst = max(float(np.max(errors[side])) for side in SIDES)
                if worst > 0.10:
                    raise RuntimeError(f"arms did not reach the table pose ({worst:.3f} turns)")
                table_log("hold", "arms at table pose; press Stop to return")
                next_hold_log = time.monotonic() + 5.0
                while not cancel_event.wait(0.10):
                    command({side: paths[side][-1] for side in SIDES})
                    if time.monotonic() >= next_hold_log:
                        table_log("hold", "table pose still active; controller heartbeat healthy")
                        next_hold_log = time.monotonic() + 5.0
                last_index = len(paths["left"]) - 1
        finally:
            if enabled:
                table_log(
                    "cleanup",
                    f"returning along checked path from index={last_index} "
                    f"cancelled={cancel_event.is_set()}",
                )
                return_indices = list(range(last_index, -1, -1))
                play(return_indices, RETURN_SECONDS, False, "return")
                command(starts)
                time.sleep(0.20)
                torque(False)
                table_log("cleanup", "start pose restored; arm torque off")
            else:
                table_log("cleanup", "torque was never enabled; no arm cleanup motion needed")
        table_log("complete", "table-rest action completed cleanly")


def main():
    parser = argparse.ArgumentParser(description="Rest both arms on a detected table")
    parser.add_argument("--pid-file", type=Path)
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--scan-only",
        action="store_true",
        help="log table perception diagnostics without opening arm writers or moving",
    )
    diagnostics.add_argument(
        "--plan-only",
        action="store_true",
        help="scan and validate IK/calibration without opening arm writers or moving",
    )
    args = parser.parse_args()

    def on_signal(signum, _frame):
        table_log("signal", f"received={signum}; requesting safe return")
        cancel_event.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    write_pid_file(args.pid_file)
    try:
        execute(
            args.pid_file,
            scan_only=args.scan_only,
            plan_only=args.plan_only,
        )
        return 0
    except Exception as exc:
        table_log("fatal", f"{type(exc).__name__}: {exc}")
        for line in traceback.format_exc().splitlines():
            table_log("trace", line)
        return 1
    finally:
        remove_pid_file(args.pid_file)


if __name__ == "__main__":
    raise SystemExit(main())
