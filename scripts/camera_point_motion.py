"""Execute one bounded camera-guided pointing motion on BracketBot.

Runs on the robot after the vision service has selected a person. The input is
a raw left-eye head-camera pixel. The runner unprojects it through the fisheye
calibration and camera mounting, ranges it with fresh ``camera.points`` depth
(falling back to a chest-height plane), and aims the gripper axis at that 3D
point. The hand itself stays on a short shell near the shoulder; it never
reaches toward the person.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_geometry import (  # noqa: E402
    DEFAULT_D,
    DEFAULT_K,
    DEFAULT_T_POINTS_CAMERA,
    camera_ray_to_points,
    fisheye_pixel_to_ray,
    point_along,
    points_to_arm,
    quaternion_from_z,
    quaternion_slerp,
    target_point,
)


DOF = 8
IK_SAMPLES = 40
MOTOR_SAMPLES = 160
RAMP_SECONDS = 3.5
HOLD_SECONDS = 1.5
MAX_LIFT_TURNS = 0.95
MAX_ARM_TURNS = 0.45
MAX_STEP_TURNS = 0.08
MAX_ENDPOINT_ERROR_METRES = 0.075
UPRIGHT_DEGREES = 25.0
MIN_HEIGHT_BELOW_SHOULDER = 0.60
MAX_HEIGHT_BELOW_SHOULDER = 0.20
MAX_LIFT_FROM_CURRENT = 0.70
# Approximate lateral offset of each arm's shoulder from the arm frame origin.
# Only the aim origin depends on it; the gripper axis is re-aimed from the
# final (clamped) hand position, so small errors do not bias the pointing ray.
SHOULDER_LATERAL_METRES = 0.15
ORIENTATION_BLENDS = (1.0, 0.8, 0.6)
LEGACY_FRAME_SIZE = (1280, 960)
DEPTH_TIMEOUT_SECONDS = 1.5
cancel_event = threading.Event()


def _load_bbos():
    bbos_root = Path.home() / "bbos"
    venv_packages = bbos_root / ".venv/lib/python3.10/site-packages"
    for path in (bbos_root, venv_packages):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    from bbos import Config, Reader, Type, Writer

    return Config, Reader, Type, Writer


def fresh(reader, timeout=2.0):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started >= timeout:
            raise RuntimeError("No fresh robot state is available")
        time.sleep(0.005)
    return reader.data


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def pointing_goal(
    target,
    current_hand_height,
    shoulder_height,
    reach=0.30,
):
    """Choose the arm, hand position, and aim quaternion for an arm-frame target.

    The hand sits ``reach`` metres from the shoulder toward the target, inside
    the existing height band. The gripper Z axis is then aimed from that final
    hand position straight at the target, so clamping never skews the ray.
    """

    target = np.asarray(target, dtype=np.float64)
    side = "left" if target[1] >= 0.0 else "right"
    side_sign = 1.0 if side == "left" else -1.0
    shoulder = np.array([0.0, side_sign * SHOULDER_LATERAL_METRES, shoulder_height])
    position, _ = point_along(shoulder, target, reach)
    # Never cross the body midline, even for a centred target.
    position[1] = side_sign * max(0.06, side_sign * position[1])
    position[0] = max(0.12, position[0])
    minimum_height = shoulder_height - MIN_HEIGHT_BELOW_SHOULDER
    maximum_height = min(
        shoulder_height - MAX_HEIGHT_BELOW_SHOULDER,
        current_hand_height + MAX_LIFT_FROM_CURRENT,
    )
    position[2] = float(np.clip(position[2], minimum_height, maximum_height))
    direction = target - position
    direction /= np.linalg.norm(direction)
    return side, position, direction, quaternion_from_z(direction)


def aim_error_degrees(quaternion, direction):
    """Angle between a quaternion's local Z axis and the desired aim ray."""

    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    forward = np.array(
        [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)]
    )
    cosine = float(np.clip(np.dot(forward, direction), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def legacy_offsets_to_pixel(x_offset, y_offset, size=LEGACY_FRAME_SIZE):
    """Map an old normalized ``[-1, 1]`` image offset to a left-eye pixel."""

    width, height = size
    return 0.5 * (x_offset + 1.0) * width, 0.5 * (y_offset + 1.0) * height


def load_camera_model(Config):
    """Live fisheye intrinsics, rectification, and extrinsic, with fallbacks."""

    K, D, rectify, source = DEFAULT_K, DEFAULT_D, None, "documented"
    try:
        calibration = Config("depth").camera_cal()
        K = np.asarray(calibration[0], dtype=np.float64)
        D = np.asarray(calibration[1], dtype=np.float64).reshape(-1)[:4]
        rectify = np.asarray(calibration[4], dtype=np.float64)
        source = "config"
    except Exception as exc:  # noqa: BLE001 - any config failure uses the fallback
        print(f"[point] stage=camera intrinsics fallback ({exc})", flush=True)
    T, extrinsic = DEFAULT_T_POINTS_CAMERA, "documented"
    for name in ("depth_b", "depth_custom", "depth"):
        try:
            T = np.asarray(Config(name).T_base_cam.mat(), dtype=np.float64)[:3]
            extrinsic = f"{name}.T_base_cam"
            break
        except Exception:  # noqa: BLE001
            continue
    else:
        try:
            T = np.asarray(Config("depth").camera_to_base_3x4, dtype=np.float64)
            extrinsic = "depth.camera_to_base_3x4"
        except Exception:  # noqa: BLE001
            pass
    return K, D, rectify, T, f"intrinsics={source} extrinsic={extrinsic}"


def fresh_depth_cloud(Reader, timeout=DEPTH_TIMEOUT_SECONDS):
    """One newly published ``camera.points`` cloud, or ``None``."""

    try:
        with Reader("camera.points", keeptime=False) as reader:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if reader.ready():
                    data = reader.data
                    count = int(data["num_points"])
                    return np.asarray(data["points"])[:count].reshape(-1, 3).copy()
                time.sleep(0.02)
    except Exception as exc:  # noqa: BLE001 - depth is optional evidence
        print(f"[point] stage=depth unavailable ({exc})", flush=True)
    return None


def locate_target(u, v, K, D, rectify, T, cloud):
    """Arm-frame 3D target for a raw left-eye pixel, plus its evidence source."""

    ray = fisheye_pixel_to_ray(u, v, K, D)
    origin, direction = camera_ray_to_points(ray, T, rectify)
    point, source = target_point(origin, direction, cloud)
    return points_to_arm(point), source


def motor_command_path(start, goal, samples=MOTOR_SAMPLES):
    """Create a continuous joint-space command path to a validated IK goal."""

    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return [
        (1.0 - index / samples) * start + (index / samples) * goal
        for index in range(samples + 1)
    ]


def plan_path(cfg, start, position, quaternion, enforce_limits=True):
    current_urdf = np.asarray(cfg.q2urdf(start.copy()), dtype=np.float64)
    cfg.ik.reset(list(current_urdf[:7]))
    nominal_urdf = list(current_urdf[:7])
    start_position, start_quaternion = cfg.ik.fk(list(current_urdf[:7]))
    start_position = np.asarray(start_position, dtype=np.float64)
    raw_path = [start.copy()]
    for index in range(1, IK_SAMPLES + 1):
        alpha = index / IK_SAMPLES
        waypoint = (1.0 - alpha) * start_position + alpha * position
        orientation = quaternion_slerp(start_quaternion, quaternion, alpha)
        solution = cfg.ik.solve_with_nominal(
            waypoint.tolist(),
            orientation.tolist(),
            nominal_urdf,
        )
        if solution is None or len(solution) < 7:
            raise RuntimeError(f"IK missed waypoint {index}/{IK_SAMPLES}")
        solved_urdf = current_urdf.copy()
        solved_urdf[:7] = np.asarray(solution[:7], dtype=np.float64)
        pose = np.asarray(cfg.urdf2q(solved_urdf), dtype=np.float64)
        pose[7:] = start[7:]
        if not np.isfinite(pose).all():
            raise RuntimeError(f"IK returned non-finite values at waypoint {index}")
        raw_path.append(pose)

    goal = raw_path[-1]
    delta = np.abs(goal - start)
    lift_move = float(delta[0])
    arm_move = float(np.max(delta[1:7]))
    goal_urdf = np.asarray(cfg.q2urdf(goal.copy()), dtype=np.float64)
    reached_position, _ = cfg.ik.fk(list(goal_urdf[:7]))
    endpoint_error = float(
        np.linalg.norm(np.asarray(reached_position, dtype=np.float64) - position)
    )
    if enforce_limits and lift_move > MAX_LIFT_TURNS:
        raise RuntimeError(
            f"lift move is {lift_move:.3f} turns (limit {MAX_LIFT_TURNS:.3f})"
        )
    if enforce_limits and arm_move > MAX_ARM_TURNS:
        raise RuntimeError(
            f"largest rotary-joint move is {arm_move:.3f} turns "
            f"(limit {MAX_ARM_TURNS:.3f})"
        )
    if enforce_limits and endpoint_error > MAX_ENDPOINT_ERROR_METRES:
        raise RuntimeError(
            f"IK endpoint error is {endpoint_error:.3f} m "
            f"(limit {MAX_ENDPOINT_ERROR_METRES:.3f})"
        )

    path = motor_command_path(start, goal)
    differences = np.abs(np.diff(np.asarray(path), axis=0))
    flat_index = int(np.argmax(differences))
    waypoint, joint = np.unravel_index(flat_index, differences.shape)
    largest_step = (float(differences[waypoint, joint]), waypoint + 1, int(joint))
    if enforce_limits and largest_step[0] > MAX_STEP_TURNS:
        raise RuntimeError(
            f"command step is {largest_step[0]:.3f} turns "
            f"(limit {MAX_STEP_TURNS:.3f})"
        )
    total = float(np.max(delta))
    return path, total, largest_step, lift_move, arm_move, endpoint_error


def play_path(writer, path, duration, cancellable):
    began = time.monotonic()
    last_pose = np.asarray(path[0], dtype=np.float64)
    last_position = 0.0
    while not (cancellable and cancel_event.is_set()):
        alpha = min((time.monotonic() - began) / duration, 1.0)
        position = smoothstep(alpha) * (len(path) - 1)
        lower = min(int(position), len(path) - 1)
        upper = min(lower + 1, len(path) - 1)
        fraction = position - lower
        last_pose = (1.0 - fraction) * path[lower] + fraction * path[upper]
        writer["pos"] = last_pose.astype(np.float32)
        last_position = position
        if alpha >= 1.0:
            break
        time.sleep(0.015)
    return last_pose, last_position


def execute(
    u,
    v,
    plan_only=False,
    reach=0.30,
    target_height=None,
):
    Config, Reader, Type, Writer = _load_bbos()
    K, D, rectify, T, model_source = load_camera_model(Config)
    cloud = fresh_depth_cloud(Reader)
    target, target_source = locate_target(u, v, K, D, rectify, T, cloud)
    side = "left" if target[1] >= 0.0 else "right"
    cfg = Config(f"arm_{side}")
    print(
        f"[point] stage=state side={side} pixel=({u:.0f},{v:.0f}) "
        f"target={np.round(target, 2)} source={target_source} {model_source} "
        f"depth_points={0 if cloud is None else len(cloud)}",
        flush=True,
    )
    with ExitStack() as stack:
        imu = stack.enter_context(Reader("imu.orientation", keeptime=False))
        state = stack.enter_context(Reader(f"arm_{side}.state", keeptime=False))
        rpy = np.asarray(fresh(imu)["rpy"], dtype=np.float64)
        if abs(rpy[0]) >= UPRIGHT_DEGREES or abs(rpy[1]) >= UPRIGHT_DEGREES:
            raise RuntimeError("robot is not upright")
        start = np.asarray(fresh(state)["pos"], dtype=np.float64).copy()
        cfg.ik.init()
        start_urdf = np.asarray(cfg.q2urdf(start.copy()), dtype=np.float64)
        current_position, current_quaternion = cfg.ik.fk(list(start_urdf[:7]))
        shoulder_height = float(Config("quest").robot_shoulder_height)
        _, position, direction, aim_quaternion = pointing_goal(
            target,
            float(current_position[2]),
            shoulder_height,
            reach=reach,
        )
        if target_height is not None:
            position[2] = target_height
        # Aim fully along the ray when IK allows it; relax toward the current
        # wrist orientation only as far as needed for a validated path.
        failures = []
        for blend in ORIENTATION_BLENDS:
            quaternion = quaternion_slerp(current_quaternion, aim_quaternion, blend)
            print(
                f"[point] stage=planning current={np.round(current_position, 3)} "
                f"goal={np.round(position, 3)} reach={reach:.2f} "
                f"orientation_blend={blend:.2f} "
                f"aim_error={aim_error_degrees(quaternion, direction):.1f}deg",
                flush=True,
            )
            try:
                path, total, largest_step, lift_move, arm_move, endpoint_error = plan_path(
                    cfg,
                    start,
                    position,
                    quaternion,
                    enforce_limits=True,
                )
                break
            except RuntimeError as exc:
                failures.append(f"blend {blend:.1f}: {exc}")
        else:
            raise RuntimeError("; ".join(failures))
    step, waypoint, joint = largest_step
    print(
        f"[point] stage=planned move={total:.3f} turns "
        f"lift={lift_move:.3f} arm={arm_move:.3f} "
        f"max_step={step:.3f} endpoint_error={endpoint_error:.3f}m",
        flush=True,
    )
    if plan_only:
        print(
            f"[point] plan-start={np.round(start, 3)} "
            f"plan-end={np.round(path[-1], 3)} "
            f"delta={np.round(np.asarray(path[-1]) - start, 3)}",
            flush=True,
        )
        print("[point] stage=plan-only-complete torque was never enabled", flush=True)
        return

    with ExitStack() as stack:
        control = stack.enter_context(
            Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)
        )
        torque = stack.enter_context(
            Writer(f"arm_{side}.torque", Type("arm_torque"), keeptime=False)
        )
        enabled = False
        try:
            print("[point] stage=torque-enable", flush=True)
            control["pos"] = start.astype(np.float32)
            time.sleep(0.10)
            enabled = True
            torque["enable"] = np.ones(DOF, dtype=np.bool_)
            time.sleep(0.70)
            print("[point] stage=pointing", flush=True)
            last_pose, path_position = play_path(control, path, RAMP_SECONDS, True)
            cancel_event.wait(HOLD_SECONDS)
            reached = min(int(path_position), len(path) - 1)
            print("[point] stage=returning", flush=True)
            play_path(control, [last_pose, *reversed(path[: reached + 1])], RAMP_SECONDS, False)
        finally:
            if enabled:
                control["pos"] = start.astype(np.float32)
                time.sleep(0.10)
                torque["enable"] = np.zeros(DOF, dtype=np.bool_)
        print("[point] stage=complete arm returned and torque is off", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Point at a camera-selected person")
    parser.add_argument("--u", type=float, help="raw left-eye pixel column")
    parser.add_argument("--v", type=float, help="raw left-eye pixel row")
    parser.add_argument(
        "--x-offset", type=float, help="legacy normalized image column in [-1, 1]"
    )
    parser.add_argument(
        "--y-offset", type=float, help="legacy normalized image row in [-1, 1]"
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="solve and report the complete path without opening arm writers",
    )
    parser.add_argument("--reach", type=float, default=0.30)
    parser.add_argument(
        "--target-height",
        type=float,
        help="calibration override for plan-only workspace checks",
    )
    args = parser.parse_args()
    if args.u is not None or args.v is not None:
        if args.u is None or args.v is None:
            parser.error("--u and --v must be given together")
        width, height = LEGACY_FRAME_SIZE
        if not 0.0 <= args.u <= width or not 0.0 <= args.v <= height:
            parser.error(f"pixel must lie inside the {width}x{height} left eye")
        u, v = args.u, args.v
    elif args.x_offset is not None and args.y_offset is not None:
        if not -1.0 <= args.x_offset <= 1.0 or not -1.0 <= args.y_offset <= 1.0:
            parser.error("offsets must be between -1 and 1")
        u, v = legacy_offsets_to_pixel(args.x_offset, args.y_offset)
    else:
        parser.error("give --u/--v (or legacy --x-offset/--y-offset)")
    if not 0.20 <= args.reach <= 0.38:
        parser.error("--reach must be between 0.20 and 0.38")
    if args.target_height is not None:
        if not args.plan_only:
            parser.error("--target-height is available only with --plan-only")
        if not 0.45 <= args.target_height <= 1.10:
            parser.error("--target-height must be between 0.45 and 1.10")
    signal.signal(signal.SIGINT, lambda *_: cancel_event.set())
    signal.signal(signal.SIGTERM, lambda *_: cancel_event.set())
    try:
        execute(
            u,
            v,
            plan_only=args.plan_only,
            reach=args.reach,
            target_height=args.target_height,
        )
    except RuntimeError as exc:
        print(f"[point] NOT SAFE TO RUN: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
