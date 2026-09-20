"""BBOS runtime for safely executing pre-recorded gestures."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from bbos import Config, Reader, Type, Writer

try:
    from .fist_target import (
        body_turn_deg,
        examine_offered_fist,
        recorded_apex,
        retarget_trajectory,
        stable_fist,
        without_target,
    )
    from .gesture_safety import (
        active_sides,
        depth_clearance,
        ease_seconds,
        plan_recorded_gesture,
        playback_speed,
        pose_at,
        spoken_safety_refusal,
        trajectory_arrays,
    )
except ImportError:
    from fist_target import (
        body_turn_deg,
        examine_offered_fist,
        recorded_apex,
        retarget_trajectory,
        stable_fist,
        without_target,
    )
    from gesture_safety import (
        active_sides,
        depth_clearance,
        ease_seconds,
        plan_recorded_gesture,
        playback_speed,
        pose_at,
        spoken_safety_refusal,
        trajectory_arrays,
    )


TICK_SECONDS = 0.015
ARM_SIDES = ("left", "right")
ARM_RESERVATION_PATH = Path(
    os.environ.get("BAYMAX_ARM_RESERVATION", "/tmp/bracketbot-demo-arm-reserved")
)


def arm_motion_reserved():
    """Whether another trusted process currently owns the robot arms."""
    return ARM_RESERVATION_PATH.exists()


def recorded_movement_name(action):
    """Return the installed recording used to perform a named voice action."""
    return "wave" if action == "goodbye" else action


def fresh(reader, timeout=2.0):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started >= timeout:
            raise RuntimeError("no fresh robot state is available")
        time.sleep(0.005)
    return reader.data


_SWEEP_CACHE: dict = {}

# Gestures whose recording is bent toward the person's hand, and the arm that
# does the reaching. How long to wait for that hand to be held out and still.
AIMED_GESTURES = {"fist bump": "right"}
FIST_WAIT_SECONDS = 2.5
# The fist has already been seen once; this is only to find it again after the
# base has turned.
FIST_RELOOK_SECONDS = 2.0
# Lift travel an aimed gesture may use, in motor turns. The right lift runs from
# -1.0 (bottom) to 0.0 (top, where the arm homes); stay off both ends.
AIMED_LIFT_RANGE = {"right": (-0.97, -0.03)}


def _read_points(depth):
    depth_data = fresh(depth)
    count = int(depth_data["num_points"])
    return np.asarray(depth_data["points"][:count], dtype=np.float32).copy()


def locate_offered_fist(depth, points, wait_seconds=FIST_WAIT_SECONDS):
    """Wait briefly for a fist that two consecutive depth frames agree on.

    Returns ``(fist, points)`` with the cloud the fist was last seen in, so the
    clearance check judges the same scene the arm is aimed into.
    """
    deadline = time.monotonic() + wait_seconds
    previous, reason = examine_offered_fist(points)
    while time.monotonic() < deadline:
        try:
            points = _read_points(depth)
        except RuntimeError:
            reason = "depth stopped updating"
            break
        seen, reason = examine_offered_fist(points)
        fist = stable_fist(previous, seen)
        if fist is not None:
            return fist, points
        if seen is not None:
            reason += ", still moving" if previous is not None else ", waiting for a second look"
        previous = seen
    print(f"[arm] no steady fist: {reason}", flush=True)
    return None, points


def _arm_kinematics(side):
    config = Config(f"arm_{side}")
    config.ik.init()

    def hand_position(motor):
        joints = config.q2urdf(np.asarray(motor, dtype=np.float64).copy())[:7]
        return np.asarray(config.ik.fk(list(joints))[0], dtype=np.float64)

    return config, hand_position


def turn_toward_fist(plan, side, fist, depth, points, turn_body):
    """Turn the base part of the way toward ``fist``, then find the fist again.

    Returns ``(fist, points)`` as seen after the turn. A refused or tiny turn
    leaves both unchanged; a fist lost during the turn comes back as None.
    """
    _, hand_position = _arm_kinematics(side)
    index, recorded = recorded_apex(hand_position, plan.poses[side])
    turn = body_turn_deg(fist, recorded[index])
    if turn == 0.0:
        return fist, points
    print(
        f"[arm] turning {turn:+.0f} deg toward fist {np.round(fist, 3).tolist()}", flush=True
    )
    if not turn_body(turn):
        print("[arm] body turn refused; the arm takes the whole correction", flush=True)
        return fist, points
    return locate_offered_fist(depth, _read_points(depth), FIST_RELOOK_SECONDS)


def aim_plan(plan, side, fist, name=None):
    """Bend one arm of a preflighted plan toward ``fist``; None keeps the recording."""
    config, hand_position = _arm_kinematics(side)
    try:
        aimed = retarget_trajectory(
            hand_position, plan.times, plan.poses[side], fist, config.q2urdf,
            AIMED_LIFT_RANGE.get(side), playback_speed(name),
        )
    except RuntimeError as exc:
        print(f"[arm] fist at {np.round(fist, 3).tolist()} not aimed at: {exc}", flush=True)
        return None
    print(
        f"[arm] aiming at fist {np.round(fist, 3).tolist()}: hand peaks at "
        f"{np.round(aimed.apex, 3).tolist()}, lift {aimed.lift_turns:+.2f} turns "
        f"({aimed.lift_metres:+.3f} m), arm moved {np.round(aimed.offset, 3).tolist()} m, "
        f"largest joint change {aimed.max_joint_delta_turns:.3f} turns",
        flush=True,
    )
    return replace(plan, poses={**plan.poses, side: aimed.trajectory})


def prepare_recorded_movement(frames, name=None, turn_body=None):
    """Read all safety evidence before opening an arm control writer.

    ``turn_body(degrees)`` turns the base in place (positive left) and says
    whether it did; without it an aimed gesture is aimed with the arm alone.
    """
    _, raw_poses = trajectory_arrays(frames)
    sides = active_sides(raw_poses)
    with ExitStack() as stack:
        imu = stack.enter_context(Reader("imu.orientation", keeptime=False))
        depth = stack.enter_context(Reader("camera.points", keeptime=False))
        arm_readers = {
            side: stack.enter_context(Reader(f"arm_{side}.state", keeptime=False))
            for side in sides
        }
        rpy = np.asarray(fresh(imu)["rpy"], dtype=np.float64).copy()
        starts = {
            side: np.asarray(fresh(reader)["pos"], dtype=np.float32).copy()
            for side, reader in arm_readers.items()
        }
        points = _read_points(depth)
        plan = plan_recorded_gesture(frames, starts, rpy, None)
        aimed_side = AIMED_GESTURES.get(name)
        aimed = None
        if aimed_side in plan.sides:
            fist, points = locate_offered_fist(depth, points)
            if fist is not None and turn_body is not None:
                fist, points = turn_toward_fist(
                    plan, aimed_side, fist, depth, points, turn_body
                )
                # The turn took a few seconds on a balancing base: judge
                # uprightness again before any arm moves.
                rpy = np.asarray(fresh(imu)["rpy"], dtype=np.float64).copy()
                plan = plan_recorded_gesture(frames, starts, rpy, None)
            if fist is None:
                print(f"[arm] no fist held out; playing the recorded {name}", flush=True)
            else:
                aimed = aim_plan(plan, aimed_side, fist, name)
        if aimed is not None:
            plan = aimed
            # The hand is meant to meet the fist, so the fist alone is not an
            # obstacle. The forearm and body behind it still are.
            points = without_target(points, fist)
    sweep_paths = {}
    for side in plan.sides:
        # The swept path depends only on the recorded poses, so it is worked
        # out once per recording. Every live input (tilt, arm start, and the
        # depth cloud it is compared against) is still read fresh each time.
        key = (side, plan.poses[side].shape, hash(plan.poses[side].tobytes()))
        if key in _SWEEP_CACHE and aimed is None:
            sweep_paths[side] = _SWEEP_CACHE[key]
            continue
        config = Config(f"arm_{side}")
        config.ik.init()
        waypoints = []
        # About 60 FK samples per recording is enough for a 16 cm clearance
        # radius while keeping the depth comparison inexpensive.
        stride = max(1, len(plan.poses[side]) // 60)
        indices = list(range(0, len(plan.poses[side]), stride))
        if indices[-1] != len(plan.poses[side]) - 1:
            indices.append(len(plan.poses[side]) - 1)
        for index in indices:
            motor = np.asarray(plan.poses[side][index], dtype=np.float64)
            joints = config.q2urdf(motor.copy())[:7]
            position, _ = config.ik.fk(list(joints))
            waypoints.append(np.asarray(position, dtype=np.float64))
        sweep_paths[side] = np.stack(waypoints)
        if aimed is None:
            # An aimed path is different every time; caching it would only grow.
            _SWEEP_CACHE[key] = sweep_paths[side]
    clearance = depth_clearance(points, plan.sides, sweep_paths)
    return replace(plan, clearance_points=clearance)


def _smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def set_arms_limp(sides=ARM_SIDES):
    """Disable arm torque without commanding a new pose."""
    failures = []
    disabled = []
    for side in sides:
        try:
            with Writer(
                f"arm_{side}.torque", Type("arm_torque"), keeptime=False
            ) as torque_writer:
                with torque_writer.buf() as buf:
                    buf["enable"][:] = False
                    buf["tau_mode"][:] = False
                    buf["compliance_mode"] = False
            disabled.append(side)
        except Exception as exc:
            failures.append(f"{side}: {exc}")
    print(f"[arm] torque off; limp={','.join(disabled)}", flush=True)
    if failures:
        raise RuntimeError("failed to disable arm torque (" + "; ".join(failures) + ")")


def play_recorded_movement(name, plan, cancel_event, shutdown_event):
    """Play a preflighted plan, return to measured starts, then torque off."""
    with ExitStack() as stack:
        controls = {
            side: stack.enter_context(
                Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)
            )
            for side in plan.sides
        }
        torque_writers = {
            side: stack.enter_context(
                Writer(f"arm_{side}.torque", Type("arm_torque"), keeptime=False)
            )
            for side in plan.sides
        }

        def command(targets):
            for side in plan.sides:
                with controls[side].buf() as buf:
                    buf["pos"][:] = targets[side]
                    buf["vel"][:] = 0
                    buf["tau"][:] = 0
                    buf["alpha"] = 0.0

        def torque(enabled):
            for side in plan.sides:
                with torque_writers[side].buf() as buf:
                    buf["enable"][:] = enabled
                    buf["tau_mode"][:] = False
                    buf["compliance_mode"] = False

        def ease(from_poses, to_poses):
            seconds = ease_seconds(from_poses, to_poses)
            began = time.monotonic()
            last = from_poses
            while True:
                alpha = _smoothstep((time.monotonic() - began) / seconds)
                last = {
                    side: from_poses[side]
                    + alpha * (to_poses[side] - from_poses[side])
                    for side in plan.sides
                }
                command(last)
                if alpha >= 1.0:
                    return last
                time.sleep(TICK_SECONDS)

        print(
            f"[arm] Safety passed for '{name}': active={','.join(plan.sides)}, "
            f"clearance_points={plan.clearance_points}",
            flush=True,
        )
        torque_enabled = False
        last = dict(plan.starts)
        try:
            for _ in range(10):
                command(plan.starts)
                time.sleep(TICK_SECONDS)
            torque_enabled = True
            torque(True)
            first = {side: plan.poses[side][0] for side in plan.sides}
            last = ease(plan.starts, first)

            if not cancel_event.is_set() and not shutdown_event.is_set():
                playback_times = plan.times / playback_speed(name)
                began = time.monotonic()
                elapsed = 0.0
                while elapsed < playback_times[-1]:
                    if cancel_event.is_set() or shutdown_event.is_set():
                        break
                    elapsed = time.monotonic() - began
                    last = {
                        side: pose_at(playback_times, plan.poses[side], elapsed)
                        for side in plan.sides
                    }
                    command(last)
                    time.sleep(TICK_SECONDS)
            ease(last, plan.starts)
            time.sleep(0.3)
        finally:
            if torque_enabled:
                torque(False)
            print(f"[arm] '{name}' complete; returned to start and torque is off", flush=True)


class RecordedGestureController:
    """Single-motion controller used by the lightweight local assistant."""

    def __init__(self, movements_dir: Path):
        self.movements_dir = Path(movements_dir)
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._shutdown = threading.Event()
        self._thread = None

    def _load(self, name):
        movement_name = recorded_movement_name(name)
        if movement_name not in {
            "wave", "salute", "handshake", "fist bump", "hug", "namaste", "dance"
        }:
            raise RuntimeError(f"gesture '{name}' is not installed in this assistant")
        path = (
            self.movements_dir.parent.parent / "mimic" / "recordings" / "dance.json"
            if movement_name == "dance"
            else self.movements_dir / f"{movement_name}.json"
        )
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not load gesture '{name}': {exc}") from exc

    def start(self, name, turn_body=None):
        if arm_motion_reserved():
            return False, "My arms are busy packing, but I can still answer questions."
        if not self._lock.acquire(blocking=False):
            return False, "Another movement is already running"
        try:
            frames = self._load(name)
            plan = prepare_recorded_movement(frames, name, turn_body)
        except Exception as exc:
            print(f"[arm] {name} preflight rejected: {exc}", flush=True)
            self._lock.release()
            return False, spoken_safety_refusal(name, exc)
        self._cancel.clear()

        def run():
            try:
                play_recorded_movement(
                    name, plan, self._cancel, self._shutdown
                )
            except Exception as exc:
                print(f"[arm] Gesture failed safely: {exc}", flush=True)
            finally:
                if name == "goodbye":
                    try:
                        set_arms_limp()
                    except Exception as exc:
                        print(f"[arm] Could not make both arms limp: {exc}", flush=True)
                self._lock.release()

        self._thread = threading.Thread(
            target=run, name=f"voice-movement-{name}", daemon=True
        )
        self._thread.start()
        return True, f"Started {name}"

    def preflight(self, name):
        """Run the same read-only gate as ``start`` without opening writers."""
        if arm_motion_reserved():
            return False, "My arms are busy packing, but I can still answer questions."
        if self._lock.locked():
            return False, "Another movement is already running"
        try:
            frames = self._load(name)
            plan = prepare_recorded_movement(frames)
        except Exception as exc:
            return False, f"Safety check failed: {exc}"
        return True, (
            f"Safety check passed for {name}; active={','.join(plan.sides)}, "
            f"clearance_points={plan.clearance_points}"
        )

    def stop(self):
        if not self.running():
            return False, "No movement is running."
        self._cancel.set()
        return True, "Okay. Stopping safely."

    def running(self):
        return self._lock.locked()

    def close(self, timeout=8.0):
        self._shutdown.set()
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
