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
    from .gesture_safety import (
        active_sides,
        depth_clearance,
        plan_recorded_gesture,
        spoken_safety_refusal,
        trajectory_arrays,
    )
except ImportError:
    from gesture_safety import (
        active_sides,
        depth_clearance,
        plan_recorded_gesture,
        spoken_safety_refusal,
        trajectory_arrays,
    )


EASE_SECONDS = 3.0
PLAYBACK_SPEED = 0.6
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


def prepare_recorded_movement(frames):
    """Read all safety evidence before opening an arm control writer."""
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
        depth_data = fresh(depth)
        count = int(depth_data["num_points"])
        points = np.asarray(depth_data["points"][:count], dtype=np.float32).copy()
    plan = plan_recorded_gesture(frames, starts, rpy, None)
    sweep_paths = {}
    for side in plan.sides:
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
            began = time.monotonic()
            last = from_poses
            while True:
                alpha = _smoothstep((time.monotonic() - began) / EASE_SECONDS)
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
                playback_times = plan.times / PLAYBACK_SPEED
                began = time.monotonic()
                index = 0
                while index < len(playback_times) - 1:
                    if cancel_event.is_set() or shutdown_event.is_set():
                        break
                    elapsed = time.monotonic() - began
                    index = min(
                        int(np.searchsorted(playback_times, elapsed)),
                        len(playback_times) - 1,
                    )
                    last = {side: plan.poses[side][index] for side in plan.sides}
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

    def start(self, name):
        if arm_motion_reserved():
            return False, "My arms are busy packing, but I can still answer questions."
        if not self._lock.acquire(blocking=False):
            return False, "Another movement is already running"
        try:
            frames = self._load(name)
            plan = prepare_recorded_movement(frames)
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
