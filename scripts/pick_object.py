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
from contextlib import ExitStack, contextmanager
import json
import math
import os
import queue
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
    graspable_candidates,
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
FULL_GRIP_RADIANS = 0.31   # jaw angle holding a drinks can across its middle
PINCH_FRACTION = 0.75      # closed further than this fraction of a full hold: a rim pinch
MAX_PICK_CYCLES = 3        # scan-and-pick rounds per object before giving up on it
MAX_OBJECTS = 8            # --all stops after this many, whatever is still in view
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
PLACE_CLEAR_RIM_METRES = 0.07    # the object's base passes this far above the rim
PLACE_MIN_RIM_METRES = 0.03      # the least the object's base may clear the rim by
PREGRASP_NEAR_BACKOFF_METRES = 0.05  # shorter hover for objects close to the body
PLACE_SINK_METRES = 0.035        # fallback: let the carry sink this much (< the rim clearance)
PLACE_INSIDE_WALL_METRES = 0.08  # release at least this far inside the walls
SCAN_SPREAD_METRES = 0.05
STEADY_WINDOW_FRAMES = 4
STEADY_TILT_DEGREES = 2.5
STEADY_EDGE_METRES = 0.03
STEADY_TIMEOUT_SECONDS = 12.0
OFF_TABLE_METRES = 0.03  # depth blurs the edge; allow this much in front of it
CARRY_SECONDS = 5.0

SERVE_IDLE_SECONDS = 1800

cancel_event = threading.Event()
shutdown_event = threading.Event()


def log(stage, message):
    print(f"[pick][{stage}] {message}", flush=True)


SOUNDS_DIR = Path(__file__).resolve().parent / "sounds"


def speak(name):
    """Say one pre-rendered line (sounds/NAME.wav) without holding up the arm.

    Speech is a courtesy: a missing file, a busy speaker or any other trouble
    is logged and ignored, and the pick carries on.
    """

    def play_mixed():
        """PulseAudio mixes this in beside whatever else is talking.

        The always-on voice assistant keeps the one ``speaker.audio`` writer,
        so that channel is usually taken.
        """
        import shutil
        import subprocess

        if shutil.which("paplay") is None:
            return False
        env = dict(os.environ, XDG_RUNTIME_DIR=os.environ.get(
            "XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        try:
            sinks = subprocess.run(["pactl", "list", "short", "sinks"], env=env, text=True,
                                   capture_output=True, timeout=3).stdout.splitlines()
            sink = next((line.split("\t")[1] for line in sinks
                         if "speakerphone" in line.lower()), None)
            command = ["paplay"] + ([f"--device={sink}"] if sink else [])
            return subprocess.run(command + [str(SOUNDS_DIR / f"{name}.wav")], env=env,
                                  capture_output=True, timeout=15).returncode == 0
        except (OSError, subprocess.SubprocessError, IndexError):
            return False

    def play():
        import wave

        if play_mixed():
            return
        try:
            _, Config, _, Type, Writer = tr._load_bbos()
            cfg = Config("speaker")
            with wave.open(str(SOUNDS_DIR / f"{name}.wav"), "rb") as source:
                if source.getframerate() != cfg.sample_rate or source.getsampwidth() != 2:
                    raise RuntimeError(f"{name}.wav is not 16-bit at {cfg.sample_rate} Hz")
                channels = source.getnchannels()
                # The schema's timing paces chunks to the speaker sample clock.
                with Writer("speaker.audio", Type("speaker_audio")) as speaker:
                    time.sleep(0.25)
                    while True:
                        raw = source.readframes(cfg.chunk_size)
                        if not raw:
                            break
                        samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
                        if channels == 1 and cfg.channels > 1:
                            samples = np.repeat(samples, cfg.channels, axis=1)
                        elif channels != cfg.channels:
                            samples = samples.mean(axis=1).astype(np.int16)[:, None]
                        if len(samples) < cfg.chunk_size:
                            samples = np.concatenate((samples, np.zeros(
                                (cfg.chunk_size - len(samples), cfg.channels), dtype=np.int16)))
                        with speaker.buf() as frame:
                            frame["audio"] = samples
        except Exception as exc:  # noqa: BLE001 - never let speech break a pick
            log("say", f"could not say '{name}': {exc}")

    if not speak.enabled:
        return
    log("say", name)
    with speak.lock:
        previous = speak.thread

        def after_previous():
            if previous is not None:
                previous.join()  # one line at a time: the speaker has one writer
            play()

        speak.thread = threading.Thread(target=after_previous, name="say", daemon=True)
        speak.thread.start()


speak.enabled = True
speak.lock = threading.Lock()
speak.thread = None


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


def grasp_waypoints(item, side, pitch_degrees, table_height_at, yaw_fraction=1.0,
                    lift_metres=None, backoff=None):
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
    backoff = PREGRASP_BACKOFF_METRES if backoff is None else backoff
    pregrasp = grasp - backoff * approach + np.array([0.0, 0.0, PREGRASP_RAISE_METRES])
    lift = grasp + np.array([0.0, 0.0, LIFT_METRES if lift_metres is None else lift_metres])
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
    if np.allclose(np.asarray(start, dtype=np.float64)[:GRIPPER_INDEX],
                   np.asarray(cfg.home, dtype=np.float64)[:GRIPPER_INDEX], atol=2e-3):
        return [start.copy()]  # already at the ready pose: nothing to raise
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


def lift_for_box(box):
    """How high to lift so the object's base clears the container's rim."""

    return LIFT_METRES if box is None else max(LIFT_METRES, box.top + PLACE_CLEAR_RIM_METRES)


def place_candidates(box, lift_point, pitch_degrees, yaw, already_placed=0, allow_sink=True):
    """Release poses to try, best first: ``(position, quaternion, description)``.

    The object is carried level at the lift height (its base already clears
    the rim), so a release only needs a spot inside the walls. The box centre
    is tried first, then spots nearer the pick (a shorter, easier reach). If
    the grasp's wrist pitch cannot reach, others are tried: a can turned in
    the hand on the way over still drops into the box.
    """

    centre = np.asarray(box.center[:2], dtype=np.float64)
    toward = np.asarray(lift_point[:2], dtype=np.float64) - centre
    distance = float(np.linalg.norm(toward))
    inside = max(0.0, min(box.length, box.width) / 2.0 - PLACE_INSIDE_WALL_METRES)
    spots = [centre]
    if distance > 1e-6:
        spots += [centre + toward / distance * min(inside, distance) * fraction
                  for fraction in (0.5, 1.0)]
    # Measured on bracketbot-184: over the box at full lift height, 45-85 deg
    # solve where a flat 25 deg wrist does not, so steeper ones are tried too.
    # Each delivery prefers a different spot, so cans do not land on each other.
    turn = already_placed % len(spots)
    spots = spots[turn:] + spots[:turn]
    pitches = [pitch_degrees] + [p for p in (65.0, 45.0, 85.0, 25.0)
                                 if abs(p - pitch_degrees) > 5.0]
    candidates = []
    # The lift height is near the top of the arm's travel, where few poses
    # solve; as a fallback the carry may sink a little on the way over. The
    # object's base still clears the rim (PLACE_CLEAR_RIM_METRES is larger).
    for sink in (0.0, PLACE_SINK_METRES) if allow_sink else (0.0,):
        for pitch in pitches:
            quaternion, _ = grasp_orientation(pitch, yaw)
            for spot in spots:
                candidates.append((np.array([spot[0], spot[1], lift_point[2] - sink]), quaternion,
                                   f"pitch {pitch:.0f} at ({spot[0]:.3f}, {spot[1]:.3f})"
                                   + (f" sunk {sink * 100:.0f} cm" if sink else "")))
    return candidates


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


def grip_is_pinched(radians, expected=None):
    """True when the jaws closed much further than a full-width hold would.

    Measured on a 66 mm can: a hold across the middle stops near 0.31 rad; one
    that only caught the rim stops near 0.19 rad, and lifting that drops the
    can. ``expected`` is the jaw angle of this arm's last good hold.
    """

    expected = FULL_GRIP_RADIANS if expected is None else expected
    return radians < PINCH_FRACTION * expected


LAST_GOOD_GRIP_RADIANS = {}  # per arm: jaw angle of the last hold that survived the lift
LAST_GOOD_NUDGE_CM = {}  # per arm: the retry nudge whose grasp last held


def ordered_nudges(remembered=None):
    """RETRY_NUDGES_CM with the nudge that held last time first.

    A steady offset between where depth puts the can and where the jaws close
    shows up as the same nudge winning run after run (measured: the plain
    grasp pinched the rim and slipped, +2 cm forward held).
    """

    nudges = list(RETRY_NUDGES_CM)
    if remembered in nudges:
        nudges.remove(remembered)
        nudges.insert(0, remembered)
    return nudges


def ordered_options(pitches, remembered=None):
    """Every (pitch, yaw_fraction) to try, the last accepted one first."""

    options = [(pitch, fraction) for fraction in (1.0, 0.5, 0.0) for pitch in pitches]
    if remembered in options:
        options.remove(remembered)
        options.insert(0, remembered)
    return options


def plan_pick(cfg, start, side, item, plane, pitch, yaw_fraction=1.0, lift_metres=None,
              backoff=None):
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
        item, side, pitch, plane.height_at, yaw_fraction, lift_metres, backoff)
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


def inside_footprint(box, xy, margin=0.02):
    """True when arm-frame ``xy`` lies within the container's rectangle."""

    major = np.array([math.cos(box.yaw), math.sin(box.yaw)])
    offset = np.asarray(xy, dtype=np.float64) - np.asarray(box.center[:2], dtype=np.float64)
    along, across = float(offset @ major), float(offset @ np.array([-major[1], major[0]]))
    return abs(along) <= box.length / 2 + margin and abs(across) <= box.width / 2 + margin


def measure_frame(camera_points, near=None, virtual=None, box_side=None, report=log,
                  filled_box=None):
    """Table, target and container from one ``camera.points`` cloud.

    Pure numpy: the robot scan and the offline replay (pick_replay.py) share
    it, so detection changes can be tried on recorded frames in seconds.
    Returns ``(plane, item, box)``; ``item`` is None when nothing is graspable.
    """

    arm = points_to_arm(np.asarray(camera_points, dtype=np.float64))
    plane = fit_table_plane(arm)
    # Only things standing on the table: a chair back in front of the near
    # edge is can-sized in depth and closer to the robot than the real target.
    objects = [found for found in find_objects(arm, plane)
               if found.center[0] >= plane.near_edge_at(found.center[1]) - OFF_TABLE_METRES]
    if filled_box is not None:
        # Cans already delivered sit inside the box: they are done, not targets.
        objects = [found for found in objects
                   if not inside_footprint(filled_box, found.center[:2])]
    measure_frame.candidates = graspable_candidates(objects, max_reach=1.0)
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
    if (box is not None and filled_box is None and virtual is None
            and inside_footprint(box, item.center[:2])):
        # The chosen object is sitting inside the container: it has been
        # delivered already. Look again with the container's contents hidden.
        return measure_frame(camera_points, near, virtual, box_side, report, filled_box=box)
    return plane, item, box


measure_frame.candidates = []


def merge_candidates(frames, skip=None, match_metres=0.04):
    """Steady positions for every other graspable object seen across ``frames``.

    ``frames`` holds each scan frame's candidate list. The newest frame says
    which objects exist; each one's position is the median of its sightings,
    like the main target's. ``skip`` is the target itself.
    """

    if not frames:
        return []
    merged = []
    for latest in frames[-1]:
        if skip is not None and math.dist(latest.center[:2], skip.center[:2]) < match_metres:
            continue
        seen = [found for frame in frames for found in frame
                if math.dist(found.center[:2], latest.center[:2]) < match_metres]
        if len(seen) < max(2, len(frames) // 2):
            continue  # a flicker, not an object
        centre = np.median([found.center for found in seen], axis=0)
        merged.append(TableObject(
            tuple(float(v) for v in centre),
            float(np.median([found.top for found in seen])),
            float(np.median([found.length for found in seen])),
            float(np.median([found.width for found in seen])),
            latest.yaw, latest.points))
    return merged


def combine_picks(picks, report=log):
    """One steady target from several per-frame ``(plane, item)`` measurements."""

    centres = np.asarray([item.center for _, item in picks])
    spread = float(np.max(np.ptp(centres[:, :2], axis=0)))
    if spread > SCAN_SPREAD_METRES:
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


def view_steady(planes, window=STEADY_WINDOW_FRAMES):
    """True when the last few table fits agree: the robot has stopped moving.

    Entering lean drives the base forward and pitches the head for several
    seconds; depth taken meanwhile gives a small, wandering table plane
    (measured: 165 -> 10000 inliers, tilt 1-19 deg over ~6 s).
    """

    if len(planes) < window:
        return False
    recent = planes[-window:]
    tilts = [plane.tilt_degrees for plane in recent]
    edges = [plane.near_edge for plane in recent]
    inliers = [plane.inliers for plane in recent]
    return bool(max(tilts) - min(tilts) <= STEADY_TILT_DEGREES
                and max(edges) - min(edges) <= STEADY_EDGE_METRES
                and min(inliers) >= 0.8 * max(inliers))


def wait_for_steady_view(Reader, timeout=STEADY_TIMEOUT_SECONDS):
    """Block until the depth view of the table stops changing (or time out)."""

    planes, last_stamp = [], None
    began = time.monotonic()
    with tr.nonsuppressing(Reader("camera.points", keeptime=False)) as reader:
        while time.monotonic() - began < timeout and not cancel_event.is_set():
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
                planes.append(fit_table_plane(arm))
            except RuntimeError:
                planes.clear()
                continue
            if view_steady(planes):
                log("scan", f"view steady after {time.monotonic() - began:.1f}s "
                            f"(tilt={planes[-1].tilt_degrees:.1f}deg "
                            f"near_edge={planes[-1].near_edge:.3f})")
                return True
    log("scan", f"view still moving after {timeout:.0f}s; scanning anyway")
    return False


def scan(Reader, frames=5, timeout=12.0):
    """Fit the table and select a graspable object on fresh, consistent frames."""

    picks, sightings = [], []
    scan.others = []
    near = scan.near
    misses = 0
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
                plane, item, box = measure_frame(cloud, near, scan.virtual, scan.box_side,
                                                 filled_box=scan.filled_box)
            except RuntimeError as exc:
                log("scan", f"frame rejected: {exc}")
                continue
            if item is None:
                # Nothing at the locked spot any more: an automatic lock made
                # on a bad frame must not blind every later one.
                misses += 1
                if scan.near is None and misses >= 2:
                    near, picks, misses = None, [], 0
                continue
            misses = 0
            if near is None:
                near = (item.center[0], item.center[1])  # lock on: no flip-flopping
            picks.append((plane, item))
            sightings.append(list(measure_frame.candidates))
            if box is not None:
                scan.last_box = box  # one frame that misses the box must not lose it
            if len(picks) < frames:
                continue
            centres = np.asarray([found.center[:2] for _, found in picks[-frames:]])
            if float(np.max(np.ptp(centres, axis=0))) <= SCAN_SPREAD_METRES:
                plane, item = combine_picks(picks[-frames:])
                scan.others = merge_candidates(sightings[-frames:], skip=item)
                log("scan", f"{len(sightings[-1])} graspable in the last frame, "
                            f"{len(scan.others)} steady besides the target: " + ", ".join(
                                f"({o.center[0]:.2f}, {o.center[1]:.2f})" for o in scan.others))
                return plane, item
            # The recent frames disagree (robot still settling, or the lock
            # landed on noise): slide the window and re-aim an automatic lock
            # at the newest sighting.
            picks = picks[-(frames - 1):]
            if scan.near is None:
                near = None
    if len(picks) < frames:
        raise RuntimeError(f"no steady graspable object within {timeout:.0f}s "
                           f"({len(picks)} agreeing frames)")
    return combine_picks(picks[-frames:])


scan.near = None
scan.virtual = None
scan.last_box = None
scan.box_side = None
scan.record_dir = None
scan.recorded = 0
scan.filled_box = None
scan.others = []


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
        initial, pitches = 0.0, []
        try:
            if self.Reader is None:
                raise RuntimeError("no IMU reader")
            with tr.nonsuppressing(self.Reader("imu.orientation", keeptime=False)) as imu:
                initial = float(np.asarray(tr.fresh(imu)["rpy"])[1])
                while time.monotonic() - began < LEAN_SETTLE_SECONDS:
                    time.sleep(LEAN_PERIOD_SECONDS)
                    if imu.ready():
                        pitches.append(float(np.asarray(imu.data["rpy"])[1]))
                    if lean_settled(pitches, initial, self.degrees):
                        break
        except Exception as exc:  # noqa: BLE001 - any IMU trouble: use the fixed wait
            log("lean", f"timed settle ({exc})")
            time.sleep(max(0.0, LEAN_SETTLE_SECONDS - (time.monotonic() - began)))
        span = f"pitch {initial:.1f} -> {pitches[-1]:.1f} deg" if pitches else "no IMU samples"
        log("lean", f"settled after {time.monotonic() - began:.1f}s ({span})")

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


TELEOP_RELAY = ("127.0.0.1", 8765)  # scripts/robot_teleop.py RELAY_PORT


@contextmanager
def drive_channel(Type, Writer):
    """Yields ``twist(v)`` for straight base moves, however the base is owned.

    bbos allows one ``drive.ctrl`` writer. When the WASD teleop runner already
    holds it, the twist goes to that runner's robot-local relay instead, which
    applies its own clamps and deadman and lets a held key override it.
    """

    try:
        manager = tr.nonsuppressing(Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False))
        drive = manager.__enter__()
    except Exception as exc:  # noqa: BLE001 - bbos raises a bare Exception for a taken topic
        if "already exists" not in str(exc):
            raise
        import socket

        log("space", f"drive.ctrl is held by the teleop runner; using its relay ({exc})")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as relay:
            yield lambda v: relay.sendto(
                json.dumps({"v": float(v), "w": 0.0}).encode(), TELEOP_RELAY)
        return
    try:
        def twist(v):
            with drive.buf() as frame:
                frame["twist"] = np.array([v, 0.0], dtype=np.float32)

        yield twist
    finally:
        manager.__exit__(None, None, None)


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
    with drive_channel(Type, Writer) as twist:
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
            auto_space_enabled=False, everything=False):
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
    def pick_everything():
        """Pick (and place) until nothing graspable is left, re-scanning each round.

        A round that misses or drops the object is followed by a fresh scan,
        so the next try aims at where the object really is now.
        """
        space_if_needed()
        kept_box, placed, failures = None, 0, 0
        pick_with.placed = pick_with.skipped = 0
        while not cancel_event.is_set():
            scan.filled_box = kept_box
            try:
                plane, item = scan(Reader)
            except RuntimeError as exc:
                if placed or failures:
                    speak("out-of-reach" if pick_with.skipped else "done" if placed else "none")
                    log("complete", f"nothing more to pick ({exc}); delivered {placed}")
                    return "PLACED" if placed else None
                speak("none")
                raise
            box = kept_box or box_for(plane, item, place)
            others = [other for other in scan.others
                      if box is None or not inside_footprint(box, other.center[:2])]
            try:
                result = pick_with(bbos, Config, Reader, Type, Writer, plane, item,
                                   plan_only, stop_at, adjust, grip_torque, box,
                                   others if everything and box is not None else ())
            except RuntimeError as exc:
                if not pick_with.placed:
                    raise
                # Something is left, but the arm cannot get to it: that ends a
                # clearing run, it does not make the run a failure.
                speak("out-of-reach")
                log("complete", f"delivered {pick_with.placed}; the object left at "
                                f"({item.center[0]:.2f}, {item.center[1]:.2f}) is out of reach "
                                f"({object_reach(item):.2f} m): {str(exc)[:120]}")
                return "PLACED"
            placed = pick_with.placed
            if plan_only or stop_at or cancel_event.is_set():
                return result
            if result == "PLACED" or (result == "PICKED" and box is None):
                failures = 0
                if result == "PLACED":
                    kept_box = box  # it stops looking empty once something is in it
                if not (everything and result == "PLACED") or placed >= MAX_OBJECTS:
                    log("complete", f"delivered {placed} object(s)")
                    return result
                continue
            failures += 1
            if failures >= MAX_PICK_CYCLES:
                log("complete", f"gave up after {failures} rounds on one object; "
                                f"delivered {placed}")
                return result
            speak("retry")
            log("retry", f"round {failures} ended {result}; scanning again for the object")
        return None

    if not allow_lean or external:
        log("lean", "lean already held elsewhere" if external else "no lean (pass --lean to hold one)")
        return pick_everything()
    with LeanHold(Type, Writer, STABILITY_LEAN_DEGREES, Reader):
        TIMING.mark("lean settled")
        wait_for_steady_view(Reader)
        TIMING.mark("view steady")
        return pick_everything()


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
              plan_only, stop_at, adjust, grip_torque, place_box=None, more_items=()):
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

    lift_metres = lift_for_box(place_box)
    if place_box is not None:
        log("place", f"lifting {lift_metres * 100:.0f} cm to clear the "
                     f"{place_box.top * 100:.0f} cm rim")
    low, high = tr.calibration_limits(bbos, side, logger=log)
    shoulder_y = SHOULDER_LATERAL_METRES if side == "left" else -SHOULDER_LATERAL_METRES

    def plan_can(target, begin):
        """Validated phases for one object, from ``begin`` (rest pose or ready pose)."""
        failures = []
        pitches = (NEAR_GRASP_PITCHES_DEGREES if object_reach(target) < NEAR_OBJECT_METRES
                   else GRASP_PITCHES_DEGREES)
        # The object rarely moves between tests, so the option that validated
        # last time is tried first instead of re-rejecting the ones before it.
        # A can close to the body needs a steep wrist, which puts the hover
        # point and the lift near the top of the arm's travel. Fall back to a
        # shorter hover and then a lower lift (still above the rim) before
        # giving up on it.
        lifts = [lift_metres]
        if place_box is not None and lift_metres > place_box.top + PLACE_MIN_RIM_METRES + 0.005:
            lifts.append(place_box.top + PLACE_MIN_RIM_METRES)
        shapes = [(lifts[0], PREGRASP_BACKOFF_METRES), (lifts[0], PREGRASP_NEAR_BACKOFF_METRES)]
        shapes += [(lift, PREGRASP_NEAR_BACKOFF_METRES) for lift in lifts[1:]]
        options = ordered_options(pitches, pick_with.accepted.get(side))
        for lift_height, backoff, pitch, yaw_fraction in (
                (lift, back, *option) for lift, back in shapes for option in options):
            try:
                path, marks, observation = plan_pick(cfg, begin, side, target, plane, pitch,
                                                     yaw_fraction, lift_height, backoff)
                phases = split_dense(path, marks)
                for name, phase in phases.items():
                    tr.validate_calibration(phase, low, high, f"{side}:{name}", logger=log)
                    if name != "raise":
                        tr.validate_playback_clearance(phase, cfg, observation,
                                                       f"{side}:{name}", logger=log)
            except RuntimeError as exc:
                failures.append(f"pitch {pitch:.0f} yaw x{yaw_fraction:.1f}: {exc}")
                log("plan", f"pitch={pitch:.0f}deg yaw_fraction={yaw_fraction:.1f} "
                            f"rejected: {exc}")
                continue
            raised = phases.pop("raise")
            phases["approach"] = np.vstack([raised, phases["approach"][1:]])
            pick_with.accepted[side] = (pitch, yaw_fraction)
            _, grasp, lift, quaternion = grasp_waypoints(
                target, side, pitch, plane.height_at, yaw_fraction, lift_height, backoff)
            if (lift_height, backoff) != shapes[0]:
                log("plan", f"used a {backoff * 100:.0f} cm hover and a "
                            f"{lift_height * 100:.0f} cm lift to reach this one")
            return {"item": target, "phases": phases, "ready_index": len(raised) - 1,
                    "observation": observation, "pitch": pitch, "yaw_fraction": yaw_fraction,
                    "grasp": grasp, "lift": lift, "quaternion": quaternion,
                    "lift_height": lift_height,
                    "options_tried": len(failures) + 1}
        raise RuntimeError("no validated grasp: " + " | ".join(failures))

    def plan_carry(job, already_placed):
        """First reachable release pose inside the box, or None (put it back)."""
        target_item, phases = job["item"], job["phases"]
        yaw = job["yaw_fraction"] * math.atan2(target_item.center[1] - shoulder_y,
                                               target_item.center[0])
        reasons = []
        # Sinking on the way over is only safe from the full-height lift.
        full_lift = job["lift_height"] >= lift_metres - 1e-6
        for target, quaternion, description in place_candidates(
                place_box, job["lift"], job["pitch"], yaw, already_placed, full_lift):
            try:
                carry = plan_place(cfg, side, phases["lift"][-1], target, quaternion,
                                   job["observation"], low, high)
            except RuntimeError:
                reasons.append(description)
                continue
            log("place", f"box at {fmt(place_box.center)} rim={place_box.top:.3f}; release "
                         f"{description} z={target[2]:.3f} ({len(carry)} poses)")
            return carry
        log("place", f"cannot reach inside the box from this lift ({len(reasons)} poses "
                     "tried); the object will be put back")
        return None

    def plan_extras(job, already_placed):
        # Runs while the arm is busy streaming poses, which never touches
        # cfg.ik; the motion loop collects the result before it needs it, so
        # the solver is never shared between threads.
        phases = job["phases"]
        attempts = None
        if adjust:
            attempts = []
            for nudge_cm in ordered_nudges(LAST_GOOD_NUDGE_CM.get(side)):
                offset = np.asarray(nudge_cm, dtype=np.float64) / 100.0
                if np.any(np.abs(offset) > MAX_ADJUST_METRES):
                    continue
                if not np.any(offset):
                    attempts.append((offset, phases["descend"], phases["lift"]))
                    continue
                try:
                    descend, lifted = plan_adjusted(
                        cfg, side, phases["approach"][-1], job["grasp"] + offset,
                        job["lift"] + offset, job["quaternion"], job["observation"], low, high)
                except RuntimeError as exc:
                    log("retry", f"nudge_cm={fmt(offset * 100)} rejected: {exc}")
                    continue
                attempts.append((offset, descend, lifted))
            log("retry", f"{len(attempts)} validated attempts planned while moving")
        carry = None
        if place_box is not None and attempts:
            carry = plan_carry(job, already_placed)
        job["attempts"], job["carry"] = attempts, carry
        return job

    TIMING.mark("scan done, planning")
    first = plan_can(item, start)
    phases = first["phases"]
    TIMING.mark(f"plan validated after {first['options_tried']} option(s)")
    grasp_urdf = np.asarray(cfg.q2urdf(phases["descend"][-1].copy()), dtype=np.float64)
    reached_xyz, _ = cfg.ik.fk(list(grasp_urdf[:7]))
    log("plan", f"grasp pose reaches {fmt(reached_xyz)}")
    log("plan", f"accepted pitch={first['pitch']:.0f}deg "
                f"yaw_fraction={first['yaw_fraction']:.1f} phases=" + ",".join(
                    f"{name}:{len(phase)}" for name, phase in phases.items()))
    # The ready pose: arm raised above the table edge, gripper open. Every
    # later object is planned from here, so the arm never goes back to rest
    # between objects.
    ready = phases["approach"][first["ready_index"]].copy()
    ready[GRIPPER_INDEX] = phases["descend"][-1][GRIPPER_INDEX]
    at_home = np.allclose(ready[:GRIPPER_INDEX],
                          np.asarray(cfg.home, dtype=np.float64)[:GRIPPER_INDEX], atol=2e-3)
    chain = [] if stop_at or not at_home else [
        other for other in more_items
        if ("left" if other.center[1] >= 0.0 else "right") == side
        and math.hypot(other.center[0], other.center[1] - shoulder_y) <= HARD_REACH_METRES]
    if more_items:
        log("plan", f"{len(more_items)} other graspable object(s) seen; ready pose is "
                    f"{'home' if at_home else 'NOT home, so no chaining'}")
    if chain:
        log("plan", f"{len(chain)} more object(s) queued for the {side} arm: "
                    + ", ".join(f"({c.center[0]:.2f}, {c.center[1]:.2f})" for c in chain))
    if plan_only:
        if place_box is not None:
            plan_carry(first, pick_with.placed)  # report, without moving, if the box is reachable
        for number, other in enumerate(chain, start=1):
            try:
                queued = plan_can(other, ready)
                log("plan", f"queued object {number} at ({other.center[0]:.2f}, "
                            f"{other.center[1]:.2f}) validated from the ready pose: "
                            f"pitch={queued['pitch']:.0f}deg")
                if place_box is not None:
                    plan_carry(queued, pick_with.placed + number)
            except RuntimeError as exc:
                log("plan", f"queued object {number} would be skipped: {exc}")
        log("complete", "plan-only succeeded; no arm writers were opened")
        return None

    jobs = queue.Queue()
    jobs.abandon = threading.Event()  # set by the motion loop when it stops early
    jobs.skipped = 0                  # queued objects the arm could not plan a way to

    def plan_everything():
        jobs.put(plan_extras(first, pick_with.placed))
        for number, other in enumerate(chain, start=1):
            if cancel_event.is_set() or jobs.abandon.is_set():
                break
            try:
                jobs.put(plan_extras(plan_can(other, ready), pick_with.placed + number))
            except RuntimeError as exc:
                jobs.skipped += 1
                log("plan", f"skipping ({other.center[0]:.2f}, {other.center[1]:.2f}): "
                            f"{str(exc)[:160]}")
        jobs.put(None)

    TIMING.mark("motion starts")
    if not pick_with.placed:
        speak("starting")
    result, placed = run_motion(Config, Reader, Type, Writer, cfg, side, start, first, stop_at,
                                grip_torque, jobs, Deferred(plan_everything))
    pick_with.placed += placed
    pick_with.skipped += jobs.skipped
    return result


pick_with.accepted = {}
pick_with.placed = 0
pick_with.skipped = 0


def run_motion(*args, **kwargs):
    """Play the pick; returns PLACED, PICKED, PINCHED, SLIPPED, MISS or None."""

    # The motion routine leaves through several early returns (and a finally
    # that retraces the arm), so the result travels in a dict, not a return.
    outcome = {"result": None, "placed": 0}
    _run_motion(outcome, *args, **kwargs)
    log("complete", f"pick attempt finished: {outcome['result']} "
                    f"({outcome['placed']} delivered this run)")
    return outcome["result"], outcome["placed"]


def _run_motion(outcome, Config, Reader, Type, Writer, cfg, side, start, job, stop_at=None,
                grip_torque=GRIP_CLOSE_TORQUE_NM, jobs=None, planner=None):
    """``job`` is the first object's plan; ``jobs`` then delivers it again with
    its retries and carry attached, followed by one job per further object
    (planned in the background from the ready pose) and finally ``None``."""

    attempts = carry = None
    phases, ready_index = job["phases"], job["ready_index"]
    with ExitStack() as stack:
        state = stack.enter_context(tr.nonsuppressing(Reader(f"arm_{side}.state", keeptime=False)))
        control = stack.enter_context(tr.nonsuppressing(
            Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)))
        torque = stack.enter_context(tr.nonsuppressing(
            Writer(f"arm_{side}.torque", Type("arm_torque"), keeptime=False)))

        grip = {"tau": 0.0, "force_mode": False, "radians": 0.0}

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
            grip["radians"] = radians
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
            if jobs is not None:
                job = jobs.get()
                attempts, carry = job["attempts"], job["carry"]
            if cancel_event.is_set():
                return
            if stop_at == "pregrasp":
                log("staged", "hovering at pregrasp; returning without descending")
                cancel_event.wait(HOLD_SECONDS)
                return
            if attempts:
                while True:
                    result = None
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
                        if holding and number < len(attempts) and grip_is_pinched(
                                grip["radians"], LAST_GOOD_GRIP_RADIANS.get(side)):
                            # Only the rim is caught. Lifting this drops the can and
                            # knocks it over; let go where it stands and re-grip.
                            log("grip", f"pinched ({grip['radians']:.3f} rad); re-gripping, not lifting")
                            holding, result = False, "PINCHED"
                        if holding:
                            stage = "lift"
                            reached = play(with_gripper(lift, hold_turns), LIFT_SECONDS, False,
                                           "lift", pace="lift")
                            data = tr.fresh(state)
                            still = gripper_radians(cfg, np.asarray(data["pos"])) >= HOLDING_MIN_RADIANS
                            result = "PICKED" if still else "SLIPPED"
                            log("evidence", f"attempt={number} after lift holding={still} "
                                            f"current={float(np.asarray(data['current'])[GRIPPER_INDEX]):.2f}A")
                            if carry is None or not still:
                                cancel_event.wait(HOLD_SECONDS)
                            if carry is not None and still:
                                # Carrying is deliberately not cancellable: a Stop
                                # mid-carry finishes over the box and releases there
                                # rather than dropping the object anywhere.
                                stage = "carry"
                                carry = np.asarray(densify([lift[-1], *carry]), dtype=np.float64)
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
                        outcome["result"] = result
                        if result in {"PICKED", "PLACED"}:
                            LAST_GOOD_NUDGE_CM[side] = tuple(
                                float(v) for v in np.round(offset * 100.0, 1))
                            LAST_GOOD_GRIP_RADIANS[side] = grip["radians"]
                            break
                        if result == "SLIPPED":
                            # It was lifted and dropped: nobody knows where it is
                            # now, so more blind nudges only knock it about. Pull
                            # back and let the caller look again.
                            log("retry", "dropped it; retreating to re-scan before another try")
                            break
                    if result == "PLACED":
                        outcome["placed"] += 1
                    if result != "PLACED" or cancel_event.is_set() or jobs is None:
                        return
                    job = jobs.get()  # planned while this object was being carried
                    if job is None or not job["attempts"]:
                        return
                    # Straight on to the next object: up to the ready pose and
                    # out again, without lowering the arm to rest in between.
                    target = job["item"].center
                    speak("next")
                    log("retry", f"next object at ({target[0]:.2f}, {target[1]:.2f}); "
                                 "not returning to rest")
                    stage = "approach"
                    play(with_gripper(approach[ready_index:][::-1], open_turns),
                         APPROACH_SECONDS, False, "to-ready", pace="approach")
                    reached = ready_index
                    # Keep the raise as the way home; swap in the new reach.
                    approach = np.vstack([approach[:ready_index + 1],
                                          job["phases"]["approach"][1:]])
                    descend, lift = job["phases"]["descend"], job["phases"]["lift"]
                    attempts, carry = job["attempts"], job["carry"]
                    reached = ready_index + play(
                        with_gripper(approach[ready_index:], open_turns), APPROACH_SECONDS,
                        True, "approach", pace="approach")
                    if cancel_event.is_set():
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
            if planner is not None:
                jobs.abandon.set()
                planner.wait()  # never leave the planner running on the shared solver
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
    parser.add_argument("--all", action="store_true",
                        help="with --place: keep going until every graspable object "
                             "beside the box has been delivered into it")
    parser.add_argument("--box-side", choices=("left", "right"),
                        help="which side of the object the container is on")
    parser.add_argument("--fix-gripper", choices=("left", "right"),
                        help="drive that gripper back into its valid range and stop")
    parser.add_argument("--lean", action="store_true",
                        help="hold a bracing forward lean for the whole pick (the old "
                             "default; the base no longer needs it)")
    parser.add_argument("--no-lean", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rest", action="store_true",
                        help="lower both arms back to their hanging rest pose and stop")
    parser.add_argument("--grip-torque", type=float, default=GRIP_CLOSE_TORQUE_NM,
                        help=f"constant gripper squeeze in Nm once contact is made "
                             f"(default {GRIP_CLOSE_TORQUE_NM}, max {MAX_GRIP_TORQUE_NM})")
    parser.add_argument("--record", type=Path, metavar="DIR",
                        help="save every scanned depth frame here for offline replay "
                             "with scripts/pick_replay.py")
    parser.add_argument("--quiet", action="store_true", help="do not speak")
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
    speak.enabled = not args.quiet
    scan.recorded = 0
    scan.last_box = None
    scan.filled_box = None
    scan.others = []
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
                grip_torque=args.grip_torque, allow_lean=args.lean and not args.no_lean,
                place=args.place, auto_space_enabled=args.auto_space,
                everything=args.all)
        return 0
    except Exception as exc:  # noqa: BLE001 - report every failure as NOT SAFE
        log("fatal", f"NOT SAFE TO RUN: {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().splitlines():
            log("trace", line)
        return 1
    finally:
        if speak.thread is not None:
            speak.thread.join(timeout=8.0)  # let the last line finish before the job ends
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
