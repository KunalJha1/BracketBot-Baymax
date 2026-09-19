# Person-Follow Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BracketBot follows one person who raised a hand and holds a runtime-adjustable gap (0.6–1.5 m, default 1.0 m) to within ±20 cm, never reversing, stopping for obstacles, and stopping safely on any loss of data, link, or balance.

**Architecture:** A pure-numpy core (`follow_core.py`: lock-on, Kalman tracker, controller, supervisor, state machine) and perception module (`follow_perception.py`: YOLO11n-pose TensorRT decode, torso localisation from `camera.points`) run inside a thin robot-side runner (`robot_follow.py`) on the Jetson. The existing localhost dashboard starts and stops the runner over SSH, streams the gap and a heartbeat on stdin, and shows status parsed from stdout. Robot gates G0–G6 (Task 11) turn it into hardware evidence.

**Tech Stack:** Python 3.10+, numpy, pytest; BBOS shared-memory `Reader`/`Writer` on the robot; TensorRT + PyCUDA (Jetson, through the greeter's dependency set); `uv run --script` (PEP 723) on the robot; Ultralytics (laptop, only for ONNX export and one optional cross-check test).

**Spec:** `docs/superpowers/specs/2026-09-19-person-follow-design.md`. Read it before starting; §11 lists what changed during prototyping.

**Provenance:** every code block in Tasks 1–10 was run before this plan was written. A generator replayed each step on a fresh copy of the repository, including every "expect FAIL" and "expect PASS" run, and the `Expected:` lines quote that replay. The replay proves the laptop-side code and tests; it proves nothing about the robot. That is Task 11's job.

## Global Constraints

- Python: laptop code must run on 3.10+ (`requires-python = ">=3.10"`); the robot runner pins `==3.10.*` in its PEP 723 header. No 3.11+ syntax or stdlib.
- `scripts/follow_core.py` and `scripts/follow_perception.py` import only the standard library and numpy at module load. TensorRT, PyCUDA, OpenCV, and BBOS are imported inside the functions that need them.
- `scripts/robot_dashboard.py` stays standard-library only. It must not import `follow_core` or numpy; it duplicates `FOLLOW_GAP_*` and `FOLLOW_STATUS_PREFIX`, and a test keeps them equal.
- Robot files are deployed flat to `/tmp` and import siblings by bare name (`import follow_core`). Tests reach them through `tests/conftest.py`.
- Internal units are metres, seconds, and radians, in robot-local `(forward, left, up)`. Degrees appear only in status and log output.
- The robot never reverses: `v >= 0` is enforced in the controller, the supervisor, and the runner.
- `FollowConfig` defaults are the spec §6.7 values; change one only together with the spec.
- The speed cap is 0.15 m/s until gate G4b passes, and never above 0.30 m/s (the drive daemon's clamp).
- On the robot, BBOS-venv commands always use `uv run --no-sync --project ~/bbos`; the runner uses `uv run --script`. Never `uv sync` or `pip install` into `~/bbos`.
- No robot gate that can move the base runs without a person at the physical e-stop.
- Simulation results are reported as simulation evidence, never as hardware evidence.
- Run laptop commands from the repository root in Git Bash. Tests: `uv run --extra dev python -m pytest …` (uv may create `uv.lock`; the repository does not track it, so leave it uncommitted).
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` (the commit commands below include it).

## File Structure

| File | Task | Runs on | Responsibility |
|---|---|---|---|
| `pyproject.toml` | 1 | — | `numpy` added to the `dev` extra |
| `tests/conftest.py` | 1 | laptop | puts `scripts/` on `sys.path` so tests import robot modules by bare name |
| `scripts/follow_core.py` | 1–4 | robot + laptop | config, geometry, lock-on, tracker, controller, corridor, supervisor, `FollowLoop`, protocol, LED, helpers |
| `tests/test_follow_core.py` | 1–4 | laptop | unit tests for all of the above |
| `tests/test_follow_sim.py` | 5 | laptop | closed-loop kinematic simulation of every spec scenario |
| `scripts/follow_perception.py` | 6 (+10) | robot (+ laptop tests) | letterbox, pose decode + NMS, hand-raised, torso rect/histogram, point-to-pixel mapping, torso position, `PoseEngine` |
| `tests/test_follow_perception.py` | 6 (+10) | laptop | decode/geometry tests and the Ultralytics cross-check |
| `scripts/robot_follow.py` | 7 (+10) | robot | PEP 723 runner: BBOS I/O, 50 Hz loop, stdin/stdout protocol, CSV log, safe exit |
| `tests/test_robot_follow_cli.py` | 7 (+10) | laptop | argument safety, stereo split, perception glue with a fake engine |
| `scripts/robot_dashboard.py` | 8 | laptop | Follow button, gap slider, heartbeat, status, exclusivity, simulation, gate flags |
| `tests/test_robot_dashboard.py` | 8 | laptop | follow tests appended |
| `README.md` | 8 | — | follow-mode section and control-table row |
| `scripts/follow_log_report.py` | 9 | laptop | gate metrics from a runner CSV |
| `tests/test_follow_log_report.py` | 9 | laptop | report on a synthetic log |
| `scripts/probe_follow.py` | 9 | robot | read-only gate G0 probe |
| `scripts/check_follow_alignment.py` | 9 | laptop | gate G0 depth-vs-image overlay |
| `scripts/build_pose_engine.sh` | 9 | robot | gate G1 TensorRT engine build with a version record |
| `docs/robot-facts.md` | 11 | — | gate results |

Task 10 is **conditional** (only if G0 finds `camera.rect` unusable). Task 11 is hardware work with a person at the e-stop; an agent may run G0 and G1 (read-only / no motion) but must stop and hand over before G2.

---

### Task 1: Branch, test setup, and `follow_core` foundations

**Files:**
- Modify: `pyproject.toml` (`dev` extra)
- Create: `tests/conftest.py`
- Create: `scripts/follow_core.py`
- Test: `tests/test_follow_core.py`

**Interfaces:**
- Consumes: nothing.
- Produces (in `follow_core`): `STATUS_PREFIX = "FOLLOW_STATUS "`; states `SEARCHING`, `FOLLOWING`, `BLOCKED`, `LOST`;
  frozen `FollowConfig` (every §6.7 parameter); `clamp(value, low, high)`; `shrink(value, deadband)`;
  `Pose2D(x, y, h)` with `integrate(v, omega, dt)`, `to_odom(forward, left) -> (x, y)`, `to_local(x, y) -> (forward, left)`;
  `PersonObservation(forward, left, score, hand_raised, hist=None)` with `.range`/`.bearing`;
  `Perception(t, people: tuple[PersonObservation, ...], points: ndarray (N, 3) forward/left/up)`;
  `Track(forward, left, range, bearing, v_radial, age)`; `hist_distance(p, q) -> float`.

Everything later builds on these types, so they come first. `numpy` joins the `dev` extra because the new tests need it (the existing `tests/test_people_detector.py` already did, and fails without it).

- [ ] **Step 1: Create the feature branch and commit the design documents**

```bash
git checkout -b feature/person-follow
git add docs/superpowers/specs/2026-09-19-person-follow-design.md docs/superpowers/plans/2026-09-19-person-follow.md
git commit -m "docs: person-follow design and implementation plan" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Add numpy to the `dev` extra and make `scripts/` importable from tests**

In `pyproject.toml`, replace:

```toml
dev = [
    "pytest>=8",
]
```

with:

```toml
dev = [
    "numpy>=1.24",
    "pytest>=8",
]
```

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

# Robot-side scripts are deployed flat to /tmp and import their siblings by bare
# name (``import follow_core``); put scripts/ on the path so tests do the same.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_follow_core.py`:

```python
import dataclasses
import json
import math

import numpy as np
import pytest

from follow_core import (
    FollowConfig,
    PersonObservation,
    Pose2D,
    hist_distance,
)

CFG = FollowConfig()
FAST = dataclasses.replace(CFG, v_max=0.30)
FRAME = 1 / 15


def person(forward, left=0.0, raised=False, hist=None):
    return PersonObservation(forward, left, 0.9, raised, hist)


def onehot(index):
    hist = np.zeros(64)
    hist[index] = 1.0
    return hist


# --- geometry -------------------------------------------------------------

def test_pose_local_and_odom_round_trip():
    pose = Pose2D(1.0, 2.0, 0.5)
    x, y = pose.to_odom(0.8, -0.3)
    assert pose.to_local(x, y) == pytest.approx((0.8, -0.3))


def test_pose_integrates_forward_and_turning_motion():
    pose = Pose2D()
    pose.integrate(0.2, 0.0, 1.0)
    assert (pose.x, pose.y, pose.h) == pytest.approx((0.2, 0.0, 0.0))
    pose.integrate(0.0, math.pi / 2, 1.0)
    assert pose.h == pytest.approx(math.pi / 2)
    assert pose.to_odom(1.0, 0.0) == pytest.approx((0.2, 1.0))


def test_histogram_distance():
    assert hist_distance(onehot(3), onehot(3)) == pytest.approx(0.0)
    assert hist_distance(onehot(3), onehot(40)) == pytest.approx(1.0)
    assert hist_distance(None, onehot(3)) == 0.0
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: FAIL; the output contains `No module named 'follow_core'`.

- [ ] **Step 5: Write `scripts/follow_core.py`**

Create `scripts/follow_core.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: PASS — `3 passed`.

- [ ] **Step 7: Run the whole suite (nothing else may break)**

Run: `uv run --extra dev python -m pytest -q`

Expected: PASS — `34 passed`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml tests/conftest.py scripts/follow_core.py tests/test_follow_core.py
git commit -m "feat(follow): core types, config, and geometry" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Lock-on and tracker

**Files:**
- Modify: `scripts/follow_core.py` (append)
- Test: `tests/test_follow_core.py` (append; widen the import)

**Interfaces:**
- Consumes: `FollowConfig`, `PersonObservation`, `Pose2D`, `Track`, `hist_distance` (Task 1).
- Produces: `ConstantVelocityKF(x, y, t, cfg)` with `predict(t)`, `mahalanobis(z)`, `update(z)`, `forget_velocity()`;
  `LockOn(cfg).update(t, people, positions) -> (PersonObservation, (x, y)) | None` and `.reset()`;
  `Tracker(cfg)` with `.locked`, `start(t, xy, hist)`, `update(t, people, positions) -> "updated" | "coasted" | "ambiguous"`,
  `mark_lost()`, `age(t)`, `reset()`, and `track(t, pose) -> Track` (does not modify the filter).
  `positions` are odometry-frame `(x, y)` tuples aligned with `people`.

Lock-on picks exactly one person holding a hand up for 0.5 s (two at once lock nobody). The tracker then follows that person with a constant-velocity Kalman filter in the odometry frame, rejects people whose clothing histogram differs, and refuses to guess between look-alikes.

- [ ] **Step 1: Widen the test import**

In `tests/test_follow_core.py`, replace:

```python
from follow_core import (
    FollowConfig,
    PersonObservation,
    Pose2D,
    hist_distance,
)
```

with:

```python
from follow_core import (
    FollowConfig,
    LockOn,
    PersonObservation,
    Pose2D,
    Tracker,
    hist_distance,
)
```

- [ ] **Step 2: Append the failing tests**

Append to the end of `tests/test_follow_core.py` (two blank lines before it):

```python
# --- lock-on --------------------------------------------------------------

def run_lock_on(frames):
    """frames: list of people lists at 15 Hz. Returns (time, result) of the first lock or None."""
    lock_on = LockOn(CFG)
    for i, people in enumerate(frames):
        t = i * FRAME
        positions = [(p.forward, p.left) for p in people]
        result = lock_on.update(t, people, positions)
        if result is not None:
            return t, result
    return None


def test_hand_raised_for_half_a_second_locks_that_person():
    frames = [[person(1.2, 0.1, raised=True), person(1.5, -0.6)] for _ in range(15)]
    locked = run_lock_on(frames)
    assert locked is not None
    t, (obs, xy) = locked
    assert t >= CFG.lock_window
    assert xy == pytest.approx((1.2, 0.1))


def test_brief_raise_does_not_lock():
    frames = [[person(1.2, raised=i < 4)] for i in range(20)]
    assert run_lock_on(frames) is None


def test_two_people_raising_hands_lock_neither():
    frames = [[person(1.2, 0.5, raised=True), person(1.2, -0.5, raised=True)] for _ in range(20)]
    assert run_lock_on(frames) is None


def test_people_outside_lock_zone_are_ignored():
    far = [[person(3.0, raised=True)] for _ in range(20)]
    wide = [[person(0.5, 1.2, raised=True)] for _ in range(20)]
    assert run_lock_on(far) is None
    assert run_lock_on(wide) is None


# --- tracker --------------------------------------------------------------

def test_tracker_estimates_walking_speed():
    tracker = Tracker(CFG)
    tracker.start(0.0, (1.0, 0.0), onehot(3))
    for i in range(1, 46):
        t = i * FRAME
        assert tracker.update(t, [person(1.0 + 0.3 * t, hist=onehot(3))], [(1.0 + 0.3 * t, 0.0)]) == "updated"
    view = tracker.track(45 * FRAME, Pose2D())
    assert view.v_radial == pytest.approx(0.3, abs=0.05)
    assert view.range == pytest.approx(1.0 + 0.3 * 3.0, abs=0.05)


def test_bystander_in_different_clothes_is_not_accepted():
    tracker = Tracker(CFG)
    tracker.start(0.0, (1.0, 0.0), onehot(3))
    bystander = person(1.02, 0.02, hist=onehot(40))
    assert tracker.update(FRAME, [bystander], [(1.02, 0.02)]) == "coasted"
    assert tracker.age(FRAME) == pytest.approx(FRAME)


def test_two_lookalikes_inside_the_gate_are_ambiguous():
    tracker = Tracker(CFG)
    tracker.start(0.0, (1.0, 0.0), onehot(3))
    people = [person(1.0, 0.05, hist=onehot(3)), person(1.0, -0.05, hist=onehot(3))]
    assert tracker.update(FRAME, people, [(1.0, 0.05), (1.0, -0.05)]) == "ambiguous"


def test_track_view_does_not_modify_the_filter():
    tracker = Tracker(CFG)
    tracker.start(0.0, (1.0, 0.0), None)
    before = tracker.kf.s.copy()
    tracker.track(5.0, Pose2D())
    assert np.array_equal(tracker.kf.s, before)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: FAIL; the output contains `cannot import name 'LockOn'`.

- [ ] **Step 4: Append the implementation to `scripts/follow_core.py`**

Append to the end of `scripts/follow_core.py` (two blank lines before it):

```python
class ConstantVelocityKF:
    """(x, y, vx, vy) in the odometry frame, observed as (x, y)."""

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def __init__(self, x, y, t, cfg, vel_sigma=0.5):
        self.cfg = cfg
        self.s = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([cfg.meas_sigma**2] * 2 + [vel_sigma**2] * 2)
        self.R = np.eye(2) * cfg.meas_sigma**2
        self.t = t

    def predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt * dt, 0.0], [0.0, 0.5 * dt * dt], [dt, 0.0], [0.0, dt]])
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + G @ G.T * self.cfg.accel_sigma**2
        self.t = t
        self._cap_position_variance()

    def _cap_position_variance(self):
        # Keeps the association gate bounded during long coasts.
        limit = self.cfg.max_pos_sigma**2
        for i in (0, 1):
            if self.P[i, i] > limit:
                k = math.sqrt(limit / self.P[i, i])
                self.P[i, :] *= k
                self.P[:, i] *= k

    def _innovation(self, z):
        y = np.asarray(z, dtype=float) - self.H @ self.s
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis(self, z):
        y, S = self._innovation(z)
        return float(math.sqrt(y @ np.linalg.solve(S, y)))

    def update(self, z):
        y, S = self._innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def forget_velocity(self, vel_sigma=0.5):
        pos_var = self.cfg.lost_pos_sigma**2
        self.s[2:] = 0.0
        self.P = np.diag([self.P[0, 0] + pos_var, self.P[1, 1] + pos_var, vel_sigma**2, vel_sigma**2])
        self._cap_position_variance()


class LockOn:
    """Chooses the single person who keeps a hand raised for ``lock_window`` seconds."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.candidates = []

    def reset(self):
        self.candidates = []

    def update(self, t, people, positions):
        """Returns ``(observation, odom_xy)`` of the locked person, or None."""
        cfg = self.cfg
        used = set()
        for obs, xy in zip(people, positions):
            if not (cfg.lock_range_min <= obs.range <= cfg.lock_range_max):
                continue
            if abs(obs.bearing) > cfg.lock_bearing_max:
                continue
            best, best_d = None, cfg.lock_assoc_dist
            for i, cand in enumerate(self.candidates):
                d = math.dist(xy, cand["xy"])
                if i not in used and d <= best_d:
                    best, best_d = i, d
            if best is None:
                self.candidates.append({"xy": xy, "first_t": t, "samples": [], "obs": obs})
                best = len(self.candidates) - 1
            cand = self.candidates[best]
            cand["xy"], cand["obs"] = xy, obs
            cand["samples"].append((t, obs.hand_raised))
            used.add(best)

        horizon = t - cfg.lock_window
        self.candidates = [c for c in self.candidates if c["samples"][-1][0] >= horizon]
        qualified = []
        for cand in self.candidates:
            cand["samples"] = [s for s in cand["samples"] if s[0] >= horizon]
            raised = [r for _, r in cand["samples"]]
            if t - cand["first_t"] >= cfg.lock_window and sum(raised) >= cfg.lock_fraction * len(raised):
                qualified.append(cand)
        if len(qualified) != 1:
            return None  # nobody yet, or two people at once: keep waiting
        return qualified[0]["obs"], qualified[0]["xy"]


class Tracker:
    """Follows the locked person frame to frame, refusing to guess between look-alikes."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.kf = None
        self.ref_hist = None
        self.last_update = None

    @property
    def locked(self):
        return self.kf is not None

    def start(self, t, xy, hist):
        self.kf = ConstantVelocityKF(xy[0], xy[1], t, self.cfg)
        self.ref_hist = None if hist is None else np.asarray(hist, dtype=float)
        self.last_update = t

    def mark_lost(self):
        self.kf.forget_velocity()

    def age(self, t):
        return t - self.last_update

    def update(self, t, people, positions):
        """Returns "updated", "coasted" (no match), or "ambiguous" (two close matches)."""
        cfg = self.cfg
        self.kf.predict(t)
        candidates = []
        for obs, xy in zip(people, positions):
            d = self.kf.mahalanobis(xy)
            h = hist_distance(self.ref_hist, obs.hist)
            if d > cfg.gate_sigma or h > cfg.hist_max_distance:
                continue
            score = 0.5 * d / cfg.gate_sigma + 0.5 * h / cfg.hist_max_distance
            candidates.append((score, obs, xy))
        if not candidates:
            return "coasted"
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[1][0] <= candidates[0][0] * (1 + cfg.ambiguity_ratio):
            return "ambiguous"
        _, obs, xy = candidates[0]
        self.kf.update(xy)
        self.last_update = t
        if obs.hist is not None:
            hist = np.asarray(obs.hist, dtype=float)
            if self.ref_hist is None:
                self.ref_hist = hist
            else:
                blended = (1 - cfg.hist_alpha) * self.ref_hist + cfg.hist_alpha * hist
                self.ref_hist = blended / blended.sum()
        return "updated"

    def track(self, t, pose):
        """Robot-relative view of the person at time ``t``; does not modify the filter."""
        dt = max(0.0, t - self.kf.t)
        x = self.kf.s[0] + self.kf.s[2] * dt
        y = self.kf.s[1] + self.kf.s[3] * dt
        forward, left = pose.to_local(x, y)
        rng = math.hypot(forward, left)
        if rng > 1e-6:
            v_radial = (self.kf.s[2] * (x - pose.x) + self.kf.s[3] * (y - pose.y)) / rng
        else:
            v_radial = 0.0
        return Track(forward, left, rng, math.atan2(left, forward), float(v_radial), self.age(t))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: PASS — `11 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/follow_core.py tests/test_follow_core.py
git commit -m "feat(follow): raise-hand lock-on and Kalman person tracker" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Controller, rate limiter, obstacle corridor, supervisor, start checks

**Files:**
- Modify: `scripts/follow_core.py` (append)
- Test: `tests/test_follow_core.py` (append; widen the import)

**Interfaces:**
- Consumes: `FollowConfig`, `Track`, `clamp`, `shrink` (Task 1).
- Produces: `follow_command(track, gap, cfg) -> (v, omega)` (v never negative); `RateLimiter(cfg).step(v, omega, dt) -> (v, omega)` with `.v`, `.omega`, `.reset()`;
  `corridor_count(points, person_forward_left | None, cfg) -> int`; `CorridorGuard(cfg).update(t, count) -> bool` with `.blocked`;
  `OdometryCheck(cfg).update(t, v_sent, omega_sent, v_meas, omega_meas) -> bool`;
  `Verdict(v, omega, rule, exit=False)`; `supervise(cfg, *, v, omega, stop_requested, heartbeat_age, roll_deg, pitch_deg, odom_mismatch, tracking, track_age, points_age, blocked, range_m) -> Verdict`;
  `start_refusal(cfg, *, roll_deg, pitch_deg, voltage, low_battery_v, drive_writers, points_fresh) -> str | None`.

These are the safety-critical pure functions: spec §6.3 (controller), §6.4 (supervisor table, rule names are part of the status protocol), §6.5 (corridor), and the start preconditions.

- [ ] **Step 1: Widen the test import**

In `tests/test_follow_core.py`, replace:

```python
from follow_core import (
    FollowConfig,
    LockOn,
    PersonObservation,
    Pose2D,
    Tracker,
    hist_distance,
)
```

with:

```python
from follow_core import (
    CorridorGuard,
    FollowConfig,
    LockOn,
    OdometryCheck,
    PersonObservation,
    Pose2D,
    RateLimiter,
    Track,
    Tracker,
    corridor_count,
    follow_command,
    hist_distance,
    start_refusal,
    supervise,
)
```

- [ ] **Step 2: Append the failing tests**

Append to the end of `tests/test_follow_core.py` (two blank lines before it):

```python
def track(range_m, bearing_deg=0.0, v_radial=0.0, age=0.05):
    b = math.radians(bearing_deg)
    return Track(range_m * math.cos(b), range_m * math.sin(b), range_m, b, v_radial, age)


def safe_inputs(**overrides):
    values = dict(
        v=0.1, omega=0.1, stop_requested=False, heartbeat_age=0.1, roll_deg=0.0,
        pitch_deg=2.0, odom_mismatch=False, tracking=True, track_age=0.05,
        points_age=0.05, blocked=False, range_m=1.0,
    )
    values.update(overrides)
    return values


# --- controller -----------------------------------------------------------

def test_inside_deadband_the_robot_holds_still():
    assert follow_command(track(1.04), 1.0, FAST) == (0.0, 0.0)


def test_range_error_drives_forward_proportionally():
    v, omega = follow_command(track(1.25), 1.0, FAST)
    assert v == pytest.approx(0.8 * 0.20)
    assert omega == 0.0


def test_person_speed_is_fed_forward():
    v, _ = follow_command(track(1.0, v_radial=0.12), 1.0, FAST)
    assert v == pytest.approx(0.12)


def test_never_reverses_when_the_person_comes_closer():
    v, _ = follow_command(track(0.6, v_radial=-0.3), 1.0, FAST)
    assert v == 0.0


def test_turns_in_place_beyond_35_degrees():
    v, omega = follow_command(track(1.5, bearing_deg=40), 1.0, FAST)
    assert v == 0.0
    assert omega == FAST.omega_max  # positive: turn left toward the person


def test_speeds_are_clamped():
    v, omega = follow_command(track(3.0, bearing_deg=-60), 1.0, FAST)
    assert v == 0.0
    assert omega == -FAST.omega_max
    v, _ = follow_command(track(3.0), 1.0, CFG)
    assert v == CFG.v_max


def test_rate_limiter_uses_separate_accel_brake_and_turn_limits():
    limiter = RateLimiter(CFG)
    assert limiter.step(0.3, 1.0, 0.02) == pytest.approx((0.008, 0.03))
    limiter.v = 0.3
    assert limiter.step(0.0, 0.03, 0.02)[0] == pytest.approx(0.3 - 0.016)


# --- obstacle corridor ----------------------------------------------------

def test_corridor_ignores_floor_person_and_self_and_counts_obstacles():
    floor = np.column_stack([np.linspace(0.1, 0.6, 50), np.zeros(50), np.zeros(50)])
    box = np.column_stack([np.full(40, 0.5), np.linspace(-0.1, 0.1, 40), np.full(40, 0.2)])
    body = np.column_stack([np.full(40, 0.05), np.zeros(40), np.full(40, 1.0)])
    cfg = dataclasses.replace(CFG, self_mask=((0.0, 0.1, -0.2, 0.2, 0.3, 1.6),))
    points = np.vstack([floor, box, body])
    assert corridor_count(points, None, cfg) == 40
    assert corridor_count(points, (0.55, 0.0), cfg) == 0  # the box is where the person is
    assert corridor_count(np.empty((0, 3)), None, cfg) == 0


def test_corridor_blocks_at_once_and_clears_with_hysteresis():
    guard = CorridorGuard(CFG)
    assert guard.update(0.0, 30) is True
    assert guard.update(0.1, 0) is True
    assert guard.update(0.55, 0) is True
    assert guard.update(0.61, 0) is False


def test_odometry_check_needs_a_sustained_opposite_sign():
    check = OdometryCheck(CFG)
    assert check.update(0.0, 0.2, 0.0, -0.1, 0.0) is False
    assert check.update(0.3, 0.2, 0.0, -0.1, 0.0) is False
    assert check.update(0.6, 0.2, 0.0, -0.1, 0.0) is True
    assert check.update(0.7, 0.2, 0.0, 0.1, 0.0) is False


# --- supervisor -----------------------------------------------------------

@pytest.mark.parametrize("override, rule", [
    ({"stop_requested": True}, "stop"),
    ({"heartbeat_age": 1.2}, "heartbeat"),
    ({"roll_deg": 26.0}, "not-upright"),
    ({"pitch_deg": -25.0}, "not-upright"),
    ({"odom_mismatch": True}, "odometry-mismatch"),
])
def test_exit_rules(override, rule):
    verdict = supervise(CFG, **safe_inputs(**override))
    assert (verdict.v, verdict.omega, verdict.rule, verdict.exit) == (0.0, 0.0, rule, True)


def test_exit_rule_priority():
    verdict = supervise(CFG, **safe_inputs(heartbeat_age=5.0, roll_deg=40.0, blocked=True))
    assert verdict.rule == "heartbeat"


@pytest.mark.parametrize("override, rule, omega", [
    ({"tracking": False, "track_age": None, "range_m": None}, "no-track", 0.0),
    ({"points_age": 0.5}, "points-stale", 0.1),
    ({"track_age": 0.4}, "track-stale", 0.0),
    ({"blocked": True}, "blocked", 0.1),
    ({"range_m": 0.4}, "min-range", 0.1),
])
def test_restriction_rules(override, rule, omega):
    verdict = supervise(CFG, **safe_inputs(**override))
    assert (verdict.v, verdict.omega, verdict.rule, verdict.exit) == (0.0, omega, rule, False)


def test_supervisor_passes_safe_commands_and_never_reverses():
    assert supervise(CFG, **safe_inputs()).rule == "ok"
    assert supervise(CFG, **safe_inputs(v=0.1)).v == pytest.approx(0.1)
    assert supervise(CFG, **safe_inputs(v=-0.2)).v == 0.0
    verdict = supervise(CFG, **safe_inputs(v=0.9, omega=-3.0))
    assert (verdict.v, verdict.omega) == (CFG.v_max, -CFG.omega_max)


def test_start_refusal():
    ok = dict(roll_deg=0.0, pitch_deg=3.0, voltage=24.0, low_battery_v=21.0,
              drive_writers=[], points_fresh=True)
    assert start_refusal(CFG, **ok) is None
    assert start_refusal(CFG, **{**ok, "pitch_deg": 30.0}) == "robot is not upright"
    assert "depth daemon" in start_refusal(CFG, **{**ok, "points_fresh": False})
    assert "greeter" in start_refusal(CFG, **{**ok, "drive_writers": ["123 python greeter/main.py"]})
    assert start_refusal(CFG, **{**ok, "voltage": 20.0}) == "battery low (20.0 V < 21.0 V)"
    assert start_refusal(CFG, **{**ok, "voltage": None}) is None
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: FAIL; the output contains `cannot import name 'CorridorGuard'`.

- [ ] **Step 4: Append the implementation to `scripts/follow_core.py`**

Append to the end of `scripts/follow_core.py` (two blank lines before it):

```python
def follow_command(track, gap, cfg):
    """(v, omega) that holds ``gap`` to the tracked person. v is never negative."""
    v = max(track.v_radial, 0.0) + cfg.k_r * shrink(track.range - gap, cfg.deadband_range)
    v *= math.cos(track.bearing)
    if abs(track.bearing) > cfg.turn_in_place_bearing:
        v = 0.0  # face the person before driving
    omega = cfg.k_theta * shrink(track.bearing, cfg.deadband_bearing)
    return clamp(v, 0.0, cfg.v_max), clamp(omega, -cfg.omega_max, cfg.omega_max)


class RateLimiter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.v = 0.0
        self.omega = 0.0

    def step(self, v, omega, dt):
        cfg = self.cfg
        self.v += clamp(v - self.v, -cfg.accel_down * dt, cfg.accel_up * dt)
        self.omega += clamp(omega - self.omega, -cfg.alpha_max * dt, cfg.alpha_max * dt)
        return self.v, self.omega


def corridor_count(points, person, cfg):
    """Points in the drive corridor that are not floor, the robot itself, or the person."""
    if points is None or len(points) == 0:
        return 0
    f, l, z = points[:, 0], points[:, 1], points[:, 2]
    half = cfg.robot_width / 2 + cfg.corridor_margin
    keep = (
        (f > 0) & (f <= cfg.corridor_length) & (np.abs(l) <= half)
        & (z >= cfg.corridor_z_min) & (z <= cfg.corridor_z_max)
    )
    if person is not None:
        keep &= np.hypot(f - person[0], l - person[1]) > cfg.person_exclusion_radius
    for f0, f1, l0, l1, z0, z1 in cfg.self_mask:
        keep &= ~((f >= f0) & (f <= f1) & (l >= l0) & (l <= l1) & (z >= z0) & (z <= z1))
    return int(np.count_nonzero(keep))


class CorridorGuard:
    """Blocks at once; clears only after ``corridor_clear_time`` below the threshold."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.blocked = False
        self._clear_since = None

    def update(self, t, count):
        if count >= self.cfg.corridor_min_points:
            self.blocked = True
            self._clear_since = None
        elif self.blocked:
            if self._clear_since is None:
                self._clear_since = t
            if t - self._clear_since >= self.cfg.corridor_clear_time:
                self.blocked = False
                self._clear_since = None
        return self.blocked


class OdometryCheck:
    """Trips when wheel feedback keeps moving opposite to what was sent (a sign bug)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._since = None

    def update(self, t, v_sent, omega_sent, v_meas, omega_meas):
        wrong = (
            (abs(v_sent) >= 0.05 and abs(v_meas) >= 0.03 and v_meas * v_sent < 0)
            or (abs(omega_sent) >= 0.2 and abs(omega_meas) >= 0.1 and omega_meas * omega_sent < 0)
        )
        if not wrong:
            self._since = None
            return False
        if self._since is None:
            self._since = t
        return t - self._since >= self.cfg.odom_mismatch_time


@dataclass(frozen=True)
class Verdict:
    v: float
    omega: float
    rule: str
    exit: bool = False


def supervise(cfg, *, v, omega, stop_requested, heartbeat_age, roll_deg, pitch_deg,
              odom_mismatch, tracking, track_age, points_age, blocked, range_m):
    """Final say over every command. Exit rules first, then restrictions (all applied)."""
    if stop_requested:
        return Verdict(0.0, 0.0, "stop", True)
    if heartbeat_age > cfg.heartbeat_timeout:
        return Verdict(0.0, 0.0, "heartbeat", True)
    if abs(roll_deg) >= cfg.upright_deg or abs(pitch_deg) >= cfg.upright_deg:
        return Verdict(0.0, 0.0, "not-upright", True)
    if odom_mismatch:
        return Verdict(0.0, 0.0, "odometry-mismatch", True)

    fired = []
    if not tracking:
        v = omega = 0.0
        fired.append("no-track")
    if points_age > cfg.points_stale:
        v = 0.0  # obstacle state unknown: turning in place is still allowed
        fired.append("points-stale")
    if track_age is not None and track_age > cfg.perception_stale:
        v = omega = 0.0
        fired.append("track-stale")
    if blocked:
        v = 0.0
        fired.append("blocked")
    if range_m is not None and range_m < cfg.min_range:
        v = 0.0
        fired.append("min-range")
    return Verdict(
        clamp(v, 0.0, cfg.v_max),
        clamp(omega, -cfg.omega_max, cfg.omega_max),
        fired[0] if fired else "ok",
    )


def start_refusal(cfg, *, roll_deg, pitch_deg, voltage, low_battery_v, drive_writers, points_fresh):
    """Why the runner must not start, or None. ``voltage``/``low_battery_v`` may be None (unknown)."""
    if abs(roll_deg) >= cfg.upright_deg or abs(pitch_deg) >= cfg.upright_deg:
        return "robot is not upright"
    if not points_fresh:
        return "camera.points is not publishing; start the depth daemon"
    if drive_writers:
        return "another app is driving: " + "; ".join(drive_writers)
    if voltage is not None and low_battery_v is not None and voltage < low_battery_v:
        return f"battery low ({voltage:.1f} V < {low_battery_v:.1f} V)"
    return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: PASS — `34 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/follow_core.py tests/test_follow_core.py
git commit -m "feat(follow): controller, corridor guard, supervisor, start checks" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `FollowLoop`, dashboard protocol, LED, and unit helpers

**Files:**
- Modify: `scripts/follow_core.py` (append)
- Test: `tests/test_follow_core.py` (append; widen the import)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `TickInputs(t, heartbeat_age, stop_requested, roll_deg, pitch_deg, measured_v, measured_omega, perception=None)`;
  `TickOutput(t, state, v, omega, v_cmd, omega_cmd, rule, exit, gap, range, bearing, blocked, corridor_points, track_age)` with `.error`;
  `FollowLoop(cfg=None, gap=None)` with `set_gap(gap) -> float` (clamped) and `tick(TickInputs) -> TickOutput`;
  `Command(kind, gap=None)`; `parse_command(line, cfg) -> Command | None`; `status_line(TickOutput) -> str`
  (`STATUS_PREFIX` + JSON with keys `state range gap error bearing_deg v w blocked age_ms rule`);
  `led_color(state, elapsed) -> (r, g, b)`; `timestamp_to_seconds(value) -> float`;
  `wheel_twist(turns_per_s, wheel_diam, robot_width, wheel_signs=(1.0, 1.0)) -> (v, omega)`.

`FollowLoop.tick` is the only thing the robot runner calls per 20 ms tick; the simulation test (Task 5) drives the same object. The state machine is spec §6.6.

- [ ] **Step 1: Widen the test import**

In `tests/test_follow_core.py`, replace:

```python
from follow_core import (
    CorridorGuard,
    FollowConfig,
    LockOn,
    OdometryCheck,
    PersonObservation,
    Pose2D,
    RateLimiter,
    Track,
    Tracker,
    corridor_count,
    follow_command,
    hist_distance,
    start_refusal,
    supervise,
)
```

with:

```python
from follow_core import (
    BLOCKED,
    FOLLOWING,
    SEARCHING,
    STATUS_PREFIX,
    Command,
    CorridorGuard,
    FollowConfig,
    FollowLoop,
    LockOn,
    OdometryCheck,
    Perception,
    PersonObservation,
    Pose2D,
    RateLimiter,
    TickInputs,
    Track,
    Tracker,
    corridor_count,
    follow_command,
    hist_distance,
    led_color,
    parse_command,
    start_refusal,
    status_line,
    supervise,
    timestamp_to_seconds,
    wheel_twist,
)
```

- [ ] **Step 2: Append the failing tests**

Append to the end of `tests/test_follow_core.py` (two blank lines before it):

```python
# --- loop -----------------------------------------------------------------

def tick_inputs(t, **overrides):
    values = dict(t=t, heartbeat_age=0.1, stop_requested=False, roll_deg=0.0, pitch_deg=0.0,
                  measured_v=0.0, measured_omega=0.0, perception=None)
    values.update(overrides)
    return TickInputs(**values)


def test_loop_holds_still_while_searching():
    loop = FollowLoop(CFG)
    frame = Perception(0.0, (person(1.5, 0.3),), np.empty((0, 3)))
    out = loop.tick(tick_inputs(0.0, perception=frame))
    assert (out.state, out.v, out.omega, out.rule) == (SEARCHING, 0.0, 0.0, "no-track")


def test_loop_locks_on_then_follows():
    loop = FollowLoop(FAST)
    out = None
    for i in range(100):
        t = i * 0.02
        frame = None
        if i % 3 == 0:
            frame = Perception(t, (person(1.6, raised=t < 1.0),), np.empty((0, 3)))
        out = loop.tick(tick_inputs(t, perception=frame))
    assert out.state == FOLLOWING
    assert out.v > 0.0


def test_loop_exits_on_heartbeat_loss():
    out = FollowLoop(CFG).tick(tick_inputs(0.0, heartbeat_age=1.5))
    assert out.exit is True
    assert (out.v, out.omega, out.rule) == (0.0, 0.0, "heartbeat")


def test_gap_is_clamped():
    loop = FollowLoop(CFG)
    assert loop.set_gap(3.0) == 1.5
    assert loop.set_gap(0.1) == 0.6


# --- protocol and helpers -------------------------------------------------

def test_parse_command():
    assert parse_command('{"type":"heartbeat"}', CFG) == Command("heartbeat")
    assert parse_command('{"type":"stop"}', CFG) == Command("stop")
    assert parse_command('{"type":"gap","m":1.2}', CFG) == Command("gap", 1.2)
    assert parse_command('{"type":"gap","m":9}', CFG) == Command("gap", 1.5)
    for bad in ("", "nope", "[]", '{"type":"gap","m":true}', '{"type":"gap","m":NaN}',
                '{"type":"gap"}', '{"type":"drive","v":1}'):
        assert parse_command(bad, CFG) is None, bad


def test_status_line_is_prefixed_json():
    loop = FollowLoop(CFG)
    out = loop.tick(tick_inputs(0.0))
    line = status_line(out)
    assert line.startswith(STATUS_PREFIX)
    payload = json.loads(line[len(STATUS_PREFIX):])
    assert payload["state"] == SEARCHING
    assert payload["gap"] == 1.0
    assert payload["range"] is None
    assert set(payload) == {"state", "range", "gap", "error", "bearing_deg", "v", "w",
                            "blocked", "age_ms", "rule"}


def test_led_patterns():
    assert led_color(FOLLOWING, 0.0) == (70, 220, 120)
    assert led_color(BLOCKED, 5.0) == (255, 160, 0)
    assert led_color("LOST", 0.0) == (255, 160, 0)
    assert led_color("LOST", 0.4) == (20, 13, 0)
    assert led_color(SEARCHING, 0.0) == (14, 25, 51)


def test_timestamp_units_are_normalised():
    for value in (1_760_000_000.5, 1_760_000_000_500, 1_760_000_000_500_000, 1_760_000_000_500_000_000):
        assert timestamp_to_seconds(value) == pytest.approx(1_760_000_000.5)


def test_wheel_twist():
    v, omega = wheel_twist((1.0, 1.0), 0.165, 0.3275)
    assert (v, omega) == pytest.approx((math.pi * 0.165, 0.0))
    v, omega = wheel_twist((-1.0, 1.0), 0.165, 0.3275)
    assert v == pytest.approx(0.0)
    assert omega == pytest.approx(2 * math.pi * 0.165 / 0.3275)
    assert wheel_twist((1.0, 1.0), 0.165, 0.3275, (-1.0, -1.0))[0] < 0
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: FAIL; the output contains `cannot import name 'Command'`.

- [ ] **Step 4: Append the implementation to `scripts/follow_core.py`**

Append to the end of `scripts/follow_core.py` (two blank lines before it):

```python
@dataclass(frozen=True, eq=False)
class TickInputs:
    t: float  # loop clock (s)
    heartbeat_age: float
    stop_requested: bool
    roll_deg: float
    pitch_deg: float
    measured_v: float  # from wheel feedback (m/s)
    measured_omega: float  # rad/s
    perception: Perception | None = None  # a new frame since the last tick, if any


@dataclass(frozen=True)
class TickOutput:
    t: float
    state: str
    v: float  # what to send
    omega: float
    v_cmd: float  # what the controller asked for
    omega_cmd: float
    rule: str
    exit: bool
    gap: float
    range: float | None
    bearing: float | None
    blocked: bool
    corridor_points: int
    track_age: float | None

    @property
    def error(self):
        return None if self.range is None else self.range - self.gap


class FollowLoop:
    """Composes lock-on, tracking, control, and supervision. One call per control tick."""

    def __init__(self, cfg=None, gap=None):
        self.cfg = cfg or FollowConfig()
        self.gap = self.cfg.gap_default
        if gap is not None:
            self.set_gap(gap)
        self.pose = Pose2D()
        self.lock_on = LockOn(self.cfg)
        self.tracker = Tracker(self.cfg)
        self.corridor = CorridorGuard(self.cfg)
        self.limiter = RateLimiter(self.cfg)
        self.odom_check = OdometryCheck(self.cfg)
        self.state = SEARCHING
        self.lost_since = None
        self.last_t = None
        self.last_points_t = None
        self.corridor_points = 0

    def set_gap(self, gap):
        self.gap = clamp(float(gap), self.cfg.gap_min, self.cfg.gap_max)
        return self.gap

    def tick(self, inp):
        cfg = self.cfg
        dt = 0.0 if self.last_t is None else max(0.0, inp.t - self.last_t)
        self.last_t = inp.t
        self.pose.integrate(inp.measured_v, inp.measured_omega, dt)
        mismatch = self.odom_check.update(
            inp.t, self.limiter.v, self.limiter.omega, inp.measured_v, inp.measured_omega
        )
        if inp.perception is not None:
            self._perceive(inp.perception)
        self._advance_state(inp.t)

        tracking = self.state in (FOLLOWING, BLOCKED)
        track = self.tracker.track(inp.t, self.pose) if tracking else None
        v_cmd, omega_cmd = follow_command(track, self.gap, cfg) if track else (0.0, 0.0)
        points_age = math.inf if self.last_points_t is None else inp.t - self.last_points_t
        verdict = supervise(
            cfg, v=v_cmd, omega=omega_cmd,
            stop_requested=inp.stop_requested, heartbeat_age=inp.heartbeat_age,
            roll_deg=inp.roll_deg, pitch_deg=inp.pitch_deg, odom_mismatch=mismatch,
            tracking=tracking, track_age=track.age if track else None,
            points_age=points_age, blocked=self.corridor.blocked,
            range_m=track.range if track else None,
        )
        if verdict.exit:
            self.limiter.reset()
            v, omega = 0.0, 0.0
        else:
            v, omega = self.limiter.step(verdict.v, verdict.omega, dt)
        return TickOutput(
            t=inp.t, state=self.state, v=v, omega=omega, v_cmd=v_cmd, omega_cmd=omega_cmd,
            rule=verdict.rule, exit=verdict.exit, gap=self.gap,
            range=track.range if track else None, bearing=track.bearing if track else None,
            blocked=self.corridor.blocked, corridor_points=self.corridor_points,
            track_age=track.age if track else None,
        )

    def _perceive(self, frame):
        positions = [self.pose.to_odom(o.forward, o.left) for o in frame.people]
        if self.state == SEARCHING:
            locked = self.lock_on.update(frame.t, frame.people, positions)
            if locked is not None:
                obs, xy = locked
                self.tracker.start(frame.t, xy, obs.hist)
                self.lock_on.reset()
                self.state = FOLLOWING
        elif self.tracker.update(frame.t, frame.people, positions) == "updated" and self.state == LOST:
            self.state = FOLLOWING
            self.lost_since = None
        person = None
        if self.tracker.locked:
            track = self.tracker.track(frame.t, self.pose)
            person = (track.forward, track.left)
        self.corridor_points = corridor_count(frame.points, person, self.cfg)
        self.corridor.update(frame.t, self.corridor_points)
        self.last_points_t = frame.t

    def _advance_state(self, t):
        if self.state in (FOLLOWING, BLOCKED):
            if self.tracker.age(t) > self.cfg.lost_after:
                self.state = LOST
                self.lost_since = t
                self.tracker.mark_lost()
            else:
                self.state = BLOCKED if self.corridor.blocked else FOLLOWING
        elif self.state == LOST and t - self.lost_since > self.cfg.lost_timeout:
            self.state = SEARCHING
            self.lost_since = None
            self.tracker.reset()
            self.lock_on.reset()


@dataclass(frozen=True)
class Command:
    kind: str  # "heartbeat" | "gap" | "stop"
    gap: float | None = None


def parse_command(line, cfg):
    """One dashboard stdin line -> Command, or None if malformed. Gaps are clamped."""
    try:
        message = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(message, dict):
        return None
    kind = message.get("type")
    if kind in ("heartbeat", "stop"):
        return Command(kind)
    if kind == "gap":
        value = message.get("m")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return Command("gap", clamp(float(value), cfg.gap_min, cfg.gap_max))
    return None


def status_line(out):
    def rounded(value, digits=3):
        return None if value is None else round(value, digits)

    payload = {
        "state": out.state,
        "range": rounded(out.range),
        "gap": round(out.gap, 2),
        "error": rounded(out.error),
        "bearing_deg": rounded(None if out.bearing is None else math.degrees(out.bearing), 1),
        "v": round(out.v, 3),
        "w": round(out.omega, 3),
        "blocked": out.blocked,
        "age_ms": None if out.track_age is None else round(out.track_age * 1000),
        "rule": out.rule,
    }
    return STATUS_PREFIX + json.dumps(payload, separators=(",", ":"))


STATE_LED = {
    SEARCHING: ((70, 125, 255), "pulse"),
    FOLLOWING: ((70, 220, 120), "solid"),
    BLOCKED: ((255, 160, 0), "solid"),
    LOST: ((255, 160, 0), "blink"),
}


def led_color(state, elapsed):
    """RGB for the state's LED pattern ``elapsed`` seconds into it (same shapes as robot_effect.py)."""
    rgb, pattern = STATE_LED[state]
    if pattern == "solid":
        scale = 1.0
    elif pattern == "blink":
        scale = 1.0 if int(elapsed * 3) % 2 == 0 else 0.08
    else:
        scale = 0.2 + 0.8 * (0.5 - 0.5 * math.cos(2 * math.pi * elapsed / 1.6))
    return tuple(round(c * scale) for c in rgb)


def timestamp_to_seconds(value):
    """BBOS timestamps have appeared in s, ms, us, and ns; normalise by magnitude."""
    value = float(value)
    if value > 1e17:
        return value / 1e9
    if value > 1e14:
        return value / 1e6
    if value > 1e11:
        return value / 1e3
    return value


def wheel_twist(turns_per_s, wheel_diam, robot_width, wheel_signs=(1.0, 1.0)):
    """drive.state.vel (turns/s, [left, right]) -> (v m/s, omega rad/s)."""
    circumference = math.pi * wheel_diam
    v_left = wheel_signs[0] * float(turns_per_s[0]) * circumference
    v_right = wheel_signs[1] * float(turns_per_s[1]) * circumference
    return (v_left + v_right) / 2.0, (v_right - v_left) / robot_width
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: PASS — `43 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/follow_core.py tests/test_follow_core.py
git commit -m "feat(follow): follow loop state machine and runner protocol" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Closed-loop simulation test

**Files:**
- Test: `tests/test_follow_sim.py`

**Interfaces:**
- Consumes: `FollowLoop`, `TickInputs`, `Perception`, `PersonObservation`, `Pose2D`, `FollowConfig`, state names.
- Produces: nothing new; this is the spec §8.2 acceptance test.

A 2D unicycle robot with a 0.15 s velocity lag, observed at 15 Hz with 100 ms latency and noise, runs every spec scenario. The thresholds were checked on 20 random seeds before this plan was written (95th-percentile gap error ≤ 5 cm up to 0.25 m/s). If a scenario fails, fix `follow_core`; never loosen a threshold to make it pass. Its results are **simulation evidence only**.

- [ ] **Step 1: Write the simulation test**

Create `tests/test_follow_sim.py`:

```python
"""Closed-loop kinematic simulation of FollowLoop.

SIMULATION EVIDENCE ONLY. The robot is a unicycle with a first-order velocity
lag; perception is 15 Hz with 100 ms latency and Gaussian range/bearing noise.
Passing here says the logic and tuning are coherent, not that the robot works.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np
import pytest

from follow_core import (
    BLOCKED, FOLLOWING, LOST, FollowConfig, FollowLoop, Perception, PersonObservation,
    Pose2D, TickInputs,
)

CFG = replace(FollowConfig(), v_max=0.30)  # the configuration targeted after gate G4b
DT = 0.02
FRAME_EVERY = 1 / 15
LATENCY = 0.10
LAG = 0.15  # balancing base velocity response time constant (s)
TARGET_HIST = np.eye(64)[3]
BYSTANDER_HIST = np.eye(64)[40]


def raised_for_first_second(t):
    return t < 1.0


@dataclass
class Scenario:
    target: callable  # t -> (x, y) in the world
    duration: float
    bystander: callable | None = None
    visible: callable = lambda t: True
    obstacle: callable = lambda t: None  # t -> (x, y) of a 0.3 m box in the world, or None
    heartbeat_until: float = math.inf
    seed: int = 0


@dataclass
class Result:
    t: list = field(default_factory=list)
    true_range: list = field(default_factory=list)
    true_bearing: list = field(default_factory=list)
    robot_x: list = field(default_factory=list)
    out: list = field(default_factory=list)

    def samples(self, start, end=math.inf):
        return [
            (t, r, b, o) for t, r, b, o in zip(self.t, self.true_range, self.true_bearing, self.out)
            if start <= t < end
        ]


def box_points(center_local):
    xs, ys, zs = np.meshgrid(
        np.linspace(-0.15, 0.15, 6), np.linspace(-0.15, 0.15, 6), np.linspace(0.05, 0.3, 4)
    )
    local = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    local[:, 0] += center_local[0]
    local[:, 1] += center_local[1]
    return local


def run(scenario, cfg=CFG, gap=1.0):
    rng = np.random.default_rng(scenario.seed)
    loop = FollowLoop(cfg, gap)
    robot = Pose2D()
    v_actual = omega_actual = 0.0
    result = Result()
    next_frame = 0.0
    pending = []  # (deliver_at, Perception)
    steps = int(scenario.duration / DT)
    for i in range(steps):
        t = i * DT
        if t >= next_frame:
            next_frame += FRAME_EVERY
            capture_t = t
            people = []
            actors = [(scenario.target, TARGET_HIST, scenario.visible(t))]
            if scenario.bystander is not None:
                actors.append((scenario.bystander, BYSTANDER_HIST, True))
            for path, hist, visible in actors:
                if not visible:
                    continue
                f, l = robot.to_local(*path(t))
                r = math.hypot(f, l) + rng.normal(0, 0.025)
                b = math.atan2(l, f) + rng.normal(0, math.radians(1.0))
                raised = path is scenario.target and raised_for_first_second(t)
                people.append(PersonObservation(r * math.cos(b), r * math.sin(b), 0.9, raised, hist))
            points = np.empty((0, 3))
            box = scenario.obstacle(t)
            if box is not None:
                points = box_points(robot.to_local(*box))
            pending.append((capture_t + LATENCY, Perception(capture_t, tuple(people), points)))
        frame = None
        if pending and pending[0][0] <= t + 1e-9:
            frame = pending.pop(0)[1]
        out = loop.tick(TickInputs(
            t=t,
            heartbeat_age=0.1 if t < scenario.heartbeat_until else t - scenario.heartbeat_until,
            stop_requested=False, roll_deg=0.0, pitch_deg=1.0,
            measured_v=v_actual + rng.normal(0, 0.005),
            measured_omega=omega_actual + rng.normal(0, 0.01),
            perception=frame,
        ))
        tx, ty = scenario.target(t)
        f, l = robot.to_local(tx, ty)
        result.t.append(t)
        result.true_range.append(math.hypot(f, l))
        result.true_bearing.append(math.atan2(l, f))
        result.robot_x.append(robot.x)
        result.out.append(out)
        if out.exit:
            break
        v_actual += (out.v - v_actual) * DT / LAG
        omega_actual += (out.omega - omega_actual) * DT / LAG
        robot.integrate(v_actual, omega_actual, DT)
    return result


def standing(x, y=0.0):
    return lambda t: (x, y)


def walking_away(speed, start=2.0, x0=1.0):
    return lambda t: (x0 + speed * max(0.0, t - start), 0.0)


def fraction_in_band(samples, gap=1.0, band=0.20):
    inside = sum(abs(r - gap) <= band for _, r, _, _ in samples)
    return inside / len(samples)


def test_standing_person_settles_near_the_gap():
    result = run(Scenario(standing(1.5), duration=12.0))
    final = result.samples(10.0)
    assert all(o.state == FOLLOWING for *_, o in final)
    assert abs(np.mean([r for _, r, _, _ in final]) - 1.0) <= 0.07


def test_slow_walk_stays_inside_the_band():
    result = run(Scenario(walking_away(0.20), duration=25.0))
    assert fraction_in_band(result.samples(6.0)) >= 0.95


def test_fast_walk_opens_the_gap_then_recovers_when_the_person_stops():
    def path(t):
        return (1.0 + 1.0 * min(max(0.0, t - 2.0), 3.0), 0.0)

    result = run(Scenario(path, duration=20.0))
    assert max(r for _, r, _, _ in result.samples(2.0, 6.0)) > 2.5
    assert fraction_in_band(result.samples(16.0)) == 1.0


def test_side_step_turns_in_place_then_recentres():
    def path(t):
        return (1.0, 0.0) if t < 3.0 else (1.0, 1.0)

    result = run(Scenario(path, duration=10.0))
    after = result.samples(3.1, 10.0)
    wide = [o for _, _, b, o in after if o.bearing is not None and abs(o.bearing) > CFG.turn_in_place_bearing]
    assert wide, "the side-step should produce a wide bearing"
    assert all(o.v_cmd == 0.0 for o in wide)
    assert all(abs(b) < math.radians(5) for _, _, b, _ in result.samples(8.0, 10.0))


def test_person_approaching_never_makes_the_robot_reverse():
    def path(t):
        return (max(0.5, 1.4 - 0.3 * max(0.0, t - 2.0)), 0.0)

    result = run(Scenario(path, duration=8.0))
    assert all(o.v >= 0.0 for o in result.out)
    assert all(o.v == 0.0 for t, r, _, o in result.samples(5.0) if r < 1.0 - 0.1)


def test_occlusion_stops_goes_lost_and_recovers():
    result = run(Scenario(walking_away(0.15), duration=14.0, visible=lambda t: not 5.0 <= t < 7.0))
    during = result.samples(5.8, 7.0)
    assert any(o.state == LOST for *_, o in during)
    assert all(o.v == 0.0 for *_, o in result.samples(6.2, 7.0))
    assert result.out[-1].state == FOLLOWING


def test_bystander_crossing_does_not_steal_the_track():
    def bystander(t):
        return (1.3, 1.5 - 0.6 * t)  # crosses between robot and target around t = 2.5 s

    result = run(Scenario(walking_away(0.1, start=3.0), duration=10.0, bystander=bystander))
    final = result.out[-1]
    assert final.state == FOLLOWING
    assert final.range == pytest.approx(result.true_range[-1], abs=0.15)


def test_obstacle_blocks_forward_motion_before_contact():
    box_x = 1.4  # a 0.3 m box appears between the robot (~0.8 m) and the person (~1.8 m)
    result = run(Scenario(walking_away(0.2), duration=12.0,
                          obstacle=lambda t: (box_x, 0.0) if t >= 6.0 else None))
    blocked = [o for *_, o in result.samples(6.0) if o.state == BLOCKED]
    assert blocked
    first = blocked[0].t
    assert first - 6.0 <= LATENCY + FRAME_EVERY + 2 * DT
    assert all(o.v == 0.0 for *_, o in result.samples(first + 0.5))
    assert max(result.robot_x) < box_x - 0.15 - 0.3  # base origin stays >= 0.3 m from the box


def test_heartbeat_loss_exits():
    result = run(Scenario(walking_away(0.2), duration=10.0, heartbeat_until=5.0))
    assert result.out[-1].exit is True
    assert result.out[-1].rule == "heartbeat"
    assert result.t[-1] <= 5.0 + CFG.heartbeat_timeout + 2 * DT
```

- [ ] **Step 2: Run it**

Run: `uv run --extra dev python -m pytest tests/test_follow_sim.py -q`

Expected: PASS — `9 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_follow_sim.py
git commit -m "test(follow): closed-loop kinematic simulation" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `follow_perception`: pose decode, torso position, TensorRT engine

**Files:**
- Create: `scripts/follow_perception.py`
- Test: `tests/test_follow_perception.py`

**Interfaces:**
- Consumes: `hist_distance` (tests only).
- Produces: COCO indices `NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP`; `BASE_LEFT_SIGN = -1.0`;
  `PoseDetection(box, score, keypoints (17, 3))`; `Letterbox(scale, pad_x, pad_y)`; `letterbox(image, size=640) -> (tensor, Letterbox)`;
  `nms(boxes, scores, iou)`; `decode_pose(raw, lb, conf=0.40, iou=0.5) -> list[PoseDetection]`; `hand_raised(det) -> bool`;
  `torso_rect(det) -> (x1, y1, x2, y2)`; `torso_histogram(image, rect) -> ndarray (64,) | None`;
  `base_to_local(points_base, left_sign=BASE_LEFT_SIGN) -> (N, 3)`; `mask_to_image_pixels(mask, depth_shape, image_shape) -> (N, 2)`;
  `person_position(points_local, pixels, rect) -> (forward, left) | None`;
  `PoseEngine(engine_path)` with `infer(image) -> list[PoseDetection]` and `close()` (Jetson only; imports TensorRT/PyCUDA lazily).

Numpy-only at import. The last test cross-checks the decoder against Ultralytics' own postprocessing of the committed checkpoint; it is skipped unless the `vision` extra is installed (it passed during planning).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_follow_perception.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from follow_core import hist_distance
from follow_perception import (
    L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST, PAD_VALUE,
    Letterbox, PoseDetection, base_to_local, decode_pose, hand_raised, letterbox,
    mask_to_image_pixels, person_position, torso_histogram, torso_rect,
)

ROOT = Path(__file__).resolve().parents[1]


def detection(box=(100, 50, 300, 450), points=None):
    """points: keypoint index -> (x, y, conf); every other keypoint has confidence 0."""
    kp = np.zeros((17, 3), dtype=np.float32)
    for index, value in (points or {}).items():
        kp[index] = value
    return PoseDetection(np.array(box, dtype=np.float32), 0.9, kp)


def upright_person(overrides=None):
    points = {NOSE: (200, 100, 0.9), L_SHOULDER: (240, 150, 0.9), R_SHOULDER: (160, 150, 0.9),
              L_HIP: (230, 280, 0.9), R_HIP: (170, 280, 0.9), L_WRIST: (250, 300, 0.9),
              R_WRIST: (150, 300, 0.9)}
    points.update(overrides or {})
    return detection(points=points)


def test_letterbox_scales_and_pads_the_head_eye():
    image = np.zeros((960, 1280, 3), dtype=np.uint8)
    tensor, lb = letterbox(image, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert lb == Letterbox(0.5, 0, 80)
    assert tensor[0, 0, 0, 0] == pytest.approx(PAD_VALUE / 255)
    assert tensor[0, 0, 320, 320] == 0.0


def test_decode_filters_suppresses_and_maps_back_to_image_pixels():
    raw = np.zeros((1, 56, 8400), dtype=np.float32)
    raw[0, :5, 0] = (320, 320, 100, 200, 0.9)  # kept
    raw[0, :5, 1] = (322, 318, 100, 200, 0.8)  # overlaps the first: suppressed
    raw[0, :5, 2] = (100, 400, 50, 100, 0.3)  # below the confidence threshold
    raw[0, 5:8, 0] = (320, 250, 0.95)  # nose
    dets = decode_pose(raw, Letterbox(0.5, 0, 80))
    assert len(dets) == 1
    assert dets[0].score == pytest.approx(0.9)
    assert dets[0].box == pytest.approx([540, 280, 740, 680])
    assert dets[0].keypoints[NOSE] == pytest.approx([640, 340, 0.95])


def test_decode_with_no_confident_boxes_is_empty():
    assert decode_pose(np.zeros((1, 56, 8400), dtype=np.float32), Letterbox(1.0, 0, 0)) == []


def test_hand_raised_needs_a_confident_wrist_well_above_the_nose():
    assert hand_raised(upright_person({L_WRIST: (250, 40, 0.9)})) is True
    assert hand_raised(upright_person({R_WRIST: (150, 80, 0.9)})) is False  # only 20 px, margin is 40
    assert hand_raised(upright_person({L_WRIST: (250, 40, 0.3)})) is False
    assert hand_raised(upright_person({L_WRIST: (250, 40, 0.9), NOSE: (200, 100, 0.2)})) is False
    assert hand_raised(upright_person()) is False


def test_torso_rect_from_keypoints_with_fallback_and_minimum_width():
    assert torso_rect(upright_person()) == pytest.approx((160, 150, 240, 280))
    assert torso_rect(detection()) == pytest.approx((100 + 200 / 3, 50 + 400 / 3, 300 - 200 / 3, 450 - 400 / 3))
    side_on = upright_person({L_SHOULDER: (202, 150, 0.9), R_SHOULDER: (198, 150, 0.9),
                              L_HIP: (201, 280, 0.9), R_HIP: (199, 280, 0.9)})
    x1, _, x2, _ = torso_rect(side_on)
    assert x2 - x1 == pytest.approx(50)


def test_torso_histogram_separates_clothing_colours():
    image = np.zeros((100, 300, 3), dtype=np.uint8)
    image[:, :100] = (200, 30, 30)  # red
    image[:, 100:200] = (30, 30, 200)  # blue
    image[:, 200:] = (245, 245, 245)  # white; black is the zero default elsewhere
    red = torso_histogram(image, (0, 0, 100, 100))
    blue = torso_histogram(image, (100, 0, 200, 100))
    white = torso_histogram(image, (200, 0, 300, 100))
    black = torso_histogram(np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 10, 10))
    assert red.shape == (64,)
    assert red.sum() == pytest.approx(1.0)
    assert hist_distance(red, blue) > 0.9
    assert hist_distance(white, black) > 0.9
    assert hist_distance(red, torso_histogram(image, (10, 10, 90, 90))) == pytest.approx(0.0)
    assert torso_histogram(image, (400, 0, 500, 50)) is None


def test_base_frame_points_become_forward_left_up():
    local = base_to_local(np.array([[0.2, 1.0, 0.5]]))
    assert local[0] == pytest.approx([1.0, -0.2, 0.5])


def test_mask_indices_map_to_detection_image_pixels():
    pixels = mask_to_image_pixels(np.array([10 * 640 + 20]), (384, 640), (768, 1280))
    assert pixels[0] == pytest.approx([40, 20])


def test_person_position_is_the_torso_median_and_ignores_background_and_floor():
    rng = np.random.default_rng(0)
    torso = np.column_stack([rng.normal(1.1, 0.02, 100), rng.normal(0.1, 0.02, 100), np.full(100, 1.2)])
    background = np.column_stack([np.full(30, 3.0), np.zeros(30), np.full(30, 1.2)])
    floor = np.column_stack([np.full(80, 0.9), np.zeros(80), np.zeros(80)])
    points = np.vstack([torso, background, floor])
    pixels = np.full((len(points), 2), 50.0)
    position = person_position(points, pixels, (0, 0, 100, 100))
    assert position == pytest.approx((1.1, 0.1), abs=0.02)
    assert person_position(points[:39], pixels[:39], (0, 0, 100, 100)) is None
    assert person_position(points, pixels, (60, 60, 100, 100)) is None


def test_decode_matches_ultralytics_on_the_committed_checkpoint():
    pytest.importorskip("ultralytics")
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    model = YOLO(str(ROOT / "yolo11n-pose.pt"))
    bgr = cv2.imread(str(ASSETS / "bus.jpg"))
    expected = model.predict(bgr, imgsz=640, conf=0.5, iou=0.5, classes=[0], verbose=False)[0]
    tensor, lb = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 640)
    with torch.no_grad():
        raw = model.model.float().eval()(torch.from_numpy(tensor))
    raw = raw[0] if isinstance(raw, (list, tuple)) else raw
    ours = decode_pose(raw.numpy(), lb, conf=0.40, iou=0.5)

    theirs = expected.boxes.xyxy.numpy()
    their_kp = expected.keypoints.xy.numpy()
    assert len(theirs) >= 3
    for box, kp in zip(theirs, their_kp):
        ious = [_iou(box, d.box) for d in ours]
        best = int(np.argmax(ious))
        assert ious[best] >= 0.85
        assert np.mean(np.linalg.norm(ours[best].keypoints[:, :2] - kp, axis=1)) <= 8.0


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_follow_perception.py -q`

Expected: FAIL; the output contains `No module named 'follow_perception'`.

- [ ] **Step 3: Write `scripts/follow_perception.py`**

Create `scripts/follow_perception.py`:

```python
"""Robot-side perception for person-follow: pose engine, decoding, and torso position.

Everything here is numpy except ``PoseEngine``, which imports TensorRT and
PyCUDA only when constructed on the Jetson. Decode and geometry are unit-tested
on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 5, 6, 9, 10, 11, 12
PAD_VALUE = 114
# camera.points uses base +y forward, +x lateral. -1 means +x points to the robot's
# right (right-handed, z up). Verified at gate G0; flip here if G0 shows otherwise.
BASE_LEFT_SIGN = -1.0


@dataclass(frozen=True, eq=False)
class PoseDetection:
    box: np.ndarray  # (4,) x1, y1, x2, y2 in source-image pixels
    score: float
    keypoints: np.ndarray  # (17, 3) x, y, confidence in source-image pixels


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int


def _resize(image, width, height):
    try:
        import cv2
    except ImportError:  # laptop without OpenCV: nearest neighbour is enough for tests
        rows = np.minimum((np.arange(height) + 0.5) * image.shape[0] / height, image.shape[0] - 1).astype(int)
        cols = np.minimum((np.arange(width) + 0.5) * image.shape[1] / width, image.shape[1] - 1).astype(int)
        return image[rows][:, cols]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def letterbox(image, size=640):
    """RGB uint8 (H, W, 3) -> float32 (1, 3, size, size) in [0, 1], and the mapping back."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = _resize(image, nw, nh)
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), Letterbox(scale, pad_x, pad_y)


def nms(boxes, scores, iou_threshold):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode_pose(raw, lb, conf=0.40, iou=0.5):
    """YOLO pose output (1, 56, N) -> detections in source-image pixels, best first."""
    preds = np.asarray(raw, dtype=np.float32)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] == 56:
        preds = preds.T
    preds = preds[preds[:, 4] >= conf]
    if len(preds) == 0:
        return []
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keypoints = preds[:, 5:].reshape(-1, 17, 3).copy()
    offset = np.array([lb.pad_x, lb.pad_y, lb.pad_x, lb.pad_y], dtype=np.float32)
    detections = []
    for i in nms(boxes, preds[:, 4], iou):
        kp = keypoints[i]
        kp[:, 0] = (kp[:, 0] - lb.pad_x) / lb.scale
        kp[:, 1] = (kp[:, 1] - lb.pad_y) / lb.scale
        detections.append(PoseDetection((boxes[i] - offset) / lb.scale, float(preds[i, 4]), kp))
    return detections


def hand_raised(det, min_conf=0.5, margin_frac=0.10):
    """Either wrist above the nose by at least ``margin_frac`` of the person's box height."""
    kp = det.keypoints
    if kp[NOSE, 2] < min_conf:
        return False
    margin = margin_frac * (det.box[3] - det.box[1])
    return any(kp[w, 2] >= min_conf and kp[NOSE, 1] - kp[w, 1] >= margin for w in (L_WRIST, R_WRIST))


def torso_rect(det, min_conf=0.5):
    """Shoulder-hip rectangle, or the middle third of the box when keypoints are unsure."""
    kp = det.keypoints
    idx = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    x1, y1, x2, y2 = (float(v) for v in det.box)
    if all(kp[i, 2] >= min_conf for i in idx):
        tx1, tx2 = float(kp[idx, 0].min()), float(kp[idx, 0].max())
        ty1, ty2 = float(kp[idx, 1].min()), float(kp[idx, 1].max())
        min_width = 0.25 * (x2 - x1)  # side-on people have overlapping shoulders
        if tx2 - tx1 < min_width:
            mid = (tx1 + tx2) / 2
            tx1, tx2 = mid - min_width / 2, mid + min_width / 2
        return tx1, ty1, tx2, ty2
    w, h = x2 - x1, y2 - y1
    return x1 + w / 3, y1 + h / 3, x2 - w / 3, y2 - h / 3


def torso_histogram(image, rect):
    """4x4x4 HSV histogram (64 bins, L1-normalised) of the torso; None if the rect is empty."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in rect)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    rgb = image[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    value = rgb.max(axis=1)
    delta = value - rgb.min(axis=1)
    sat = np.where(value > 0, delta / np.maximum(value, 1e-6), 0.0)
    hue = np.zeros_like(value)
    chroma = delta > 1e-6
    rmax = chroma & (value == r)
    gmax = chroma & (value == g) & ~rmax
    bmax = chroma & ~rmax & ~gmax
    hue[rmax] = ((g - b)[rmax] / delta[rmax]) % 6
    hue[gmax] = (b - r)[gmax] / delta[gmax] + 2
    hue[bmax] = (r - g)[bmax] / delta[bmax] + 4
    hist, _ = np.histogramdd(
        np.column_stack([hue / 6.0, sat, value]), bins=(4, 4, 4), range=((0, 1), (0, 1), (0, 1))
    )
    return (hist / hist.sum()).ravel()


def base_to_local(points_base, left_sign=BASE_LEFT_SIGN):
    """camera.points base frame (x lateral, y forward, z up) -> (forward, left, up)."""
    p = np.asarray(points_base, dtype=np.float64)
    return np.column_stack([p[:, 1], left_sign * p[:, 0], p[:, 2]])


def mask_to_image_pixels(mask, depth_shape, image_shape):
    """camera.points ``mask`` (flat depth-pixel indices) -> (N, 2) u, v in the detection image."""
    depth_h, depth_w = depth_shape[:2]
    image_h, image_w = image_shape[:2]
    mask = np.asarray(mask, dtype=np.int64)
    rows, cols = mask // depth_w, mask % depth_w
    return np.column_stack([cols * (image_w / depth_w), rows * (image_h / depth_h)])


def person_position(points_local, pixels, rect, min_points=40, z_range=(0.2, 2.0)):
    """Median (forward, left) of the points that land on the torso; None if too few."""
    u, v = pixels[:, 0], pixels[:, 1]
    x1, y1, x2, y2 = rect
    z = points_local[:, 2]
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2) & (z >= z_range[0]) & (z <= z_range[1])
    if np.count_nonzero(inside) < min_points:
        return None
    selected = points_local[inside]
    return float(np.median(selected[:, 0])), float(np.median(selected[:, 1]))


class PoseEngine:
    """yolo11n-pose TensorRT engine on the Jetson (TensorRT >= 8.5 tensor-address API)."""

    def __init__(self, engine_path, conf=0.40, iou=0.5):
        import pycuda.driver as cuda
        import tensorrt as trt

        cuda.init()
        self._cuda = cuda
        self._ctx = cuda.Device(0).make_context()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as source, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(source.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.buffers[name] = (host, device, shape, is_input)
        self.input_name = next(n for n, b in self.buffers.items() if b[3])
        self.output_name = next(n for n, b in self.buffers.items() if not b[3])
        self.size = int(self.buffers[self.input_name][2][-1])
        self.conf, self.iou = conf, iou

    def infer(self, image):
        tensor, lb = letterbox(image, self.size)
        host, device, _, _ = self.buffers[self.input_name]
        np.copyto(host, tensor.ravel().astype(host.dtype))
        self._cuda.memcpy_htod_async(device, host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        out_host, out_device, out_shape, _ = self.buffers[self.output_name]
        self._cuda.memcpy_dtoh_async(out_host, out_device, self.stream)
        self.stream.synchronize()
        return decode_pose(out_host.reshape(out_shape), lb, self.conf, self.iou)

    def close(self):
        self._ctx.pop()
        self._ctx.detach()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_follow_perception.py -q`

Expected: PASS — `9 passed, 1 skipped`.
(The skip is the Ultralytics cross-check, which needs the `vision` extra.)

- [ ] **Step 5: Run the Ultralytics cross-check (downloads PyTorch the first time)**

Run: `uv run --extra dev --extra vision python -m pytest tests/test_follow_perception.py -q`

Expected: PASS — `10 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/follow_perception.py tests/test_follow_perception.py
git commit -m "feat(follow): pose decoding, torso localisation, TensorRT engine" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Robot runner `robot_follow.py`

**Files:**
- Create: `scripts/robot_follow.py`
- Test: `tests/test_robot_follow_cli.py`

**Interfaces:**
- Consumes: `follow_core` (Tasks 1–4) and `follow_perception` (Task 6).
- Produces: the PEP 723 script `/tmp/robot_follow.py` with flags `--gap --v-max --engine --pid-file --log-dir --dry-run --rotate-only --no-heartbeat --check`;
  stdout lines `[follow] follow active …` (the dashboard's readiness marker), `FOLLOW_STATUS {json}` at 5 Hz, `[follow] exit: <rule>`;
  stdin JSON lines per spec §7; CSV log `/tmp/baymax_follow_<timestamp>.csv` with columns `CSV_FIELDS`;
  module functions `parse_args(argv)`, `loop_config(args)`, `left_eye(image)`, `perceive(engine, image, points_base, pixels, t)`, `make_pixel_mapper(depth_shape)`;
  constants `WHEEL_ORDER = (0, 1)`, `WHEEL_SIGNS = (1.0, 1.0)`.

The runner makes no decisions: it reads BBOS topics, runs the engine, calls `FollowLoop.tick`, and writes `drive.ctrl`/`led.ctrl`. BBOS is imported inside `run()`, so the argument checks and the perception glue are testable on a laptop. The dependency header mirrors `bbapps/greeter/main.py`, the app already proven to run TensorRT through PyCUDA on this robot.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_robot_follow_cli.py`:

```python
import sys

import numpy as np
import pytest

import robot_follow
from follow_perception import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER, PoseDetection


def test_defaults_are_the_bring_up_limits():
    args = robot_follow.parse_args([])
    assert (args.gap, args.v_max, args.dry_run, args.rotate_only) == (1.0, 0.15, False, False)
    assert robot_follow.loop_config(args).v_max == 0.15


def test_rotate_only_holds_forward_speed_at_zero():
    assert robot_follow.loop_config(robot_follow.parse_args(["--rotate-only"])).v_max == 0.0


@pytest.mark.parametrize("argv", [
    ["--gap", "2.0"], ["--gap", "0.3"], ["--v-max", "0.5"], ["--v-max", "0"], ["--no-heartbeat"],
])
def test_unsafe_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        robot_follow.parse_args(argv)


def test_only_a_dry_run_may_skip_the_heartbeat():
    assert robot_follow.parse_args(["--dry-run", "--no-heartbeat"]).no_heartbeat is True


def test_left_eye_splits_side_by_side_stereo_only():
    assert robot_follow.left_eye(np.zeros((960, 2560, 3))).shape == (960, 1280, 3)
    assert robot_follow.left_eye(np.zeros((384, 640, 3))).shape == (384, 640, 3)


def test_perceive_turns_detections_into_located_people():
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[NOSE] = (150, 110, 0.9)
    kp[L_SHOULDER], kp[R_SHOULDER] = (180, 150, 0.9), (120, 150, 0.9)
    kp[L_HIP], kp[R_HIP] = (175, 250, 0.9), (125, 250, 0.9)

    class FakeEngine:
        def infer(self, image):
            return [PoseDetection(np.array([100, 90, 200, 380], dtype=np.float32), 0.8, kp)]

    rows, cols = np.meshgrid(np.arange(160, 240, 5), np.arange(130, 170, 5))
    mask = (rows * 640 + cols).ravel()
    points_base = np.tile([-0.1, 1.2, 1.1], (len(mask), 1))  # x=-0.1 is 0.1 m to the left
    image = np.zeros((384, 640, 3), dtype=np.uint8)

    pixels = robot_follow.mask_to_image_pixels(mask, (384, 640), image.shape)
    perception = robot_follow.perceive(FakeEngine(), image, points_base, pixels, 5.0)

    assert perception.t == 5.0
    assert perception.points.shape == (len(mask), 3)
    [person] = perception.people
    assert (person.forward, person.left) == pytest.approx((1.2, 0.1))
    assert person.hand_raised is False
    assert person.hist.shape == (64,)


def test_runner_imports_without_bbos():
    assert "bbos" not in sys.modules
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_robot_follow_cli.py -q`

Expected: FAIL; the output contains `No module named 'robot_follow'`.

- [ ] **Step 3: Write `scripts/robot_follow.py`**

Create `scripts/robot_follow.py`:

```python
# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "bbos",
#   "bbai",
#   "numpy",
#   "opencv-python",
#   "pycuda",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# bbai = { path = "/home/bracketbot/bbai", editable = true }
# ///
"""Person-follow runner for BracketBot. Copied to /tmp by robot_dashboard.py.

The dependency block mirrors bbapps/greeter/main.py, the app already proven to
run TensorRT through PyCUDA on this robot (bbai brings the TensorRT bindings).
While running it is the only writer of drive.ctrl and led.ctrl. Every decision
lives in follow_core.FollowLoop; this file only moves data between BBOS topics,
the pose engine, and the loop.

    uv run --script /tmp/robot_follow.py --pid-file /tmp/f.pid     # from the dashboard
    uv run --script /tmp/robot_follow.py --check                   # gate G1
    uv run --script /tmp/robot_follow.py --dry-run --no-heartbeat  # gate G2
    uv run --script /tmp/robot_follow.py --rotate-only ...         # gate G3
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from follow_core import (
    FollowConfig, FollowLoop, Perception, PersonObservation, TickInputs, led_color,
    parse_command, start_refusal, status_line, timestamp_to_seconds, wheel_twist,
)
from follow_perception import (
    PoseEngine, base_to_local, hand_raised, mask_to_image_pixels, person_position,
    torso_histogram, torso_rect,
)

PERIOD = 0.02  # 50 Hz control loop; drive.ctrl times out after 0.1 s
STATUS_PERIOD = 0.2
CSV_PERIOD = 0.05
PAIR_TOLERANCE = 0.05  # max seconds between the RGB frame and the point cloud it is paired with
# drive.state.vel -> (left, right) turns/s, forward-positive. Verified at gates G3/G4a.
WHEEL_ORDER = (0, 1)
WHEEL_SIGNS = (1.0, 1.0)
DRIVE_WRITER_PATTERNS = (
    "greeter/main.py", "nav/main.py", "bbapps/teleop.py", "quest_teleop/main.py",
    "leader_follower_teleop.py", "live_inference.py",
)
CSV_FIELDS = (
    "t", "state", "rule", "gap", "range", "bearing", "error", "v_cmd", "omega_cmd", "v", "omega",
    "measured_v", "measured_omega", "blocked", "corridor_points", "track_age", "people",
)
STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def build_parser():
    cfg = FollowConfig()
    parser = argparse.ArgumentParser(description="Follow one person at a held distance")
    parser.add_argument("--gap", type=float, default=cfg.gap_default)
    parser.add_argument("--v-max", type=float, default=cfg.v_max)
    parser.add_argument("--engine", type=Path, default=Path.home() / ".cache/baymax/yolo11n-pose.engine")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--dry-run", action="store_true", help="compute and log; never open drive.ctrl")
    parser.add_argument("--rotate-only", action="store_true", help="forward speed held at 0 (gate G3)")
    parser.add_argument("--no-heartbeat", action="store_true", help="only with --dry-run: no dashboard needed")
    parser.add_argument("--check", action="store_true", help="gate G1: time the engine on a live frame, then exit")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = FollowConfig()
    if not cfg.gap_min <= args.gap <= cfg.gap_max:
        parser.error(f"--gap must be between {cfg.gap_min} and {cfg.gap_max} m")
    if not 0.0 < args.v_max <= 0.30:
        parser.error("--v-max must be above 0 and at most 0.30 m/s (the drive daemon's clamp)")
    if args.no_heartbeat and not args.dry_run:
        parser.error("--no-heartbeat is only allowed with --dry-run")
    return args


def loop_config(args):
    return replace(FollowConfig(), v_max=0.0 if args.rotate_only else args.v_max)


def field(data, name):
    try:
        return data[name]
    except (KeyError, ValueError, IndexError):
        return None


def wait_fresh(reader, timeout, topic):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started > timeout:
            raise RuntimeError(f"no fresh {topic} sample; is its daemon running?")
        time.sleep(0.002)
    return reader.data


def left_eye(image):
    # A side-by-side stereo image is more than twice as wide as it is tall.
    return image[:, : image.shape[1] // 2] if image.shape[1] > 2.5 * image.shape[0] else image


def stamp(data, arrival):
    value = field(data, "timestamp")
    return arrival if value is None else timestamp_to_seconds(np.asarray(value).item())


def other_drive_writers():
    found = []
    for pattern in DRIVE_WRITER_PATTERNS:
        result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
        found.extend(
            line.strip() for line in result.stdout.splitlines()
            if line.strip() and int(line.split()[0]) != os.getpid()
        )
    return found


def perceive(engine, image, points_base, pixels, t):
    """Pose detections + the point cloud -> located people. ``pixels``: each point's (u, v) in ``image``."""
    detections = engine.infer(image)
    local = base_to_local(points_base)
    people = []
    for det in detections:
        rect = torso_rect(det)
        position = person_position(local, pixels, rect)
        if position is not None:
            people.append(PersonObservation(
                position[0], position[1], det.score, hand_raised(det), torso_histogram(image, rect)
            ))
    return Perception(t, tuple(people), local)


def make_pixel_mapper(depth_shape):
    """Where each camera.points point lands in the detection image: (points_base, mask, image_shape) -> (N, 2)."""
    return lambda points_base, mask, image_shape: mask_to_image_pixels(mask, depth_shape, image_shape)


def start_command_reader(commands):
    def read():
        for line in sys.stdin:
            commands.put(line)
        commands.put(None)  # EOF: the dashboard or SSH is gone

    threading.Thread(target=read, name="follow-stdin", daemon=True).start()


def write_twist(writer, v, omega):
    with writer.buf() as frame:
        frame["twist"] = np.array([v, omega], dtype=np.float32)


def write_led(writer, rgb):
    with writer.buf() as frame:
        frame["rgb"] = np.asarray(rgb, dtype=np.uint8)
        frame["brightness"] = np.int16(-1)
        frame["period_ms"] = np.uint16(0)


def run_check(engine, reader_cls, image_topic):
    with reader_cls(image_topic, keeptime=False) as source:
        image = left_eye(np.asarray(wait_fresh(source, 3.0, image_topic)["rgb"]).copy())
    timings, detections = [], []
    for _ in range(20):
        started = time.perf_counter()
        detections = engine.infer(image)
        timings.append((time.perf_counter() - started) * 1000)
    best = detections[0] if detections else None
    print(json.dumps({
        "image_shape": list(image.shape),
        "latency_ms_p50": round(float(np.median(timings[5:])), 1),
        "latency_ms_max": round(float(np.max(timings[5:])), 1),
        "people": len(detections),
        "best": None if best is None else {
            "box": [round(float(v)) for v in best.box], "score": round(best.score, 3),
            "hand_raised": hand_raised(best),
        },
    }), flush=True)


def control_loop(args, cfg, engine, readers, drive, led, to_pixels, wheel_diam, robot_width):
    camera, points, imu, drive_state = readers
    loop = FollowLoop(cfg, args.gap)
    commands = queue.Queue()
    if not args.no_heartbeat:
        start_command_reader(commands)
    now = time.monotonic()
    last_heartbeat = now
    command_stop = False
    rpy = np.zeros(3)
    measured = (0.0, 0.0)
    rgb_frame = None  # (arrival, stamp, image)
    last_status = last_csv = 0.0
    state, state_since = None, now
    log_path = args.log_dir / f"baymax_follow_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with log_path.open("w", newline="") as log_file:
        log = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
        log.writeheader()
        print(f"[follow] logging to {log_path}", flush=True)
        while True:
            t = time.monotonic()
            while not commands.empty():
                line = commands.get()
                if line is None:
                    command_stop = True
                    break
                command = parse_command(line, cfg)
                if command is None:
                    print(f"[follow] ignored command: {line.strip()[:80]}", flush=True)
                    continue
                last_heartbeat = t
                if command.kind == "stop":
                    command_stop = True
                elif command.kind == "gap":
                    print(f"[follow] gap {loop.set_gap(command.gap):.2f} m", flush=True)
            if args.no_heartbeat:
                last_heartbeat = t

            if imu.ready():
                rpy = np.asarray(imu.data["rpy"], dtype=float)
            if drive_state.ready():
                vel = np.asarray(drive_state.data["vel"], dtype=float)
                measured = wheel_twist(vel[list(WHEEL_ORDER)], wheel_diam, robot_width, WHEEL_SIGNS)
            if camera.ready():
                data = camera.data
                rgb_frame = (t, stamp(data, t), left_eye(np.asarray(data["rgb"]).copy()))
            perception = None
            if points.ready():
                data = points.data
                n = int(data["num_points"])
                cloud_stamp = stamp(data, t)
                if rgb_frame is not None and abs(rgb_frame[1] - cloud_stamp) <= PAIR_TOLERANCE \
                        and t - rgb_frame[0] <= PAIR_TOLERANCE:
                    image, points_base = rgb_frame[2], np.asarray(data["points"][:n])
                    pixels = to_pixels(points_base, np.asarray(data["mask"][:n]), image.shape)
                    perception = perceive(engine, image, points_base, pixels, t)

            out = loop.tick(TickInputs(
                t=t, heartbeat_age=t - last_heartbeat,
                stop_requested=STOP_REQUESTED or command_stop,
                roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]),
                measured_v=measured[0], measured_omega=measured[1], perception=perception,
            ))
            if drive is not None:
                write_twist(drive, out.v, out.omega)
            if out.state != state:
                print(f"[follow] state {out.state}", flush=True)
                state, state_since = out.state, t
            write_led(led, led_color(out.state, t - state_since))
            if t - last_status >= STATUS_PERIOD:
                print(status_line(out), flush=True)
                last_status = t
            if t - last_csv >= CSV_PERIOD:
                log.writerow({
                    "t": round(t, 3), "state": out.state, "rule": out.rule, "gap": out.gap,
                    "range": out.range, "bearing": out.bearing, "error": out.error,
                    "v_cmd": out.v_cmd, "omega_cmd": out.omega_cmd, "v": out.v, "omega": out.omega,
                    "measured_v": measured[0], "measured_omega": measured[1], "blocked": out.blocked,
                    "corridor_points": out.corridor_points, "track_age": out.track_age,
                    "people": "" if perception is None else len(perception.people),
                })
                last_csv = t
            if out.exit:
                print(f"[follow] exit: {out.rule}", flush=True)
                return
            time.sleep(max(0.0, PERIOD - (time.monotonic() - t)))


def run(args):
    from bbos import Config, Reader, Type, Writer

    engine = PoseEngine(args.engine)
    try:
        image_topic = "camera.rect"
        if args.check:
            run_check(engine, Reader, image_topic)
            return
        cfg = loop_config(args)
        drive_cfg = Config("drive")
        wheel_diam = float(drive_cfg.wheel_diam)
        robot_width = float(getattr(drive_cfg, "robot_width", cfg.robot_width))
        low_battery_v = getattr(Config("base"), "low_battery_v", None)
        with ExitStack() as stack:
            camera = stack.enter_context(Reader(image_topic, keeptime=False))
            points = stack.enter_context(Reader("camera.points", keeptime=False))
            depth = stack.enter_context(Reader("camera.depth", keeptime=False))
            imu = stack.enter_context(Reader("imu.orientation", keeptime=False))
            drive_state = stack.enter_context(Reader("drive.state", keeptime=False))
            drive_status = stack.enter_context(Reader("drive.status", keeptime=False))

            depth_shape = np.asarray(wait_fresh(depth, 3.0, "camera.depth")["depth"]).shape
            rpy = np.asarray(wait_fresh(imu, 2.0, "imu.orientation")["rpy"], dtype=float)
            try:
                wait_fresh(points, 2.0, "camera.points")
                points_fresh = True
            except RuntimeError:
                points_fresh = False
            try:
                voltage = float(wait_fresh(drive_status, 2.0, "drive.status")["voltage"])
            except RuntimeError:
                voltage = None
            if low_battery_v is None:
                print("[follow] warning: base.low_battery_v unknown; battery not checked", flush=True)
            refusal = start_refusal(
                cfg, roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]), voltage=voltage,
                low_battery_v=None if low_battery_v is None else float(low_battery_v),
                drive_writers=other_drive_writers(), points_fresh=points_fresh,
            )
            if refusal:
                raise RuntimeError(f"refusing to start: {refusal}")
            engine.infer(np.zeros((depth_shape[0], depth_shape[1], 3), dtype=np.uint8))  # warm-up

            drive = None
            if not args.dry_run:
                drive = stack.enter_context(Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False))
            led = stack.enter_context(Writer("led.ctrl", Type("led_ctrl"), keeptime=False))
            mode = "dry run" if args.dry_run else "rotate only" if args.rotate_only else f"v_max {cfg.v_max:.2f} m/s"
            print(f"[follow] follow active ({mode}, gap {args.gap:.2f} m) - raise a hand", flush=True)
            try:
                control_loop(args, cfg, engine, (camera, points, imu, drive_state),
                             drive, led, make_pixel_mapper(depth_shape), wheel_diam, robot_width)
            finally:
                if drive is not None:
                    for _ in range(6):
                        write_twist(drive, 0.0, 0.0)
                        time.sleep(PERIOD)
                write_led(led, (0, 0, 0))
                print("[follow] stopped; zero twist sent", flush=True)
    finally:
        engine.close()


def main(argv=None):
    args = parse_args(argv)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_stop)
    if args.pid_file is not None:
        args.pid_file.write_text(f"{os.getpid()}\n")
    try:
        run(args)
    except RuntimeError as exc:
        print(f"[follow] {exc}", flush=True)
        raise SystemExit(1) from exc
    finally:
        if args.pid_file is not None:
            try:
                if args.pid_file.read_text().strip() == str(os.getpid()):
                    args.pid_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_robot_follow_cli.py -q`

Expected: PASS — `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/robot_follow.py tests/test_robot_follow_cli.py
git commit -m "feat(follow): robot-side follow runner" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Dashboard: Follow button, distance slider, heartbeat, exclusivity

**Files:**
- Modify: `scripts/robot_dashboard.py` (24 exact edits below)
- Modify: `README.md`
- Test: `tests/test_robot_dashboard.py` (append)

**Interfaces:**
- Consumes: the runner's CLI and stdout/stdin protocol (Task 7). Does **not** import `follow_core` (the dashboard stays standard-library only; a test asserts the duplicated constants match).
- Produces: `FOLLOW_RUNNER`, `FOLLOW_MODULES`, `REMOTE_FOLLOW_RUNNER`, `FOLLOW_GAP_MIN/MAX/DEFAULT`, `FOLLOW_STATUS_PREFIX`, `FOLLOW_HEARTBEAT_PERIOD`;
  `remote_script_command(script, *args)`; `RobotController(ssh_hosts, simulate=False, follow_args=())` with
  `set_follow(enabled) -> (ok, message)` and `set_follow_gap(gap) -> (ok, message)`; snapshot keys
  `follow_enabled follow_transition follow_phase follow_status follow_gap follow_gap_min follow_gap_max`;
  HTTP `POST /api/follow {enabled}` and `POST /api/follow/gap {gap}`; CLI flags `--follow-v-max` (default 0.15, max 0.30) and `--follow-rotate-only`.

Follow owns the base, LEDs, and camera: it refuses to start during an action or lean, and actions and lean refuse while it runs. `Esc` now works even when the slider has keyboard focus.

- [ ] **Step 1: Import the new names at the top of `tests/test_robot_dashboard.py`**

In `tests/test_robot_dashboard.py`, replace:

```python
import time
from types import SimpleNamespace
import wave
```

with:

```python
import json
import threading
import time
from types import SimpleNamespace
import wave
```

In `tests/test_robot_dashboard.py`, replace:

```python
    EFFECT_RUNNER,
    ROUTINE_LIST,
```

with:

```python
    EFFECT_RUNNER,
    FOLLOW_GAP_DEFAULT,
    FOLLOW_GAP_MAX,
    FOLLOW_GAP_MIN,
    FOLLOW_MODULES,
    FOLLOW_RUNNER,
    FOLLOW_STATUS_PREFIX,
    ROUTINE_LIST,
```

In `tests/test_robot_dashboard.py`, replace:

```python
    remote_python_command,
)
```

with:

```python
    remote_python_command,
    remote_script_command,
)
```

- [ ] **Step 2: Append the failing tests**

Append to the end of `tests/test_robot_dashboard.py` (two blank lines before it):

```python
# --- follow mode ------------------------------------------------------------


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def follow_off(controller):
    state = controller.state.snapshot()
    return not state["follow_enabled"] and not state["follow_transition"]


def test_follow_protocol_matches_the_runner():
    import follow_core

    cfg = follow_core.FollowConfig()
    assert FOLLOW_STATUS_PREFIX == follow_core.STATUS_PREFIX
    assert (FOLLOW_GAP_MIN, FOLLOW_GAP_MAX, FOLLOW_GAP_DEFAULT) == (cfg.gap_min, cfg.gap_max, cfg.gap_default)

    controller = RobotController(("not-used",), simulate=True)
    controller.set_follow(True)
    wait_for(lambda: controller.state.snapshot()["follow_status"] is not None)
    simulated = controller.state.snapshot()["follow_status"]
    controller.stop()
    wait_for(lambda: follow_off(controller))

    out = follow_core.FollowLoop(cfg).tick(follow_core.TickInputs(0.0, 0.1, False, 0.0, 0.0, 0.0, 0.0))
    real = json.loads(follow_core.status_line(out)[len(FOLLOW_STATUS_PREFIX):])
    assert set(simulated) == set(real)


def test_follow_simulation_reports_status_and_stops():
    controller = RobotController(("not-used",), simulate=True)

    assert controller.set_follow(True) == (True, "Starting follow mode")
    wait_for(lambda: (controller.state.snapshot()["follow_status"] or {}).get("state") == "FOLLOWING")
    assert controller.state.snapshot()["follow_enabled"] is True
    assert controller.set_follow(True) == (False, "Follow is already on")

    assert controller.stop() == (True, "Stop requested")
    wait_for(lambda: follow_off(controller))
    state = controller.state.snapshot()
    assert state["follow_phase"] == "Follow off"
    assert state["follow_status"] is None
    assert state["error"] is None


def test_follow_excludes_actions_and_lean():
    controller = RobotController(("not-used",), simulate=True)
    controller.set_follow(True)
    wait_for(lambda: controller.state.snapshot()["follow_enabled"])

    assert controller.run_action("wave") == (False, "Stop following before running actions")
    assert controller.run_routine("welcome") == (False, "Stop following before running actions")
    assert controller.set_lean(True) == (False, "Stop following before enabling lean")

    assert controller.set_follow(False) == (True, "Stopping follow mode")
    wait_for(lambda: follow_off(controller))
    assert controller.set_lean(True) == (True, "Lean enabled")
    assert controller.set_follow(True) == (False, "Return to balance mode before following")


def test_follow_gap_validation():
    controller = RobotController(("not-used",), simulate=True)

    for bad in ("1.2", True, None, float("nan")):
        assert controller.set_follow_gap(bad) == (False, "gap must be a number of metres")
    assert controller.set_follow_gap(2.0) == (False, "gap must be between 0.6 and 1.5 m")
    assert controller.set_follow_gap(1.25) == (True, "Gap set to 1.25 m")
    assert controller.state.snapshot()["follow_gap"] == 1.25


def test_action_bundle_ships_the_follow_runner_and_its_modules():
    bundle = action_bundle_paths()
    assert FOLLOW_RUNNER in bundle
    assert all(module in bundle for module in FOLLOW_MODULES)


def test_follow_runs_as_a_pep723_script():
    command = remote_script_command("/tmp/robot_follow.py", "--gap", "1.00")
    assert 'exec "$HOME/.local/bin/uv" run --script /tmp/robot_follow.py --gap 1.00' in command


class FakeFollowProcess:
    """Stands in for the ssh process: records stdin lines, emits runner output."""

    def __init__(self, command, **kwargs):
        self.command = command
        self.lines = []
        self.done = threading.Event()
        self.stdin = self
        self.stdout = self._output()

    def write(self, text):
        self.lines.append(json.loads(text))
        if self.lines[-1]["type"] == "stop":
            self.done.set()

    def flush(self):
        pass

    def _output(self):
        yield "[follow] follow active (v_max 0.15 m/s, gap 1.00 m) - raise a hand\n"
        yield 'FOLLOW_STATUS {"state":"SEARCHING","range":null}\n'
        self.done.wait(5)
        yield "[follow] exit: stop\n"

    def poll(self):
        return 0 if self.done.is_set() else None

    def wait(self):
        self.done.wait(5)
        return 0


def test_follow_robot_mode_streams_heartbeats_gap_and_stop(monkeypatch):
    processes, remote_stops = [], []

    def fake_popen(command, **kwargs):
        processes.append(FakeFollowProcess(command, **kwargs))
        return processes[-1]

    monkeypatch.setattr("scripts.robot_dashboard.subprocess.Popen", fake_popen)
    controller = RobotController(("bot",))
    controller.state.host = "bot"
    monkeypatch.setattr(controller, "_deploy", lambda host, *paths: False)
    monkeypatch.setattr(controller, "_request_remote_stop",
                        lambda host, pid_file, name: remote_stops.append(pid_file))

    assert controller.set_follow(True) == (True, "Starting follow mode")
    wait_for(lambda: controller.state.snapshot()["follow_enabled"])
    process = processes[0]
    assert "run --script /tmp/robot_follow.py --gap 1.00 --pid-file /tmp/bracketbot-follow-1.pid" in process.command[-1]
    wait_for(lambda: sum(line["type"] == "heartbeat" for line in process.lines) >= 2)
    assert controller.state.snapshot()["follow_status"] == {"state": "SEARCHING", "range": None}

    controller.set_follow_gap(1.3)
    assert {"type": "gap", "m": 1.3} in process.lines

    assert controller.stop() == (True, "Stop requested")
    wait_for(lambda: follow_off(controller))
    assert process.lines[-1] == {"type": "stop"}
    assert remote_stops == ["/tmp/bracketbot-follow-1.pid"]
    state = controller.state.snapshot()
    assert state["follow_phase"] == "Follow off"
    assert state["error"] is None
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_robot_dashboard.py -q`

Expected: FAIL; the output contains `cannot import name 'FOLLOW_GAP_DEFAULT'`.

- [ ] **Step 4: Apply the dashboard edits in order (each `Find` text occurs exactly once)**

**Edit 1 — import `math`.** Find:

```python
import json
from pathlib import Path
```

Replace with:

```python
import json
import math
from pathlib import Path
```

**Edit 2 — follow constants.** Find:

```python
REMOTE_BASE_RUNNER = "/tmp/robot_base_mode.py"
```

Replace with:

```python
REMOTE_BASE_RUNNER = "/tmp/robot_base_mode.py"
FOLLOW_RUNNER = ROOT / "scripts" / "robot_follow.py"
FOLLOW_MODULES = (ROOT / "scripts" / "follow_core.py", ROOT / "scripts" / "follow_perception.py")
REMOTE_FOLLOW_RUNNER = "/tmp/robot_follow.py"
# Must match follow_core (FollowConfig gap bounds and STATUS_PREFIX); a test checks this.
FOLLOW_GAP_MIN = 0.6
FOLLOW_GAP_MAX = 1.5
FOLLOW_GAP_DEFAULT = 1.0
FOLLOW_STATUS_PREFIX = "FOLLOW_STATUS "
FOLLOW_HEARTBEAT_PERIOD = 0.25
```

**Edit 3 — ship the follow files with the action bundle.** Find:

```python
    paths = [RUNNER, EFFECT_RUNNER, BASE_RUNNER]
```

Replace with:

```python
    paths = [RUNNER, EFFECT_RUNNER, BASE_RUNNER, FOLLOW_RUNNER, *FOLLOW_MODULES]
```

**Edit 4 — `remote_script_command` (before `class DashboardState`).** Find:

```python
class DashboardState:
```

Replace with:

```python
def remote_script_command(script, *args):
    """Run a PEP 723 script in its own uv environment, as the bbapps apps run."""
    payload = " ".join(shlex.quote(str(value)) for value in (script, *args))
    return f'export PATH="$HOME/.local/bin:$PATH"; exec "$HOME/.local/bin/uv" run --script {payload}'


class DashboardState:
```

**Edit 5 — follow state fields.** Find:

```python
        self.lean_process = None
        self.lean_pid_file = None

    def snapshot(self):
```

Replace with:

```python
        self.lean_process = None
        self.lean_pid_file = None
        self.follow_enabled = False
        self.follow_requested = False
        self.follow_transition = False
        self.follow_phase = "Follow off"
        self.follow_status = None
        self.follow_gap = FOLLOW_GAP_DEFAULT
        self.follow_process = None
        self.follow_pid_file = None

    def follow_active(self):
        """Caller holds ``lock``."""
        return self.follow_enabled or self.follow_requested or self.follow_transition

    def snapshot(self):
```

**Edit 6 — follow fields in `snapshot()`.** Find:

```python
                "lean_phase": self.lean_phase,
            }
```

Replace with:

```python
                "lean_phase": self.lean_phase,
                "follow_enabled": self.follow_enabled,
                "follow_transition": self.follow_transition,
                "follow_phase": self.follow_phase,
                "follow_status": self.follow_status,
                "follow_gap": self.follow_gap,
                "follow_gap_min": FOLLOW_GAP_MIN,
                "follow_gap_max": FOLLOW_GAP_MAX,
            }
```

**Edit 7 — `add_follow_log`.** Find:

```python
            self.log.append(f"[lean] {line}")
            del self.log[:-80]
            self.lean_phase = line
```

Replace with:

```python
            self.log.append(f"[lean] {line}")
            del self.log[:-80]
            self.lean_phase = line

    def add_follow_log(self, line):
        line = line.strip()
        if not line:
            return
        if line.startswith(FOLLOW_STATUS_PREFIX):
            try:
                status = json.loads(line[len(FOLLOW_STATUS_PREFIX):])
            except json.JSONDecodeError:
                return
            with self.lock:
                self.follow_status = status
            return
        with self.lock:
            self.log.append(f"[follow] {line.removeprefix('[follow] ')}")
            del self.log[:-80]
            self.follow_phase = line.removeprefix("[follow] ")
```

**Edit 8 — follow counters in `RobotController.__init__`.** Find:

```python
        self._lean_counter = 0
```

Replace with:

```python
        self._lean_counter = 0
        self._follow_counter = 0
        self._follow_write_lock = threading.Lock()
```

**Edit 9 — actions refuse while following.** Find:

```python
            if self.state.running:
                return False, f"{self.state.action} is already running"
            host = self.state.host
            if host is None:
                return False, "Robot is not connected; choose Reconnect"
            self.state.running = True
```

Replace with:

```python
            if self.state.running:
                return False, f"{self.state.action} is already running"
            if self.state.follow_active():
                return False, "Stop following before running actions"
            host = self.state.host
            if host is None:
                return False, "Robot is not connected; choose Reconnect"
            self.state.running = True
```

**Edit 10 — lean refuses while following.** Find:

```python
            if self.state.lean_transition:
                return False, "Lean mode is already changing"
```

Replace with:

```python
            if self.state.lean_transition:
                return False, "Lean mode is already changing"
            if enabled and self.state.follow_active():
                return False, "Stop following before enabling lean"
```

**Edit 11 — follow control methods (inserted before `stop()`) and the head of `stop()`.** Find:

```python
    def stop(self):
        with self.state.lock:
            action_running = self.state.running
            lean_running = (
                self.state.lean_enabled
                or self.state.lean_requested
                or self.state.lean_transition
            )
            if not action_running and not lean_running:
                return False, "No action is running"
```

Replace with:

```python
    def set_follow(self, enabled):
        if not isinstance(enabled, bool):
            return False, "enabled must be true or false"
        with self.state.lock:
            state = self.state
            if state.host is None:
                return False, "Robot is not connected; choose Reconnect"
            if state.follow_transition:
                return False, "Follow mode is already changing"
            if enabled == state.follow_enabled:
                return False, "Follow is already on" if enabled else "Follow is already off"
            if enabled:
                if state.running:
                    return False, f"{state.action} is running; stop it first"
                if state.lean_enabled or state.lean_requested or state.lean_transition:
                    return False, "Return to balance mode before following"
                self._follow_counter += 1
                pid_file = f"/tmp/bracketbot-follow-{self._follow_counter}.pid"
                state.follow_pid_file = pid_file
                state.follow_requested = True
                state.follow_transition = True
                state.follow_status = None
                state.error = None
                state.follow_phase = "Starting follow…"
                host = state.host
        if not enabled:
            self._stop_follow()
            return True, "Stopping follow mode"
        target = self._simulate_follow if self.state.simulate else self._run_follow
        threading.Thread(
            target=target, args=(host, pid_file), name="follow-control", daemon=True
        ).start()
        return True, "Starting follow mode"

    def set_follow_gap(self, gap):
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap):
            return False, "gap must be a number of metres"
        if not FOLLOW_GAP_MIN <= gap <= FOLLOW_GAP_MAX:
            return False, f"gap must be between {FOLLOW_GAP_MIN} and {FOLLOW_GAP_MAX} m"
        with self.state.lock:
            self.state.follow_gap = round(float(gap), 2)
            gap = self.state.follow_gap
        self._send_follow({"type": "gap", "m": gap})
        return True, f"Gap set to {gap:.2f} m"

    def _send_follow(self, message, process=None):
        with self.state.lock:
            process = process or self.state.follow_process
        if process is None or process.stdin is None:
            return False
        try:
            with self._follow_write_lock:
                process.stdin.write(json.dumps(message) + "\n")
                process.stdin.flush()
        except (OSError, ValueError):
            return False
        return True

    def _follow_heartbeat(self, process):
        # The runner stops by itself when these stop arriving (dashboard or link loss).
        while process.poll() is None and self._send_follow({"type": "heartbeat"}, process):
            time.sleep(FOLLOW_HEARTBEAT_PERIOD)

    def _stop_follow(self):
        with self.state.lock:
            state = self.state
            if not state.follow_active():
                return
            state.follow_requested = False
            state.follow_transition = True
            state.follow_phase = "Stopping follow…"
            host = state.host
            pid_file = state.follow_pid_file
        if self.state.simulate:
            return
        self._send_follow({"type": "stop"})
        self._request_remote_stop(host, pid_file, "follow-stop")

    def _run_follow(self, host, pid_file):
        return_code = None
        try:
            self._deploy(host, FOLLOW_RUNNER, *FOLLOW_MODULES)
            with self.state.lock:
                if not self.state.follow_requested:
                    return
                gap = self.state.follow_gap
            remote_command = remote_script_command(
                REMOTE_FOLLOW_RUNNER, "--gap", f"{gap:.2f}", "--pid-file", pid_file, *self.follow_args
            )
            process = subprocess.Popen(
                ["ssh", *SSH_OPTIONS, host, remote_command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self.state.lock:
                self.state.follow_process = process
            threading.Thread(
                target=self._follow_heartbeat, args=(process,), name="follow-heartbeat", daemon=True
            ).start()
            assert process.stdout is not None
            for line in process.stdout:
                self.state.add_follow_log(line)
                if "follow active" in line:
                    with self.state.lock:
                        self.state.follow_enabled = True
                        self.state.follow_transition = False
            return_code = process.wait()
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            with self.state.lock:
                self.state.error = f"Follow control failed: {exc}"
        finally:
            with self.state.lock:
                state = self.state
                unexpected = state.follow_requested
                if unexpected and state.error is None:
                    state.error = f"Follow stopped (status {return_code}): {state.follow_phase}"
                state.follow_enabled = False
                state.follow_requested = False
                state.follow_transition = False
                state.follow_process = None
                state.follow_pid_file = None
                state.follow_status = None
                state.follow_phase = "Follow stopped unexpectedly" if unexpected else "Follow off"

    def _simulate_follow(self, host, pid_file):
        started = time.monotonic()
        simulated_range = 1.6
        with self.state.lock:
            self.state.follow_enabled = True
            self.state.follow_transition = False
            self.state.follow_phase = "Following (simulation)"
            self.state.log.append("[follow] simulated follow active")
        try:
            while True:
                with self.state.lock:
                    if not self.state.follow_requested:
                        return
                    gap = self.state.follow_gap
                searching = time.monotonic() - started < 1.0
                if not searching:
                    simulated_range += (gap - simulated_range) * 0.1
                status = {
                    "state": "SEARCHING" if searching else "FOLLOWING",
                    "range": None if searching else round(simulated_range, 3),
                    "gap": gap,
                    "error": None if searching else round(simulated_range - gap, 3),
                    "bearing_deg": None if searching else 0.0,
                    "v": 0.0,
                    "w": 0.0,
                    "blocked": False,
                    "age_ms": None if searching else 60,
                    "rule": "no-track" if searching else "ok",
                }
                with self.state.lock:
                    self.state.follow_status = status
                time.sleep(0.05)
        finally:
            with self.state.lock:
                self.state.follow_enabled = False
                self.state.follow_requested = False
                self.state.follow_transition = False
                self.state.follow_status = None
                self.state.follow_pid_file = None
                self.state.follow_phase = "Follow off"
                self.state.log.append("[follow] simulated follow stopped")

    def stop(self):
        with self.state.lock:
            action_running = self.state.running
            lean_running = (
                self.state.lean_enabled
                or self.state.lean_requested
                or self.state.lean_transition
            )
            follow_running = self.state.follow_active()
            if not action_running and not lean_running and not follow_running:
                return False, "No action is running"
```

**Edit 12 — `stop()` also stops follow.** Find:

```python
        if not self.state.simulate:
            if action_running:
                self._request_remote_stop(host, pid_file, "action-stop")
            if lean_running:
                self._request_remote_stop(host, lean_pid_file, "lean-stop")
        return True, "Stop requested"
```

Replace with:

```python
        if follow_running:
            self._stop_follow()
        if not self.state.simulate:
            if action_running:
                self._request_remote_stop(host, pid_file, "action-stop")
            if lean_running:
                self._request_remote_stop(host, lean_pid_file, "lean-stop")
        return True, "Stop requested"
```

**Edit 13 — CSS for four controls and the slider.** Find:

```python
.controls { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:18px; }
```

Replace with:

```python
.controls { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:18px; }
.follow-gap { display:flex; align-items:center; gap:14px; margin-top:14px; color:var(--muted); font-size:.9rem; font-weight:700; }
.follow-gap input { flex:1; accent-color:var(--red); }
```

**Edit 14 — follow status line (HTML).** Find:

```python
      <p id="base-detail">Base: balance mode</p>
```

Replace with:

```python
      <p id="base-detail">Base: balance mode</p>
      <p id="follow-detail">Follow: off</p>
```

**Edit 15 — style the follow status line.** Find:

```python
#detail, #base-detail, #latency-detail {
```

Replace with:

```python
#detail, #base-detail, #follow-detail, #latency-detail {
```

**Edit 16 — Follow button and distance slider (HTML).** Find:

```python
    <button id="lean" class="secondary"><span class="key">Z</span>Enable lean</button>
    <button id="stop" class="stop" disabled><span class="key">Esc</span>Stop action</button>
  </div>
```

Replace with:

```python
    <button id="lean" class="secondary"><span class="key">Z</span>Enable lean</button>
    <button id="follow" class="secondary" aria-pressed="false"><span class="key">F</span>Follow me</button>
    <button id="stop" class="stop" disabled><span class="key">Esc</span>Stop action</button>
  </div>
  <div class="follow-gap">
    <label for="gap">Following distance</label>
    <input id="gap" type="range" min="0.6" max="1.5" step="0.1" value="1.0">
    <output id="gap-value" for="gap">1.0 m</output>
  </div>
```

**Edit 17 — JS element handles and `followText`.** Find:

```python
const latencyDetail=document.getElementById('latency-detail');
```

Replace with:

```python
const latencyDetail=document.getElementById('latency-detail');
const follow=document.getElementById('follow'), followDetail=document.getElementById('follow-detail');
const gap=document.getElementById('gap'), gapValue=document.getElementById('gap-value');
let gapDragging=false;
function followText(c) {
  if(!c.follow_enabled&&!c.follow_transition) return 'Follow: off';
  const s=c.follow_status;
  if(!s) return `Follow: ${c.follow_phase}`;
  if(s.state==='SEARCHING') return 'Follow: raise a hand to be followed';
  const range=s.range==null?'—':`${s.range.toFixed(2)} m`;
  const err=s.error==null?'':` · ${s.error>=0?'+':''}${Math.round(s.error*100)} cm from target`;
  return `Follow: ${s.state.toLowerCase()} · ${range}${err}`;
}
```

**Edit 18 — JS `refresh()` follow wiring.** Find:

```python
    const ready=current.connected&&!current.running&&!current.checking;
    buttons.forEach(b=>b.disabled=!ready); stop.disabled=!(current.running||current.lean_enabled||current.lean_transition);
    reconnect.disabled=current.running||current.checking||current.lean_enabled||current.lean_transition;
    lean.disabled=!current.connected||current.lean_transition;
```

Replace with:

```python
    const followOn=current.follow_enabled||current.follow_transition;
    const ready=current.connected&&!current.running&&!current.checking&&!followOn;
    buttons.forEach(b=>b.disabled=!ready); stop.disabled=!(current.running||current.lean_enabled||current.lean_transition||followOn);
    reconnect.disabled=current.running||current.checking||current.lean_enabled||current.lean_transition||followOn;
    lean.disabled=!current.connected||current.lean_transition||(followOn&&!current.lean_enabled);
    follow.disabled=!current.connected||current.follow_transition||(!current.follow_enabled&&(current.running||current.lean_enabled||current.lean_transition));
    follow.className='secondary '+(current.follow_enabled?'lean-active':'');
    follow.setAttribute('aria-pressed',String(current.follow_enabled));
    follow.innerHTML=`<span class="key">F</span>${current.follow_transition?'Changing follow…':current.follow_enabled?'Stop following':'Follow me'}`;
    followDetail.textContent=followText(current);
    gap.min=current.follow_gap_min; gap.max=current.follow_gap_max;
    if(!gapDragging){ gap.value=current.follow_gap; gapValue.textContent=`${Number(current.follow_gap).toFixed(1)} m`; }
```

**Edit 19 — JS button and slider listeners.** Find:

```python
lean.addEventListener('click',()=>post('/api/lean',{enabled:!current.lean_enabled}).then(refresh));
```

Replace with:

```python
lean.addEventListener('click',()=>post('/api/lean',{enabled:!current.lean_enabled}).then(refresh));
follow.addEventListener('click',()=>post('/api/follow',{enabled:!current.follow_enabled}).then(refresh));
gap.addEventListener('input',()=>{ gapDragging=true; gapValue.textContent=`${Number(gap.value).toFixed(1)} m`; });
gap.addEventListener('change',()=>{ gapDragging=false; post('/api/follow/gap',{gap:Number(gap.value)}).then(refresh); });
```

**Edit 20 — JS keys: Esc first (works with the slider focused), then F.** Find:

```python
  if(event.repeat||event.target.matches('input,textarea,select')) return;
  if(event.key==='Escape'&&(current.running||current.lean_enabled||current.lean_transition)){ event.preventDefault(); stop.click(); return; }
  if(event.key.toLowerCase()==='z'&&!lean.disabled){ event.preventDefault(); lean.click(); return; }
```

Replace with:

```python
  if(event.key==='Escape'&&!stop.disabled){ event.preventDefault(); stop.click(); return; }
  if(event.repeat||event.target.matches('input,textarea,select')) return;
  if(event.key.toLowerCase()==='z'&&!lean.disabled){ event.preventDefault(); lean.click(); return; }
  if(event.key.toLowerCase()==='f'&&!follow.disabled){ event.preventDefault(); follow.click(); return; }
```

**Edit 21 — HTTP routes.** Find:

```python
        elif self.path == "/api/stop":
```

Replace with:

```python
        elif self.path == "/api/follow":
            ok, message = self.controller.set_follow(self._json_body().get("enabled"))
            self._send({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.CONFLICT)
        elif self.path == "/api/follow/gap":
            ok, message = self.controller.set_follow_gap(self._json_body().get("gap"))
            self._send({"ok": ok, "message": message}, HTTPStatus.ACCEPTED if ok else HTTPStatus.BAD_REQUEST)
        elif self.path == "/api/stop":
```

**Edit 22 — shutdown waits for follow to stop.** Find:

```python
            if not state["running"] and not state["lean_enabled"] and not state["lean_transition"]:
                break
```

Replace with:

```python
            if not (state["running"] or state["lean_enabled"] or state["lean_transition"]
                    or state["follow_enabled"] or state["follow_transition"]):
                break
```

**Edit 23 — `follow_args` launch option.** Find:

```python
    def __init__(self, ssh_hosts, simulate=False):
        self.state = DashboardState(ssh_hosts, simulate=simulate)
```

Replace with:

```python
    def __init__(self, ssh_hosts, simulate=False, follow_args=()):
        self.state = DashboardState(ssh_hosts, simulate=simulate)
        self.follow_args = tuple(follow_args)
```

**Edit 24 — gate launch flags in `main()`.** Find:

```python
        help="exercise actions and routines locally without SSH or robot hardware",
    )
    args = parser.parse_args()

    controller = RobotController(args.ssh_hosts, simulate=args.simulate)
```

Replace with:

```python
        help="exercise actions and routines locally without SSH or robot hardware",
    )
    parser.add_argument(
        "--follow-v-max",
        type=float,
        default=0.15,
        help="follow speed cap in m/s (0.15 until robot gate G4b passes; at most 0.30)",
    )
    parser.add_argument(
        "--follow-rotate-only",
        action="store_true",
        help="follow by turning in place only (robot gate G3)",
    )
    args = parser.parse_args()
    if not 0.0 < args.follow_v_max <= 0.30:
        parser.error("--follow-v-max must be above 0 and at most 0.30 m/s")
    follow_args = ("--v-max", f"{args.follow_v_max:.2f}")
    if args.follow_rotate_only:
        follow_args += ("--rotate-only",)

    controller = RobotController(args.ssh_hosts, simulate=args.simulate, follow_args=follow_args)
```

- [ ] **Step 5: Run the dashboard tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_robot_dashboard.py -q`

Expected: PASS — `22 passed`.

- [ ] **Step 6: Check the page script parses and try it in simulation**

```bash
uv run --extra dev python -c "import re,sys; sys.path.insert(0,'scripts'); import robot_dashboard as d; open('page.js','w').write(re.search(r'<script>(.*)</script>', d.PAGE, re.S).group(1))" && node --check page.js && rm page.js
python scripts/robot_dashboard.py --simulate   # open http://127.0.0.1:8020
```

In the browser: **Follow me** shows "raise a hand to be followed" for about a second, then a range converging on the slider value; moving the slider changes it; `Esc` (also with the slider focused) and **Stop action** end follow; gestures and **Enable lean** are disabled while following.

- [ ] **Step 7: Document follow mode in the README**

In `README.md`, replace:

```markdown
| Base mode | Toggle 4° lean / balance | `Z` |
```

with:

```markdown
| Base mode | Toggle 4° lean / balance | `Z` |
| Follow | Follow the person who raises a hand; distance slider | `F` |
```

In `README.md`, replace:

```markdown
## Run local person and expression detection
```

with:

```markdown
## Follow mode (person following)

**Follow me** (`F`) starts `scripts/robot_follow.py` on the robot. The person
who then raises a hand above their head for about half a second is locked on.
The robot holds the **Following distance** slider's gap (0.6–1.5 m, default
1.0 m) to within ±20 cm while that person stands, turns, or walks slowly. The
base is clamped to 0.3 m/s, so it cannot keep pace with normal walking and
catches up when they pause. It never drives backward, stops for anything in a
0.6 m corridor ahead, and stops by itself if the dashboard, the link, or
balance is lost. `Esc` stops it like every other action; `--simulate` exercises
the whole path without a robot.

Do not use follow mode around people until every robot gate in
`docs/superpowers/plans/2026-09-19-person-follow.md` (Task 11) has passed, with
a person at the physical e-stop for each gate that moves the robot. Until gate
G4b passes, the speed cap is 0.15 m/s; `--follow-v-max 0.30` and
`--follow-rotate-only` exist for the gates. Design:
`docs/superpowers/specs/2026-09-19-person-follow-design.md`.

## Run local person and expression detection
```

- [ ] **Step 8: Run the whole suite**

Run: `uv run --extra dev python -m pytest -q`

Expected: PASS — `110 passed, 1 skipped`.
(The skip is the Ultralytics cross-check, which needs the `vision` extra.)

- [ ] **Step 9: Commit**

```bash
git add scripts/robot_dashboard.py tests/test_robot_dashboard.py README.md
git commit -m "feat(dashboard): follow mode with distance slider and heartbeat" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Robot-gate tooling: probe, alignment check, engine build, log report

**Files:**
- Create: `scripts/follow_log_report.py`
- Test: `tests/test_follow_log_report.py`
- Create: `scripts/probe_follow.py`
- Create: `scripts/check_follow_alignment.py`
- Create: `scripts/build_pose_engine.sh`

**Interfaces:**
- Consumes: `robot_follow.CSV_FIELDS` (Task 7).
- Produces: `follow_log_report.summarise(rows) -> dict` with `samples following_samples in_band_fraction abs_error_p95_m bearing_within_10deg_fraction v_sign_agreement omega_sign_agreement blocked_to_stop_s rules last_rule`;
  the read-only probe (`--out DIR [--person-left]`, writes `report.json`, `points.npz`, `camera_*.npy`);
  `check_follow_alignment.py PROBE_DIR` (writes `alignment.png`); `build_pose_engine.sh ONNX` (writes the engine and a `.txt` version record).

Task 11 uses these to turn each gate into numbers. Only the log report has laptop logic worth unit-testing; the probe and build script are robot-only and are syntax-checked.

- [ ] **Step 1: Write the failing log-report tests**

Create `tests/test_follow_log_report.py`:

```python
import csv

import pytest

from follow_log_report import load, summarise
from robot_follow import CSV_FIELDS


def write_log(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def row(t, state="FOLLOWING", error=0.0, bearing=0.0, v=0.1, measured_v=0.1, omega=0.0,
        measured_omega=0.0, rule="ok"):
    return {"t": t, "state": state, "rule": rule, "gap": 1.0,
            "range": None if error is None else 1.0 + error, "error": error, "bearing": bearing,
            "v": v, "measured_v": measured_v, "omega": omega, "measured_omega": measured_omega}


def test_report_on_a_synthetic_log(tmp_path):
    rows = [row(0.00, state="SEARCHING", error=None, bearing=None, v=0.0, measured_v=0.0, rule="no-track")]
    rows += [row(0.05 * i, error=0.05, bearing=0.02) for i in range(1, 19)]
    rows += [row(0.95, error=0.30, bearing=0.5)]  # one out-of-band, off-axis sample
    rows += [row(1.00, state="BLOCKED", v=0.06, rule="blocked"), row(1.05, state="BLOCKED", v=0.0, rule="blocked")]
    rows += [row(1.10, v=0.2, measured_v=-0.1, omega=0.5, measured_omega=0.4, rule="heartbeat")]
    path = tmp_path / "log.csv"
    write_log(path, rows)

    report = summarise(load(path))

    assert report["samples"] == len(rows)
    assert report["following_samples"] == 20
    assert report["in_band_fraction"] == pytest.approx(19 / 20)
    assert report["abs_error_p95_m"] == pytest.approx(0.05)
    assert report["bearing_within_10deg_fraction"] == pytest.approx(19 / 20)
    assert report["v_sign_agreement"] == pytest.approx(20 / 21)
    assert report["omega_sign_agreement"] == 1.0
    assert report["blocked_to_stop_s"] == pytest.approx(0.05)
    assert report["rules"]["blocked"] == 2
    assert report["last_rule"] == "heartbeat"


def test_report_handles_a_log_with_no_following(tmp_path):
    path = tmp_path / "log.csv"
    write_log(path, [row(0.0, state="SEARCHING", error=None, bearing=None, v=0.0, rule="no-track")])
    report = summarise(load(path))
    assert report["in_band_fraction"] is None
    assert report["v_sign_agreement"] is None
    assert report["blocked_to_stop_s"] is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_follow_log_report.py -q`

Expected: FAIL; the output contains `No module named 'follow_log_report'`.

- [ ] **Step 3: Write `scripts/follow_log_report.py`**

Create `scripts/follow_log_report.py`:

```python
"""Summarise a robot_follow CSV log into the numbers the robot gates check.

    scp 'bot:/tmp/baymax_follow_*.csv' artifacts/follow/
    python scripts/follow_log_report.py artifacts/follow/baymax_follow_20260920_101500.csv

Standard library only. Fractions are over FOLLOWING samples; sign agreement
compares what was sent with wheel feedback while the command was clearly nonzero.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
import sys


def load(path):
    with open(path, newline="") as source:
        return list(csv.DictReader(source))


def num(row, key):
    value = row.get(key, "")
    return None if value in ("", "None") else float(value)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1)]


def sign_agreement(rows, sent, measured, threshold):
    pairs = [(num(r, sent), num(r, measured)) for r in rows if abs(num(r, sent) or 0.0) >= threshold]
    if not pairs:
        return None
    return sum(a * b > 0 for a, b in pairs) / len(pairs)


def blocked_to_stop(rows):
    """Seconds from the first BLOCKED sample until the sent speed reached zero."""
    for i, row in enumerate(rows):
        if row["state"] == "BLOCKED":
            start = num(row, "t")
            for later in rows[i:]:
                if (num(later, "v") or 0.0) == 0.0:
                    return round(num(later, "t") - start, 3)
            return None
    return None


def summarise(rows, band=0.20):
    following = [r for r in rows if r["state"] == "FOLLOWING" and num(r, "error") is not None]
    errors = [num(r, "error") for r in following]
    bearings = [abs(math.degrees(num(r, "bearing"))) for r in following if num(r, "bearing") is not None]
    return {
        "samples": len(rows),
        "following_samples": len(following),
        "in_band_fraction": None if not errors else round(sum(abs(e) <= band for e in errors) / len(errors), 3),
        "abs_error_p95_m": None if not errors else round(percentile([abs(e) for e in errors], 95), 3),
        "bearing_within_10deg_fraction": None if not bearings else round(sum(b <= 10 for b in bearings) / len(bearings), 3),
        "v_sign_agreement": sign_agreement(rows, "v", "measured_v", 0.05),
        "omega_sign_agreement": sign_agreement(rows, "omega", "measured_omega", 0.2),
        "blocked_to_stop_s": blocked_to_stop(rows),
        "rules": dict(Counter(r["rule"] for r in rows)),
        "last_rule": rows[-1]["rule"] if rows else None,
    }


def main(argv=None):
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        raise SystemExit("usage: follow_log_report.py LOG.csv [LOG.csv ...]")
    for path in paths:
        print(path)
        print(json.dumps(summarise(load(path)), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_follow_log_report.py -q`

Expected: PASS — `2 passed`.

- [ ] **Step 5: Write the read-only probe `scripts/probe_follow.py`**

Create `scripts/probe_follow.py`:

```python
"""Read-only probe for person-follow gate G0. Runs ON THE ROBOT and opens no writers.

    scp scripts/probe_follow.py bot:/tmp/
    ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/probe_follow.py --out /tmp/follow_probe'
    ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/probe_follow.py --out /tmp/follow_probe_left --person-left'
    scp -r bot:/tmp/follow_probe bot:/tmp/follow_probe_left artifacts/

First run: nobody in front of the robot (measures the robot's own body in the
depth cloud). --person-left: one person stands about 1 m ahead and 0.5 m to the
robot's LEFT (tells which way base +x points).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from bbos import Config, Reader

TOPICS = ("camera.rect", "camera.depth", "camera.points", "camera.head",
          "imu.orientation", "drive.state", "drive.status")
CONFIGS = ("drive", "depth", "cam_head", "base")


def field_names(data):
    names = getattr(getattr(data, "dtype", None), "names", None)
    if names:
        return list(names)
    try:
        return list(data.keys())
    except AttributeError:
        return []


def describe(value):
    array = np.asarray(value)
    info = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size <= 8:
        info["value"] = array.tolist()
    return info


def sample_topic(topic, seconds):
    frames, stamps, first = 0, [], None
    try:
        with Reader(topic, keeptime=False) as reader:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                if not reader.ready():
                    time.sleep(0.002)
                    continue
                data = reader.data
                frames += 1
                if first is None:
                    first = {name: np.array(data[name]).copy() for name in field_names(data)}
                if first and "timestamp" in first:
                    stamps.append(float(np.asarray(data["timestamp"]).item()))
    except Exception as exc:  # report and keep probing the other topics
        return {"error": repr(exc)}, None
    report = {"rate_hz": round(frames / seconds, 1),
              "fields": {name: describe(value) for name, value in (first or {}).items()}}
    if len(stamps) > 1:
        report["timestamp_first"] = stamps[0]
        report["timestamp_span_over_window"] = stamps[-1] - stamps[0]  # ~seconds, ms, or ns: tells the unit
    return report, first


def rect_points_pairing(seconds=2.0):
    diffs = []
    with Reader("camera.rect", keeptime=False) as rect, Reader("camera.points", keeptime=False) as points:
        last_rect = None
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if rect.ready():
                last_rect = float(np.asarray(rect.data["timestamp"]).item())
            if points.ready() and last_rect is not None:
                diffs.append(float(np.asarray(points.data["timestamp"]).item()) - last_rect)
            time.sleep(0.001)
    return {"pairs": len(diffs), "min": min(diffs, default=None), "max": max(diffs, default=None)}


def config_values(name):
    try:
        cfg = Config(name)
    except Exception as exc:
        return {"error": repr(exc)}
    values = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        try:
            value = getattr(cfg, key)
        except Exception:
            continue
        if isinstance(value, (bool, int, float, str)):
            values[key] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 16:
            values[key] = [v if isinstance(v, (bool, int, float, str)) else repr(v) for v in value]
    return values


def cloud_summary(points, person_left):
    p = np.asarray(points, dtype=float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    near = (y < 0.45) & (np.abs(x) < 0.4) & (z > 0.05) & (z < 1.7)
    summary = {"num_points": int(len(p)), "self_points": int(near.sum())}
    if near.any():
        summary["self_box_base_xyz_min"] = [round(float(v), 3) for v in p[near].min(axis=0)]
        summary["self_box_base_xyz_max"] = [round(float(v), 3) for v in p[near].max(axis=0)]
    if person_left:
        body = (y > 0.6) & (y < 1.6) & (z > 0.8) & (z < 1.6) & (np.abs(x) < 1.0)
        median_x = float(np.median(x[body])) if body.any() else None
        summary["person_points"] = int(body.sum())
        summary["person_median_base_x"] = median_x
        if median_x is not None:
            summary["verdict"] = (
                "+x is LEFT: set BASE_LEFT_SIGN = +1.0" if median_x > 0
                else "+x is RIGHT: keep BASE_LEFT_SIGN = -1.0"
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Read-only probe for person-follow gate G0")
    parser.add_argument("--out", type=Path, default=Path("/tmp/follow_probe"))
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--person-left", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"topics": {}, "configs": {name: config_values(name) for name in CONFIGS}}
    frames = {}
    for topic in TOPICS:
        report["topics"][topic], frames[topic] = sample_topic(topic, args.seconds)
    try:
        report["rect_minus_points_timestamp"] = rect_points_pairing()
    except Exception as exc:
        report["rect_minus_points_timestamp"] = {"error": repr(exc)}

    points = frames.get("camera.points")
    if points and "points" in points:
        n = int(np.asarray(points["num_points"]).item())
        report["cloud"] = cloud_summary(points["points"][:n], args.person_left)
        np.savez_compressed(args.out / "points.npz", points=points["points"][:n], mask=points["mask"][:n])
    for topic, name in (("camera.rect", "rgb"), ("camera.depth", "depth"), ("camera.head", "rgb")):
        frame = frames.get(topic)
        if frame and name in frame:
            np.save(args.out / f"{topic.replace('.', '_')}.npy", frame[name])

    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=repr))
    print(json.dumps(report, indent=2, default=repr))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write `scripts/check_follow_alignment.py`**

Create `scripts/check_follow_alignment.py`:

```python
"""Laptop check for gate G0: does camera.rect line up with camera.depth?

    uv run --extra vision python scripts/check_follow_alignment.py artifacts/follow_probe

Writes alignment.png: the rect image with depth edges drawn in magenta. If the
edges sit on the object outlines (door frames, table edges, a person), the
detection image and camera.points agree and follow_perception can map depth
pixels by scaling. If they are visibly offset, use the fisheye fallback (plan Task 9).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dir", type=Path)
    args = parser.parse_args()
    rect = np.load(args.probe_dir / "camera_rect.npy")
    if rect.shape[1] > 2.5 * rect.shape[0]:
        rect = rect[:, : rect.shape[1] // 2]
    depth = np.load(args.probe_dir / "camera_depth.npy").astype(np.float32)
    depth = cv2.resize(depth, (rect.shape[1], rect.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = depth > 0
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.clip(depth[valid] / 5000.0 * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(scaled, 20, 60) > 0
    overlay = cv2.cvtColor(np.ascontiguousarray(rect[..., :3]), cv2.COLOR_RGB2BGR)
    overlay[edges] = (255, 0, 255)
    out = args.probe_dir / "alignment.png"
    cv2.imwrite(str(out), overlay)
    print(f"rect {rect.shape}, depth {depth.shape} -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Write `scripts/build_pose_engine.sh`**

Create `scripts/build_pose_engine.sh`:

```bash
#!/usr/bin/env bash
# Build the yolo11n-pose TensorRT engine ON THE JETSON (robot gate G1).
# Engines are tied to the exact JetPack/TensorRT/GPU: never copy one between machines.
#
#   scp yolo11n-pose.onnx scripts/build_pose_engine.sh bot:/tmp/
#   ssh bot 'bash /tmp/build_pose_engine.sh /tmp/yolo11n-pose.onnx'
set -euo pipefail
onnx="${1:-/tmp/yolo11n-pose.onnx}"
out_dir="$HOME/.cache/baymax"
engine="$out_dir/yolo11n-pose.engine"
trtexec="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$out_dir"
"$trtexec" --onnx="$onnx" --saveEngine="$engine" --fp16
{
  echo "built: $(date -Is)"
  echo "onnx_sha256: $(sha256sum "$onnx" | cut -d' ' -f1)"
  echo "checkpoint_sha256: 869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
  echo "input: 1x3x640x640 fp32 io, fp16 layers"
  echo "l4t: $(head -n1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
  dpkg-query -W -f='${Package}=${Version}\n' 'tensorrt*' 'libnvinfer*' 2>/dev/null | sort -u
} > "$engine.txt"
echo "engine: $engine"
cat "$engine.txt"
```

- [ ] **Step 8: Syntax-check the robot-only scripts and run the whole suite**

```bash
python -m py_compile scripts/probe_follow.py scripts/check_follow_alignment.py
bash -n scripts/build_pose_engine.sh
```

Run: `uv run --extra dev python -m pytest -q`

Expected: PASS — `112 passed, 1 skipped`.
(The skip is the Ultralytics cross-check, which needs the `vision` extra.)

- [ ] **Step 9: Commit**

```bash
git add scripts/follow_log_report.py tests/test_follow_log_report.py scripts/probe_follow.py scripts/check_follow_alignment.py scripts/build_pose_engine.sh
git commit -m "feat(follow): robot gate probe, engine build, and log report" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: CONDITIONAL — fisheye fallback (only if gate G0 finds `camera.rect` misaligned)

**Files:**
- Modify: `scripts/follow_perception.py` (append)
- Modify: `scripts/robot_follow.py` (5 edits)
- Test: `tests/test_follow_perception.py`, `tests/test_robot_follow_cli.py` (append)

**Interfaces:**
- Consumes: `Config("depth").camera_cal()` (13-tuple; `mtx_l, dist_l` at 0–1, `R1` at 4) and `Config("depth").camera_to_base_3x4` on the robot.
- Produces: `project_to_fisheye(points_base, camera_to_base, rect_rotation, K, D) -> (N, 2)`; runner flag `--image-source {head,rect}` defaulting to `head`; `make_pixel_mapper(depth_shape, image_source="rect")`.

Skip this task unless Task 11 G0 shows the depth edges do not sit on the `camera.rect` image outlines, or `camera.rect` is absent. Detection then runs on the raw left fisheye eye of `camera.head`, and each 3D point is projected into that eye.

- [ ] **Step 1: Append the failing tests**

Append to the end of `tests/test_follow_perception.py` (two blank lines before it):

```python
def test_fisheye_projection_of_base_points():
    pytest.importorskip("cv2")
    from follow_perception import project_to_fisheye

    # Camera 1.5 m up looking straight ahead: camera x = base x, y = -base z, z = base y.
    camera_to_base = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 1.5]])
    K = np.array([[447.0, 0, 618.0], [0, 447.0, 498.0], [0, 0, 1.0]])
    points = np.array([[0.0, 2.0, 1.5], [0.5, 2.0, 1.5], [0.0, -1.0, 1.5]])
    pixels = project_to_fisheye(points, camera_to_base, np.eye(3), K, np.zeros(4))
    assert pixels[0] == pytest.approx([618.0, 498.0])
    assert pixels[1] == pytest.approx([618.0 + 447.0 * np.arctan(0.25), 498.0], abs=0.01)
    assert pixels[2].tolist() == [-1.0, -1.0]  # behind the camera
```

Append to the end of `tests/test_robot_follow_cli.py` (two blank lines before it):

```python
def test_fisheye_fallback_is_the_default_image_source():
    assert robot_follow.parse_args([]).image_source == "head"
    assert robot_follow.parse_args(["--image-source", "rect"]).image_source == "rect"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_robot_follow_cli.py -q`

Expected: FAIL; the output contains `has no attribute 'image_source'`.

- [ ] **Step 3: Append `project_to_fisheye` to `scripts/follow_perception.py`**

Append to the end of `scripts/follow_perception.py` (two blank lines before it):

```python
def project_to_fisheye(points_base, camera_to_base, rect_rotation, K, D):
    """Base-frame points -> (N, 2) pixels in the raw left fisheye eye; (-1, -1) if behind the camera.

    Fallback when camera.rect does not line up with camera.depth (gate G0).
    ``camera_to_base`` maps rectified-left-camera coordinates into the base frame
    (Config("depth").camera_to_base_3x4); ``rect_rotation`` is the stereo
    rectification R1 (p_rect = R1 @ p_raw); K, D are the left eye's fisheye intrinsics.
    """
    import cv2

    T = np.asarray(camera_to_base, dtype=np.float64)
    rect_cam = (np.asarray(points_base, dtype=np.float64) - T[:3, 3]) @ T[:3, :3]  # R^T (p - t), row-wise
    raw_cam = rect_cam @ np.asarray(rect_rotation, dtype=np.float64)  # R1^T p_rect, row-wise
    pixels = np.full((len(raw_cam), 2), -1.0)
    ahead = raw_cam[:, 2] > 0.05
    if ahead.any():
        projected, _ = cv2.fisheye.projectPoints(
            raw_cam[ahead].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
            np.asarray(K, dtype=np.float64), np.asarray(D, dtype=np.float64).reshape(4, 1),
        )
        pixels[ahead] = projected.reshape(-1, 2)
    return pixels
```

- [ ] **Step 4: Apply the runner edits**

**Edit 1 — import.** Find:

```python
    PoseEngine, base_to_local, hand_raised, mask_to_image_pixels, person_position,
    torso_histogram, torso_rect,
)
```

Replace with:

```python
    PoseEngine, base_to_local, hand_raised, mask_to_image_pixels, person_position,
    project_to_fisheye, torso_histogram, torso_rect,
)
```

**Edit 2 — `--image-source` flag.** Find:

```python
    parser.add_argument("--check", action="store_true", help="gate G1: time the engine on a live frame, then exit")
```

Replace with:

```python
    parser.add_argument("--check", action="store_true", help="gate G1: time the engine on a live frame, then exit")
    parser.add_argument("--image-source", choices=("head", "rect"), default="head",
                        help="head: raw left fisheye eye with projected points (G0 found camera.rect misaligned)")
```

**Edit 3 — `make_pixel_mapper` for both sources.** Find:

```python
def make_pixel_mapper(depth_shape):
    """Where each camera.points point lands in the detection image: (points_base, mask, image_shape) -> (N, 2)."""
    return lambda points_base, mask, image_shape: mask_to_image_pixels(mask, depth_shape, image_shape)
```

Replace with:

```python
def make_pixel_mapper(depth_shape, image_source="rect"):
    """Where each camera.points point lands in the detection image: (points_base, mask, image_shape) -> (N, 2)."""
    if image_source == "rect":
        return lambda points_base, mask, image_shape: mask_to_image_pixels(mask, depth_shape, image_shape)
    from bbos import Config

    depth_cfg = Config("depth")
    mtx_l, dist_l, _, _, rect_rotation, *_ = depth_cfg.camera_cal()
    camera_to_base = np.asarray(depth_cfg.camera_to_base_3x4, dtype=np.float64)
    return lambda points_base, mask, image_shape: project_to_fisheye(
        points_base, camera_to_base, rect_rotation, mtx_l, dist_l
    )
```

**Edit 4 — image topic follows the flag.** Find:

```python
        image_topic = "camera.rect"
```

Replace with:

```python
        image_topic = "camera.rect" if args.image_source == "rect" else "camera.head"
```

**Edit 5 — pass the source to the mapper.** Find:

```python
make_pixel_mapper(depth_shape), wheel_diam, robot_width)
```

Replace with:

```python
make_pixel_mapper(depth_shape, args.image_source), wheel_diam, robot_width)
```

- [ ] **Step 5: Run the tests (the projection test needs OpenCV, so use the vision extra)**

Run: `uv run --extra dev --extra vision python -m pytest tests/test_follow_perception.py tests/test_robot_follow_cli.py -q`

Expected: PASS — `23 passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/follow_perception.py scripts/robot_follow.py tests/test_follow_perception.py tests/test_robot_follow_cli.py
git commit -m "feat(follow): fisheye fallback when camera.rect is misaligned" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Robot gates G0–G6 (hardware; person at the e-stop)

**Files:**
- Modify: `docs/robot-facts.md` (gate results)
- Modify, only if a gate says so: `scripts/follow_perception.py` (`BASE_LEFT_SIGN`), `scripts/follow_core.py` (`FollowConfig.self_mask`, finally `v_max`), `scripts/robot_follow.py` (`WHEEL_ORDER`, `WHEEL_SIGNS`), `scripts/robot_dashboard.py` (`--follow-v-max` default), `tests/test_robot_follow_cli.py`

**Interfaces:**
- Consumes: everything above; the robot reachable as `ssh bot` (or another alias from the dashboard's list), BBOS at `~/bbos`, `uv` at `~/.local/bin/uv`.
- Produces: recorded evidence per gate, and the final `v_max` default of 0.30 m/s.

Gate order: **G0 → G1 → G2 → G3 → G4a → G5 (0.15 m/s) → G6 → G4b → G5 again (0.30 m/s) → raise the default.** Any failed gate stops the sequence: record what happened, fix, and repeat that gate. The dashboard deploys the runner files itself; for G1/G2 copy them by hand as shown. Keep every CSV (`scp 'bot:/tmp/baymax_follow_*.csv' artifacts/follow/`); `artifacts/` is gitignored.

- [ ] **Step 1: G0: read-only probe**

Start the BBOS depth daemon the way this robot normally starts it (it was **off** at the last probe; `camera.depth`, `camera.points`, `camera.rect` were not publishing) and write the exact method into `docs/robot-facts.md`. Then, with nobody in front of the robot:

```bash
mkdir -p artifacts/follow
scp scripts/probe_follow.py bot:/tmp/
ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/probe_follow.py --out /tmp/follow_probe'
```

Now one person stands about 1 m ahead and 0.5 m to the robot's **left**:

```bash
ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/probe_follow.py --out /tmp/follow_probe_left --person-left'
scp -r bot:/tmp/follow_probe bot:/tmp/follow_probe_left artifacts/
uv run --extra vision python scripts/check_follow_alignment.py artifacts/follow_probe
```

Pass: `camera.rect`, `camera.depth`, `camera.points`, `imu.orientation`, `drive.state`, and `drive.status` each report a nonzero `rate_hz`. Open `artifacts/follow_probe/alignment.png`: the magenta depth edges must sit on object outlines. If they are visibly offset, or `camera.rect` has no `rgb` field, do **Task 10** before G1.

- [ ] **Step 2: G0: apply what the probe found**

1. `cloud.verdict` in `artifacts/follow_probe_left/report.json` says whether base +x points left or right. If it says `+x is LEFT`, change `BASE_LEFT_SIGN = -1.0` to `BASE_LEFT_SIGN = 1.0` in `scripts/follow_perception.py` and update `test_base_frame_points_become_forward_left_up` to expect `[1.0, 0.2, 0.5]`.
2. If `cloud.self_points` in `artifacts/follow_probe/report.json` is above 0, convert `self_box_base_xyz_min/max` (base x, y, z) to one robot-local box with a 3 cm margin and set it as the `FollowConfig` default: `forward` = [y_min − 0.03, y_max + 0.03]; `left` = [−x_max − 0.03, −x_min + 0.03] when `BASE_LEFT_SIGN` is −1, or [x_min − 0.03, x_max + 0.03] when it is +1; `up` = [z_min − 0.03, z_max + 0.03]. For example `self_mask: tuple[...] = ((0.0, 0.33, -0.27, 0.27, 0.9, 1.5),)`.
3. Append a `## Person follow (gate results)` section to `docs/robot-facts.md` with the date and robot ID, and record: the depth-daemon start method; each topic's rate and the `camera.rect`/`camera.depth` shapes and dtypes; the timestamp unit (`timestamp_span_over_window` ≈ 3 means seconds, ≈ 3000 ms, ≈ 3e9 ns) and the `rect_minus_points_timestamp` range; the alignment verdict; base +x direction and `BASE_LEFT_SIGN`; the self box and resulting `self_mask`; `Config("drive")` wheel diameter, width, speed clamps, and command timeout; `Config("base").low_battery_v`.
4. Run the suite and commit:

```bash
uv run --extra dev python -m pytest -q
git add docs/robot-facts.md scripts/follow_perception.py scripts/follow_core.py tests/test_follow_perception.py
git commit -m "chore(follow): gate G0 results" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: G1: build the engine and time it on the Jetson**

```bash
uv run --extra vision yolo export model=yolo11n-pose.pt format=onnx opset=17 imgsz=640
scp yolo11n-pose.onnx scripts/build_pose_engine.sh bot:/tmp/
ssh bot 'bash /tmp/build_pose_engine.sh /tmp/yolo11n-pose.onnx'
scp scripts/robot_follow.py scripts/follow_core.py scripts/follow_perception.py bot:/tmp/
```

With a person standing about 1.5 m in front of the robot:

```bash
ssh bot '~/.local/bin/uv run --script /tmp/robot_follow.py --check'
```

If Task 10 was applied, the runner already defaults to the fisheye eye. The first run builds the script's environment from the uv cache the greeter already populated; if it cannot resolve without network, connect the robot to the internet once and repeat. Never install into `~/bbos` instead.

Pass: `latency_ms_p50` ≤ 50, `latency_ms_max` ≤ 80, `people` ≥ 1 with a plausible `best.box`. Record the JSON and `~/.cache/baymax/yolo11n-pose.engine.txt` in `docs/robot-facts.md`. If TensorRT or PyCUDA fails to import or the engine fails to deserialize, stop: record the error and ask before changing anything on the robot.

**An agent stops here.** G2–G6 need people in front of the robot and an e-stop operator.

- [ ] **Step 4: G2: dry run (never opens `drive.ctrl`)**

Tape floor marks straight ahead of the wheel axle at 0.6, 1.0, and 1.5 m, and at 1.0 m ±30°. Then:

```bash
ssh -t bot '~/.local/bin/uv run --script /tmp/robot_follow.py --dry-run --no-heartbeat'
```

1. Person A stands on the 1.0 m mark and raises a hand for about a second. The state line changes to `FOLLOWING` and the LED turns solid green. Nobody else raising a hand before that must lock anyone.
2. Person A stands on each mark (centre of the feet on the tape) for 5 s; read `range` from the `FOLLOW_STATUS` lines.
3. Person B walks slowly between the robot and person A. `range` must come back to A's distance, not settle on B's.
4. Person A leaves the camera's view: the state goes `LOST` (blinking amber) and returns to `FOLLOWING` when A comes back within 10 s.
5. `Ctrl-C`, then copy the CSV and run `python scripts/follow_log_report.py artifacts/follow/<file>.csv`.

Pass: every mark within ±0.05 m; bystander rejected; lock only on a raised hand. If every range is off by a similar amount, **do not add an offset in code**: the point cloud is miscalibrated (a reflash has shifted another robot's cloud by 0.78 m). Recalibrate the depth pipeline the BBOS way, repeat G0 and G2, and record it.

- [ ] **Step 5: G3: rotate only**

```bash
python scripts/robot_dashboard.py --follow-rotate-only
```

Open <http://127.0.0.1:8020>, press **Follow me**, and have the person raise a hand at about 1.2 m, then walk a slow arc ±60° around the robot at 1.0–1.5 m for 60 s. Press **Stop action**, copy the CSV, and run the report.

Pass: `bearing_within_10deg_fraction` ≥ 0.95, `omega_sign_agreement` ≥ 0.9, and no `odometry-mismatch` exit. If `omega_sign_agreement` is low or the runner exits with `odometry-mismatch`, set `WHEEL_ORDER = (1, 0)` in `scripts/robot_follow.py`, record it, and repeat G3.

- [ ] **Step 6: G4a: follow at 0.15 m/s**

`python scripts/robot_dashboard.py` (default cap 0.15 m/s) on an open floor with at least 4 m clear ahead. Run two sessions (each start writes its own CSV):

1. **Stroll:** lock at 1.0 m; the person stands 20 s, then shuffles away at about 0.1 m/s (one short step every 2 s) for 30 s, then stops on a tape mark. When the robot has settled, tape-measure from the wheel axle to the centre of the person's feet. Move the slider to 0.7 m and confirm it settles there too. Stop.
2. **Step-backs:** lock at 1.0 m; the person takes one quick 0.5 m step back, waits 5 s, five times. Stop.

Pass: session 1 `in_band_fraction` ≥ 0.95; the tape agrees with the logged `range` within 0.05 m; `v_sign_agreement` ≥ 0.9; `last_rule` is `stop`. In session 2 the robot is back in the band within 5 s of every step. If `v_sign_agreement` is low or the runner exits with `odometry-mismatch` (G3 already proved the turn direction), flip both wheel signs and swap the wheel order, which negates `v` and leaves `omega` as it was: set `WHEEL_SIGNS = (-1.0, -1.0)` and change `WHEEL_ORDER` from `(0, 1)` to `(1, 0)` or back. Record it, then repeat G3 and G4a.

- [ ] **Step 7: G5 at 0.15 m/s: obstacle**

While following a person strolling at about 0.1 m/s, a helper places a 30 cm box on the floor in the robot's path about 0.5 m ahead of it. Then remove it.

Pass: the robot stops before contact (state `BLOCKED`, LED solid amber); the report's `blocked_to_stop_s` ≤ 0.5; it resumes following about 0.5 s after the box is removed. Film it: the box must be seen and the robot stopped within 0.3 s of the box entering the camera's view.

- [ ] **Step 8: G6: link loss**

While following at 0.15 m/s: (a) press `Ctrl-C` in the dashboard terminal; (b) restart, follow again, then turn the laptop's Wi-Fi off. Pass: both times the robot is stationary within 1.2 s, and the CSV's last rule is `stop` (a) or `heartbeat` (b).

- [ ] **Step 9: G4b and G5 at 0.30 m/s**

`python scripts/robot_dashboard.py --follow-v-max 0.30`. Repeat the G4a stroll session with the person walking slowly at up to 0.25 m/s (a normal step every 2–3 s), then repeat G5.

Pass: `in_band_fraction` ≥ 0.95 with the person at ≤ 0.25 m/s; no `not-upright` exit and no visible pitching or oscillation of the balancing base; G5 criteria as before.

- [ ] **Step 10: Record the gates and raise the default speed cap**

Record G2–G6 results (report JSON, tape readings, video file names) in `docs/robot-facts.md`, and add this standing rule to that section: **repeat G0 and G2 after any BBOS reflash or depth-daemon change** (spec §9). Then make 0.30 m/s the default:

In `scripts/follow_core.py`, replace:

```python
    v_max: float = 0.15  # default rises to 0.30 only after robot gate G4b passes
```

with:

```python
    v_max: float = 0.30  # raised after robot gate G4b passed (docs/robot-facts.md)
```

In `scripts/robot_dashboard.py`, replace:

```python
        default=0.15,
        help="follow speed cap in m/s (0.15 until robot gate G4b passes; at most 0.30)",
```

with:

```python
        default=0.30,
        help="follow speed cap in m/s (at most 0.30; use 0.15 when bringing up a new robot)",
```

In `tests/test_robot_follow_cli.py`, replace:

```python
def test_defaults_are_the_bring_up_limits():
    args = robot_follow.parse_args([])
    assert (args.gap, args.v_max, args.dry_run, args.rotate_only) == (1.0, 0.15, False, False)
    assert robot_follow.loop_config(args).v_max == 0.15
```

with:

```python
def test_default_limits():
    args = robot_follow.parse_args([])
    assert (args.gap, args.v_max, args.dry_run, args.rotate_only) == (1.0, 0.30, False, False)
    assert robot_follow.loop_config(args).v_max == 0.30
```

Run `uv run --extra dev python -m pytest -q` and expect PASS (`112 passed, 1 skipped` during planning). Update the README sentence "Until gate G4b passes, the speed cap is 0.15 m/s" to "The speed cap is 0.30 m/s (0.15 m/s with `--follow-v-max 0.15` when bringing up a new robot)". Commit with the message `feat(follow): 0.30 m/s default after robot gates passed` and the trailer.

- [ ] **Step 11: Finish the branch**

Use superpowers:finishing-a-development-branch to decide how `feature/person-follow` is merged.
