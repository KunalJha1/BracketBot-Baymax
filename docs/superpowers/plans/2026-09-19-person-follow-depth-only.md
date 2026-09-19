# Person-Follow Depth-Only Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch person-follow from the YOLO-pose detector to depth-only clustering (no neural network, no camera image), then finish the dashboard, gate tooling, and robot gates.

**Architecture:** People are person-sized clusters on a 10 cm floor grid built from `camera.points` (`follow_perception.find_people`). Lock-on becomes "the only person-sized cluster in the start zone for 0.5 s" (`follow_core.LockOn`). The runner reads only depth, IMU and drive topics and runs in the BBOS venv like the Lean runner. Everything else — tracker, controller, supervisor, corridor, state machine, dashboard protocol — is unchanged from `docs/superpowers/plans/2026-09-19-person-follow.md` Tasks 1–7, which are done.

**Tech Stack:** Python 3.10+, numpy, pytest; BBOS `Reader`/`Writer` on the robot.

**Spec:** `docs/superpowers/specs/2026-09-19-person-follow-design.md` (§12 describes this change; §6.1 is the new perception).

**Replaces:** Tasks 8–11 of `docs/superpowers/plans/2026-09-19-person-follow.md`. Its Tasks 1–7 stay done; its Task 8 (dashboard) and Task 9 (tooling) are rewritten here as Tasks 2 and 3 without the TensorRT parts; its Task 10 (fisheye fallback) and gate G1 are dropped.

**Provenance:** every code block below was run before this plan was written. A generator replayed each step on a fresh copy of the branch at `083ea54` in wave order, including every "expect FAIL" and "expect PASS" run; the `Expected:` lines quote that replay. That proves the laptop-side code and tests; it proves nothing about the robot — that is Task 5.

## Execution Waves (parallel)

| Wave | Tasks | Why they can run together |
|---|---|---|
| 1 | Task 1 (core), Task 2 (dashboard), Task 3 (gate tooling) | Disjoint files; none imports another's new code. |
| 2 | Task 4 (perception + runner) | Needs Task 1's `PersonObservation(forward, left, score=1.0, hist=None)`. |
| — | Task 5 (robot gates) | Hardware; after both waves, with people and an e-stop operator. |

Rules while tasks share one worktree in parallel:
- Run only your own task's test files. Between waves the controller runs the whole suite.
- After wave 1 and before wave 2, exactly one test is expected to fail: `tests/test_robot_follow_cli.py::test_perceive_turns_detections_into_located_people` (the old pose runner builds `PersonObservation` with the removed `hand_raised` argument). Task 4 replaces that file.
- Commit only your own files, with the exact commands given: `git add <files>` then `git commit ... -- <files>`. The trailing path list commits only those files even if another task has staged others. If git reports `index.lock` exists, wait a few seconds and retry; never delete the lock file.

## Global Constraints

- Python: laptop code must run on 3.10+ (`requires-python = ">=3.10"`); the robot runs Python 3.10 in `~/bbos/.venv`. No 3.11+ syntax or stdlib.
- `scripts/follow_core.py` and `scripts/follow_perception.py` import only the standard library and numpy. No TensorRT, PyCUDA, OpenCV, or neural networks anywhere in follow mode.
- `scripts/robot_dashboard.py` stays standard-library only. It must not import `follow_core` or numpy; it duplicates `FOLLOW_GAP_*` and `FOLLOW_STATUS_PREFIX`, and a test keeps them equal.
- Robot files are deployed flat to `/tmp` and import siblings by bare name (`import follow_core`). Tests reach them through `tests/conftest.py`.
- Internal units are metres, seconds, and radians, in robot-local `(forward, left, up)`. Degrees appear only in status and log output.
- The robot never reverses: `v >= 0` is enforced in the controller, the supervisor, and the runner.
- `FollowConfig` and `ClusterConfig` defaults are the spec §6.1 and §6.7 values; change one only together with the spec.
- The speed cap is 0.15 m/s until gate G4b passes, and never above 0.30 m/s (the drive daemon's clamp).
- On the robot, commands use the BBOS venv (`~/.local/bin/uv run --no-sync --project ~/bbos python …` or `~/bbos/.venv/bin/python …`). Never `uv sync` or `pip install` into `~/bbos`.
- No robot gate that can move the base runs without a person at the physical e-stop.
- Simulation results are reported as simulation evidence, never as hardware evidence.
- Run laptop commands from the repository root in Git Bash. Tests: `uv run --extra dev python -m pytest …` (uv may create `uv.lock`; the repository does not track it, so leave it uncommitted).
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` as its own paragraph (the commit commands below do this with a second `-m`).

## File Structure

| File | Task | Change |
|---|---|---|
| `scripts/follow_core.py` | 1 | `PersonObservation` loses `hand_raised`; start-zone lock-on; zone limits 2.0 m / 30° |
| `tests/test_follow_core.py` | 1 | lock-on tests rewritten; `person()` helper updated |
| `tests/test_follow_sim.py` | 1 | observations carry no appearance cue; no hand raise |
| `scripts/robot_dashboard.py` | 2 | Follow button, gap slider, heartbeat, exclusivity, simulation, gate flags (23 edits) |
| `tests/test_robot_dashboard.py` | 2 | follow tests appended |
| `README.md` | 2 | follow-mode section and control-table row |
| `scripts/follow_log_report.py` + `tests/test_follow_log_report.py` | 3 | gate metrics from a runner CSV |
| `scripts/probe_follow.py` | 3 | read-only gate G0 probe |
| `scripts/follow_perception.py` + `tests/test_follow_perception.py` | 4 | replaced: depth clustering |
| `scripts/robot_follow.py` + `tests/test_robot_follow_cli.py` | 4 | replaced: depth-only runner in the BBOS venv |
| `docs/robot-facts.md` | 5 | gate results |

---

### Task 1: Core: start-zone lock-on, no hand raise (wave 1)

**Files:**
- Modify: `scripts/follow_core.py` (4 edits)
- Modify: `tests/test_follow_core.py` (3 edits)
- Modify: `tests/test_follow_sim.py` (3 edits)

**Interfaces:**
- Consumes: the existing `follow_core` (plan 1, Tasks 1–4).
- Produces: `PersonObservation(forward, left, score=1.0, hist=None)` (no `hand_raised`);
  `LockOn(cfg).update(t, people, positions)` locks the only candidate seen for `lock_window` s in ≥ `lock_fraction` of the frames in that window,
  inside `lock_range_min..lock_range_max` = 0.5–2.0 m and `|bearing| <= lock_bearing_max` = 30°.

Runs in parallel with Tasks 2 and 3. Run only `tests/test_follow_core.py` and `tests/test_follow_sim.py`.

- [ ] **Step 1: Update the tests first**

**Edit 1 — `person()` helper.** Find:

```python
def person(forward, left=0.0, raised=False, hist=None):
    return PersonObservation(forward, left, 0.9, raised, hist)
```

Replace with:

```python
def person(forward, left=0.0, hist=None):
    return PersonObservation(forward, left, hist=hist)
```

**Edit 2 — lock-on tests.** Find:

```python
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
```

Replace with:

```python
def test_person_in_the_start_zone_for_half_a_second_is_locked():
    frames = [[person(1.2, 0.1), person(1.5, -0.95)] for _ in range(15)]  # the second is 32 degrees off-axis
    locked = run_lock_on(frames)
    assert locked is not None
    t, (obs, xy) = locked
    assert t >= CFG.lock_window
    assert xy == pytest.approx((1.2, 0.1))


def test_brief_presence_does_not_lock():
    frames = [[person(1.2)] if i < 4 else [] for i in range(20)]
    assert run_lock_on(frames) is None


def test_flickering_candidate_does_not_lock():
    frames = [[person(1.2)] if i % 3 == 0 else [] for i in range(30)]
    assert run_lock_on(frames) is None


def test_two_people_in_the_zone_lock_neither():
    frames = [[person(1.2, 0.3), person(1.2, -0.3)] for _ in range(20)]
    assert run_lock_on(frames) is None


def test_people_outside_the_start_zone_are_ignored():
    far = [[person(2.3)] for _ in range(20)]
    wide = [[person(1.0, 0.8)] for _ in range(20)]  # 39 degrees off-axis
    near = [[person(0.4)] for _ in range(20)]
    assert run_lock_on(far) is None
    assert run_lock_on(wide) is None
    assert run_lock_on(near) is None
```

**Edit 3 — loop test person.** Find:

```python
            frame = Perception(t, (person(1.6, raised=t < 1.0),), np.empty((0, 3)))
```

Replace with:

```python
            frame = Perception(t, (person(1.6),), np.empty((0, 3)))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: FAIL; the output contains `missing 2 required positional arguments`.

- [ ] **Step 3: Change `scripts/follow_core.py`**

**Edit 1 — `PersonObservation` fields.** Find:

```python
    score: float
    hand_raised: bool
    hist: np.ndarray | None = None  # (64,) L1-normalised 4x4x4 HSV torso histogram
```

Replace with:

```python
    score: float = 1.0
    hist: np.ndarray | None = None  # optional appearance cue; depth-only perception leaves it None
```

**Edit 2 — `lock_fraction` comment.** Find:

```python
    lock_fraction: float = 0.8
```

Replace with:

```python
    lock_fraction: float = 0.8  # share of the window's frames the candidate must appear in
```

**Edit 3 — start-zone limits.** Find:

```python
    lock_range_max: float = 2.5
    lock_bearing_max: float = math.radians(60.0)
```

Replace with:

```python
    lock_range_max: float = 2.0
    lock_bearing_max: float = math.radians(30.0)
```

**Edit 4 — `LockOn` docstring, reset, frame log.** Find:

```python
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
```

Replace with:

```python
class LockOn:
    """Chooses the single person who stays in the start zone for ``lock_window`` seconds."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.candidates = []
        self.frames = []  # times of the frames seen while searching

    def update(self, t, people, positions):
        """Returns ``(observation, odom_xy)`` of the locked person, or None."""
        cfg = self.cfg
        self.frames.append(t)
        used = set()
```

**Edit 5 — `LockOn` presence rule.** Find:

```python
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
```

Replace with:

```python
            if best is None:
                self.candidates.append({"xy": xy, "first_t": t, "seen": [], "obs": obs})
                best = len(self.candidates) - 1
            cand = self.candidates[best]
            cand["xy"], cand["obs"] = xy, obs
            cand["seen"].append(t)
            used.add(best)

        horizon = t - cfg.lock_window
        self.frames = [f for f in self.frames if f >= horizon]
        self.candidates = [c for c in self.candidates if c["seen"][-1] >= horizon]
        qualified = []
        for cand in self.candidates:
            cand["seen"] = [s for s in cand["seen"] if s >= horizon]
            if t - cand["first_t"] >= cfg.lock_window and len(cand["seen"]) >= cfg.lock_fraction * len(self.frames):
                qualified.append(cand)
        if len(qualified) != 1:
            return None  # nobody yet, or two people in the zone: keep waiting
```

- [ ] **Step 4: Run the core tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_follow_core.py -q`

Expected: PASS — `44 passed`.

- [ ] **Step 5: Update the simulation to carry no appearance cue**

**Edit 1 — docstring.** Find:

```python
lag; perception is 15 Hz with 100 ms latency and Gaussian range/bearing noise.
Passing here says the logic and tuning are coherent, not that the robot works.
```

Replace with:

```python
lag; perception is 15 Hz with 100 ms latency and Gaussian range/bearing noise,
and observations carry no appearance cue (as with depth-only perception).
Passing here says the logic and tuning are coherent, not that the robot works.
```

**Edit 2 — drop histograms and hand raise.** Find:

```python
LAG = 0.15  # balancing base velocity response time constant (s)
TARGET_HIST = np.eye(64)[3]
BYSTANDER_HIST = np.eye(64)[40]


def raised_for_first_second(t):
    return t < 1.0
```

Replace with:

```python
LAG = 0.15  # balancing base velocity response time constant (s)
```

**Edit 3 — observations.** Find:

```python
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
```

Replace with:

```python
            actors = [(scenario.target, scenario.visible(t))]
            if scenario.bystander is not None:
                actors.append((scenario.bystander, True))
            for path, visible in actors:
                if not visible:
                    continue
                f, l = robot.to_local(*path(t))
                r = math.hypot(f, l) + rng.normal(0, 0.025)
                b = math.atan2(l, f) + rng.normal(0, math.radians(1.0))
                people.append(PersonObservation(r * math.cos(b), r * math.sin(b)))
```

- [ ] **Step 6: Run the simulation**

Run: `uv run --extra dev python -m pytest tests/test_follow_sim.py -q`

Expected: PASS — `9 passed`.

- [ ] **Step 7: Commit only these files**

```bash
git add scripts/follow_core.py tests/test_follow_core.py tests/test_follow_sim.py
git commit -m "feat(follow): lock onto whoever stands in the start zone" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- scripts/follow_core.py tests/test_follow_core.py tests/test_follow_sim.py
```

---

### Task 2: Dashboard: Follow button, distance slider, heartbeat, exclusivity (wave 1)

**Files:**
- Modify: `scripts/robot_dashboard.py` (23 exact edits)
- Modify: `README.md`
- Test: `tests/test_robot_dashboard.py` (2 import edits + appended tests)

**Interfaces:**
- Consumes: the runner CLI `--gap --pid-file --v-max --rotate-only` and stdout/stdin protocol (plan 1 Task 7; Task 4 keeps it).
  Does **not** import `follow_core` (stdlib only; one test compares the duplicated constants).
- Produces: `FOLLOW_RUNNER`, `FOLLOW_MODULES`, `REMOTE_FOLLOW_RUNNER`, `FOLLOW_GAP_MIN/MAX/DEFAULT`, `FOLLOW_STATUS_PREFIX`, `FOLLOW_HEARTBEAT_PERIOD`;
  `RobotController(ssh_hosts, simulate=False, follow_args=())` with `set_follow(enabled)` and `set_follow_gap(gap)`;
  snapshot keys `follow_enabled follow_transition follow_phase follow_status follow_gap follow_gap_min follow_gap_max`;
  `POST /api/follow {enabled}`, `POST /api/follow/gap {gap}`; CLI flags `--follow-v-max` (default 0.15, max 0.30) and `--follow-rotate-only`.
  The runner is launched with the existing `remote_python_command` (BBOS venv).

Runs in parallel with Tasks 1 and 3. Run only `tests/test_robot_dashboard.py`. Follow owns the base, LEDs, and camera: it refuses to start during an action or lean, and actions and lean refuse while it runs. `Esc` works even when the slider has keyboard focus.

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
        yield "[follow] follow active (v_max 0.15 m/s, gap 1.00 m) - stand in front of the robot\n"
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
    controller = RobotController(("bot",), follow_args=("--v-max", "0.15"))
    controller.state.host = "bot"
    monkeypatch.setattr(controller, "_deploy", lambda host, *paths: False)
    monkeypatch.setattr(controller, "_request_remote_stop",
                        lambda host, pid_file, name: remote_stops.append(pid_file))

    assert controller.set_follow(True) == (True, "Starting follow mode")
    wait_for(lambda: controller.state.snapshot()["follow_enabled"])
    process = processes[0]
    assert "/tmp/robot_follow.py --gap 1.00 --pid-file /tmp/bracketbot-follow-1.pid --v-max 0.15" in process.command[-1]
    assert '"$HOME/bbos/.venv/bin/python"' in process.command[-1]
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

**Edit 4 — follow state fields.** Find:

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

**Edit 5 — follow fields in `snapshot()`.** Find:

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

**Edit 6 — `add_follow_log`.** Find:

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

**Edit 7 — follow counters in `RobotController.__init__`.** Find:

```python
        self._lean_counter = 0
```

Replace with:

```python
        self._lean_counter = 0
        self._follow_counter = 0
        self._follow_write_lock = threading.Lock()
```

**Edit 8 — actions refuse while following.** Find:

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

**Edit 9 — lean refuses while following.** Find:

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

**Edit 10 — follow control methods (inserted before `stop()`) and the head of `stop()`.** Find:

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
            remote_command = remote_python_command(
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

**Edit 11 — `stop()` also stops follow.** Find:

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

**Edit 12 — CSS for four controls and the slider.** Find:

```python
.controls { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:18px; }
```

Replace with:

```python
.controls { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:18px; }
.follow-gap { display:flex; align-items:center; gap:14px; margin-top:14px; color:var(--muted); font-size:.9rem; font-weight:700; }
.follow-gap input { flex:1; accent-color:var(--red); }
```

**Edit 13 — follow status line (HTML).** Find:

```python
      <p id="base-detail">Base: balance mode</p>
```

Replace with:

```python
      <p id="base-detail">Base: balance mode</p>
      <p id="follow-detail">Follow: off</p>
```

**Edit 14 — style the follow status line.** Find:

```python
#detail, #base-detail, #latency-detail {
```

Replace with:

```python
#detail, #base-detail, #follow-detail, #latency-detail {
```

**Edit 15 — Follow button and distance slider (HTML).** Find:

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

**Edit 16 — JS element handles and `followText`.** Find:

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
  if(s.state==='SEARCHING') return 'Follow: stand in front of the robot';
  const range=s.range==null?'—':`${s.range.toFixed(2)} m`;
  const err=s.error==null?'':` · ${s.error>=0?'+':''}${Math.round(s.error*100)} cm from target`;
  return `Follow: ${s.state.toLowerCase()} · ${range}${err}`;
}
```

**Edit 17 — JS `refresh()` follow wiring.** Find:

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

**Edit 18 — JS button and slider listeners.** Find:

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

**Edit 19 — JS keys: Esc first (works with the slider focused), then F.** Find:

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

**Edit 20 — HTTP routes.** Find:

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

**Edit 21 — shutdown waits for follow to stop.** Find:

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

**Edit 22 — `follow_args` launch option.** Find:

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

**Edit 23 — gate launch flags in `main()`.** Find:

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

Expected: PASS — `21 passed`.

- [ ] **Step 6: Check the page script parses and try it in simulation**

```bash
uv run --extra dev python -c "import re,sys; sys.path.insert(0,'scripts'); import robot_dashboard as d; open('page.js','w').write(re.search(r'<script>(.*)</script>', d.PAGE, re.S).group(1))" && node --check page.js && rm page.js
python scripts/robot_dashboard.py --simulate   # open http://127.0.0.1:8020
```

In the browser: **Follow me** shows "stand in front of the robot" for about a second, then a range converging on the slider value; moving the slider changes it; `Esc` (also with the slider focused) and **Stop action** end follow; gestures and **Enable lean** are disabled while following.

- [ ] **Step 7: Document follow mode in the README**

In `README.md`, replace:

```markdown
| Base mode | Toggle 4° lean / balance | `Z` |
```

with:

```markdown
| Base mode | Toggle 4° lean / balance | `Z` |
| Follow | Follow the person standing in front; distance slider | `F` |
```

In `README.md`, replace:

```markdown
## Run local person and expression detection
```

with:

```markdown
## Follow mode (person following)

**Follow me** (`F`) starts `scripts/robot_follow.py` on the robot. Stand
0.5–2 m in front of it: the only person-sized shape in that zone for half a
second is locked on. It finds you in the depth camera's 3D points alone (no
neural network), so it cannot tell you from a pillar or coat rack, and loses
you if you stand right against a wall or another person. The robot holds the
**Following distance** slider's gap (0.6–1.5 m, default 1.0 m, measured to the
front of your torso) to within ±20 cm while you stand, turn, or walk slowly.
The base is clamped to 0.3 m/s, so it cannot keep pace with normal walking and
catches up when you pause. It never drives backward, stops for anything in a
0.6 m corridor ahead, and stops by itself if the dashboard, the link, or
balance is lost. After losing you for 10 s it locks onto whoever next stands in
front of it. `Esc` stops it like every other action; `--simulate` exercises the
whole path without a robot.

Do not use follow mode around people until every robot gate in
`docs/superpowers/plans/2026-09-19-person-follow-depth-only.md` (Task 5) has
passed, with a person at the physical e-stop for each gate that moves the
robot. Until gate G4b passes, the speed cap is 0.15 m/s; `--follow-v-max 0.30`
and `--follow-rotate-only` exist for the gates. Design:
`docs/superpowers/specs/2026-09-19-person-follow-design.md`.

## Run local person and expression detection
```

- [ ] **Step 8: Commit only these files**

```bash
git add scripts/robot_dashboard.py tests/test_robot_dashboard.py README.md
git commit -m "feat(dashboard): follow mode with distance slider and heartbeat" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- scripts/robot_dashboard.py tests/test_robot_dashboard.py README.md
```

---

### Task 3: Gate tooling: log report and read-only probe (wave 1)

**Files:**
- Create: `scripts/follow_log_report.py`
- Test: `tests/test_follow_log_report.py`
- Create: `scripts/probe_follow.py`

**Interfaces:**
- Consumes: `robot_follow.CSV_FIELDS` (unchanged by Task 4).
- Produces: `follow_log_report.summarise(rows) -> dict` with `samples following_samples in_band_fraction abs_error_p95_m bearing_within_10deg_fraction v_sign_agreement omega_sign_agreement blocked_to_stop_s rules last_rule`;
  the read-only probe (`--out DIR [--person-left]`, writes `report.json`, `points.npz`, `camera_depth.npy`).

Runs in parallel with Tasks 1 and 2. Run only `tests/test_follow_log_report.py`. Task 5 uses these to turn each gate into numbers.

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

- [ ] **Step 5: Write the read-only probe `scripts/probe_follow.py` and syntax-check it**

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

TOPICS = ("camera.depth", "camera.points", "imu.orientation", "drive.state", "drive.status")
CONFIGS = ("drive", "depth", "base")


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

    points = frames.get("camera.points")
    if points and "points" in points:
        n = int(np.asarray(points["num_points"]).item())
        report["cloud"] = cloud_summary(points["points"][:n], args.person_left)
        np.savez_compressed(args.out / "points.npz", points=points["points"][:n], mask=points["mask"][:n])
    depth = frames.get("camera.depth")
    if depth and "depth" in depth:
        np.save(args.out / "camera_depth.npy", depth["depth"])

    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=repr))
    print(json.dumps(report, indent=2, default=repr))


if __name__ == "__main__":
    main()
```

```bash
python -m py_compile scripts/probe_follow.py
```

- [ ] **Step 6: Commit only these files**

```bash
git add scripts/follow_log_report.py tests/test_follow_log_report.py scripts/probe_follow.py
git commit -m "feat(follow): gate log report and read-only probe" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- scripts/follow_log_report.py tests/test_follow_log_report.py scripts/probe_follow.py
```

---

### Between the waves (controller)

Run `uv run --extra dev python -m pytest -q`. Expected: exactly one failure, `tests/test_robot_follow_cli.py::test_perceive_turns_detections_into_located_people` (it builds `PersonObservation` with the removed `hand_raised`; Task 4 replaces the file). Anything else failing means a wave-1 task went wrong.

---

### Task 4: Depth-only perception and runner (wave 2)

**Files:**
- Replace: `scripts/follow_perception.py`
- Replace: `tests/test_follow_perception.py`
- Replace: `scripts/robot_follow.py`
- Replace: `tests/test_robot_follow_cli.py`

**Interfaces:**
- Consumes: `PersonObservation(forward, left, score=1.0, hist=None)` (Task 1); everything else in `follow_core` unchanged.
- Produces: `follow_perception.BASE_LEFT_SIGN`, `ClusterConfig` (spec §6.1 values), `Cluster(forward, left, points, top, depth, width)`,
  `base_to_local(points_base, left_sign)`, `find_people(points_local, cfg=ClusterConfig()) -> list[Cluster]` (nearest first);
  runner flags `--gap --v-max --pid-file --log-dir --dry-run --rotate-only --no-heartbeat --check` (no `--engine`);
  `robot_follow.perceive(points_base, t, cluster_cfg=ClusterConfig()) -> Perception`; `CSV_FIELDS`, `WHEEL_ORDER`, `WHEEL_SIGNS` unchanged.

Wave 2: start after Task 1 is committed. The pose engine, keypoints, torso colour histogram, `camera.rect`, and the PEP 723 header all go away; the runner runs in the BBOS venv. Clustering took about 9 ms per 40k-point frame on a laptop.

- [ ] **Step 1: Replace the perception tests**

Replace the entire contents of `tests/test_follow_perception.py` with:

```python
import math

import numpy as np
import pytest

from follow_perception import ClusterConfig, base_to_local, find_people

RNG = np.random.default_rng(7)


def standing_person(forward, left=0.0, height=1.75, radius=0.18, n=900):
    """Points on the half of a vertical cylinder that faces the robot (what stereo sees)."""
    centre = np.array([forward, left])
    toward_robot = -centre / np.linalg.norm(centre)
    side = np.array([-toward_robot[1], toward_robot[0]])
    phi = RNG.uniform(-math.pi / 2, math.pi / 2, n)
    xy = centre + radius * (np.cos(phi)[:, None] * toward_robot + np.sin(phi)[:, None] * side)
    z = RNG.uniform(0.05, height, n)
    return np.column_stack([xy, z])


def floor(n=2000):
    return np.column_stack([RNG.uniform(0.3, 3.5, n), RNG.uniform(-2.0, 2.0, n), RNG.normal(0.0, 0.01, n)])


def slab(f0, f1, l0, l1, z0, z1, n):
    return np.column_stack([RNG.uniform(f0, f1, n), RNG.uniform(l0, l1, n), RNG.uniform(z0, z1, n)])


def test_a_standing_person_is_found_at_the_front_of_their_torso():
    people = find_people(np.vstack([standing_person(1.2, 0.3), floor()]))
    assert len(people) == 1
    person = people[0]
    # the camera sees the front surface: about 0.7 of the radius nearer than the centre
    assert person.forward == pytest.approx(1.2 - 0.7 * 0.18, abs=0.05)
    assert person.left == pytest.approx(0.3 - 0.3 / 1.24 * 0.7 * 0.18, abs=0.05)
    assert person.top >= 1.6


def test_floor_table_wall_and_box_are_not_people():
    table = np.vstack([slab(1.0, 1.8, -0.4, 0.4, 0.72, 0.76, 1500),  # top
                       slab(1.0, 1.05, -0.4, -0.35, 0.1, 0.72, 100), slab(1.75, 1.8, 0.35, 0.4, 0.1, 0.72, 100)])
    wall = slab(2.95, 3.0, -2.0, 2.0, 0.1, 2.0, 4000)
    box = slab(0.8, 1.1, -0.9, -0.6, 0.1, 0.4, 600)
    assert find_people(np.vstack([floor(), table, wall, box])) == []


def test_two_separated_people_give_two_clusters_nearest_first():
    people = find_people(np.vstack([standing_person(2.0, 0.6), standing_person(1.3, -0.4)]))
    assert len(people) == 2
    assert people[0].forward < people[1].forward
    assert people[0].left < 0 < people[1].left


def test_a_person_a_step_away_from_a_wall_is_still_found():
    wall = slab(1.75, 1.8, -2.0, 2.0, 0.1, 2.0, 4000)
    people = find_people(np.vstack([standing_person(1.3), wall]))
    assert len(people) == 1
    assert people[0].forward == pytest.approx(1.3 - 0.7 * 0.18, abs=0.05)


def test_a_person_against_a_wall_merges_with_it_and_is_lost():
    # documented limit: within about 10 cm the person and wall form one too-wide cluster
    wall = slab(1.33, 1.38, -2.0, 2.0, 0.1, 2.0, 4000)
    assert find_people(np.vstack([standing_person(1.3), wall])) == []


def test_crop_and_empty_input():
    assert find_people(standing_person(4.0)) == []
    assert find_people(np.empty((0, 3))) == []
    assert find_people(standing_person(1.2, n=40)) == []  # too few points
    assert find_people(standing_person(1.2), ClusterConfig(min_top=1.9)) == []


def test_base_frame_points_become_forward_left_up():
    local = base_to_local(np.array([[0.2, 1.0, 0.5]]))
    assert local[0] == pytest.approx([1.0, -0.2, 0.5])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_follow_perception.py -q`

Expected: FAIL; the output contains `cannot import name 'ClusterConfig'`.

- [ ] **Step 3: Replace `scripts/follow_perception.py`**

Replace the entire contents of `scripts/follow_perception.py` with:

```python
"""Depth-only person finding for person-follow: no neural network, no camera image.

Converts ``camera.points`` to robot-local (forward, left, up) and finds
person-sized clusters on a floor grid. Pure numpy, so it runs unchanged in the
laptop tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# camera.points uses base +y forward, +x lateral. -1 means +x points to the robot's
# right (right-handed, z up). Verified at gate G0; flip here if G0 shows otherwise.
BASE_LEFT_SIGN = -1.0


@dataclass(frozen=True)
class ClusterConfig:
    cell: float = 0.10  # floor-grid cell size (m)
    min_cell_points: int = 3  # a cell with fewer points is empty
    z_min: float = 0.10  # drops the floor
    z_max: float = 2.0  # drops the ceiling
    forward_min: float = 0.3
    forward_max: float = 3.5
    left_max: float = 2.5
    max_footprint: float = 0.8  # wider or deeper than this is not a person (m)
    min_footprint: float = 0.15  # the larger extent must reach this (m)
    min_top: float = 1.2  # highest point must reach this (m)
    min_points: int = 60
    torso_z: tuple[float, float] = (0.8, 1.6)
    min_torso_points: int = 10


@dataclass(frozen=True)
class Cluster:
    forward: float  # torso median (m); the surface the camera sees
    left: float
    points: int
    top: float  # highest point (m)
    depth: float  # footprint extent along forward (m)
    width: float  # footprint extent along left (m)


def base_to_local(points_base, left_sign=BASE_LEFT_SIGN):
    """camera.points base frame (x lateral, y forward, z up) -> (forward, left, up)."""
    p = np.asarray(points_base, dtype=np.float64)
    return np.column_stack([p[:, 1], left_sign * p[:, 0], p[:, 2]])


def _label(occupied):
    """8-connected component labels of a boolean grid; 0 is background."""
    labels = np.zeros(occupied.shape, dtype=int)
    rows, cols = occupied.shape
    current = 0
    for start in zip(*np.nonzero(occupied)):
        if labels[start]:
            continue
        current += 1
        labels[start] = current
        stack = [start]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and occupied[ni, nj] and not labels[ni, nj]:
                        labels[ni, nj] = current
                        stack.append((ni, nj))
    return labels


def find_people(points_local, cfg=ClusterConfig()):
    """Person-sized clusters in a robot-local cloud, nearest-first."""
    p = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    f, l, z = p[:, 0], p[:, 1], p[:, 2]
    keep = (
        (z >= cfg.z_min) & (z <= cfg.z_max)
        & (f >= cfg.forward_min) & (f <= cfg.forward_max) & (np.abs(l) <= cfg.left_max)
    )
    p = p[keep]
    if len(p) == 0:
        return []
    rows = int(np.ceil((cfg.forward_max - cfg.forward_min) / cfg.cell)) + 1
    cols = int(np.ceil(2 * cfg.left_max / cfg.cell)) + 1
    gi = np.clip(((p[:, 0] - cfg.forward_min) / cfg.cell).astype(int), 0, rows - 1)
    gj = np.clip(((p[:, 1] + cfg.left_max) / cfg.cell).astype(int), 0, cols - 1)
    counts = np.zeros((rows, cols), dtype=int)
    np.add.at(counts, (gi, gj), 1)
    labels = _label(counts >= cfg.min_cell_points)
    point_labels = labels[gi, gj]

    people = []
    for k in range(1, labels.max() + 1):
        sel = p[point_labels == k]
        if len(sel) < cfg.min_points:
            continue
        depth = float(sel[:, 0].max() - sel[:, 0].min())
        width = float(sel[:, 1].max() - sel[:, 1].min())
        if depth > cfg.max_footprint or width > cfg.max_footprint or max(depth, width) < cfg.min_footprint:
            continue
        top = float(sel[:, 2].max())
        if top < cfg.min_top:
            continue
        torso = sel[(sel[:, 2] >= cfg.torso_z[0]) & (sel[:, 2] <= cfg.torso_z[1])]
        body = torso if len(torso) >= cfg.min_torso_points else sel
        people.append(Cluster(float(np.median(body[:, 0])), float(np.median(body[:, 1])),
                              len(sel), top, depth, width))
    return sorted(people, key=lambda c: np.hypot(c.forward, c.left))
```

- [ ] **Step 4: Run the perception tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_follow_perception.py -q`

Expected: PASS — `7 passed`.

- [ ] **Step 5: Replace the runner tests**

Replace the entire contents of `tests/test_robot_follow_cli.py` with:

```python
import math
import sys

import numpy as np
import pytest

import robot_follow


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


def test_perceive_finds_a_person_in_a_base_frame_cloud():
    rng = np.random.default_rng(3)
    phi = rng.uniform(-math.pi / 2, math.pi / 2, 900)
    forward = 1.4 - 0.18 * np.cos(phi)  # front half of a person standing 1.4 m ahead
    left = 0.2 + 0.18 * np.sin(phi)  # ... and 0.2 m to the left
    z = rng.uniform(0.05, 1.75, 900)
    points_base = np.column_stack([-left, forward, z])  # base +x points right (BASE_LEFT_SIGN = -1)

    perception = robot_follow.perceive(points_base, 5.0)

    assert perception.t == 5.0
    assert perception.points.shape == (900, 3)
    [person] = perception.people
    assert person.forward == pytest.approx(1.4 - 0.7 * 0.18, abs=0.05)
    assert person.left == pytest.approx(0.2, abs=0.05)
    assert person.hist is None


def test_perceive_with_nobody_there():
    assert robot_follow.perceive(np.empty((0, 3)), 1.0).people == ()


def test_runner_imports_without_bbos():
    assert "bbos" not in sys.modules
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_robot_follow_cli.py -q`

Expected: FAIL; the output contains `cannot import name 'PoseEngine'`.

- [ ] **Step 7: Replace `scripts/robot_follow.py`**

Replace the entire contents of `scripts/robot_follow.py` with:

```python
"""Person-follow runner for BracketBot. Copied to /tmp by robot_dashboard.py.

Depth only: people are person-sized clusters in camera.points (see
follow_perception.py); no camera image and no neural network. Needs only BBOS
and numpy, so it runs in the BBOS venv like robot_base_mode.py. While running it
is the only writer of drive.ctrl and led.ctrl. Every decision lives in
follow_core.FollowLoop; this file only moves data between BBOS topics and the loop.

    ~/bbos/.venv/bin/python /tmp/robot_follow.py --pid-file /tmp/f.pid     # from the dashboard
    ~/bbos/.venv/bin/python /tmp/robot_follow.py --check                   # gate G0: list clusters
    ~/bbos/.venv/bin/python /tmp/robot_follow.py --dry-run --no-heartbeat  # gate G2
    ~/bbos/.venv/bin/python /tmp/robot_follow.py --rotate-only ...         # gate G3
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
    parse_command, start_refusal, status_line, wheel_twist,
)
from follow_perception import ClusterConfig, base_to_local, find_people

PERIOD = 0.02  # 50 Hz control loop; drive.ctrl times out after 0.1 s
STATUS_PERIOD = 0.2
CSV_PERIOD = 0.05
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
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--dry-run", action="store_true", help="compute and log; never open drive.ctrl")
    parser.add_argument("--rotate-only", action="store_true", help="forward speed held at 0 (gate G3)")
    parser.add_argument("--no-heartbeat", action="store_true", help="only with --dry-run: no dashboard needed")
    parser.add_argument("--check", action="store_true", help="gate G0: print the clusters seen for 3 s, then exit")
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


def wait_fresh(reader, timeout, topic):
    started = time.monotonic()
    while not reader.ready():
        if time.monotonic() - started > timeout:
            raise RuntimeError(f"no fresh {topic} sample; is its daemon running?")
        time.sleep(0.002)
    return reader.data


def other_drive_writers():
    found = []
    for pattern in DRIVE_WRITER_PATTERNS:
        result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
        found.extend(
            line.strip() for line in result.stdout.splitlines()
            if line.strip() and int(line.split()[0]) != os.getpid()
        )
    return found


def perceive(points_base, t, cluster_cfg=ClusterConfig()):
    """One camera.points frame -> Perception with every person-sized cluster."""
    local = base_to_local(points_base)
    people = tuple(PersonObservation(c.forward, c.left) for c in find_people(local, cluster_cfg))
    return Perception(t, people, local)


def cloud(data):
    n = int(data["num_points"])
    return np.asarray(data["points"][:n])


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


def run_check(reader_cls, seconds=3.0):
    """Read-only: print the person-sized clusters in each camera.points frame."""
    with reader_cls("camera.points", keeptime=False) as points:
        wait_fresh(points, 3.0, "camera.points")
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if points.ready():
                people = find_people(base_to_local(cloud(points.data)))
                print(json.dumps({"clusters": [
                    {"forward": round(c.forward, 2), "left": round(c.left, 2), "points": c.points,
                     "top": round(c.top, 2), "depth": round(c.depth, 2), "width": round(c.width, 2)}
                    for c in people
                ]}), flush=True)
            time.sleep(0.05)


def control_loop(args, cfg, readers, drive, led, wheel_diam, robot_width):
    points, imu, drive_state = readers
    loop = FollowLoop(cfg, args.gap)
    commands = queue.Queue()
    if not args.no_heartbeat:
        start_command_reader(commands)
    now = time.monotonic()
    last_heartbeat = now
    command_stop = False
    rpy = np.zeros(3)
    measured = (0.0, 0.0)
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
            perception = perceive(cloud(points.data), t) if points.ready() else None

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

    if args.check:
        run_check(Reader)
        return
    cfg = loop_config(args)
    drive_cfg = Config("drive")
    wheel_diam = float(drive_cfg.wheel_diam)
    robot_width = float(getattr(drive_cfg, "robot_width", cfg.robot_width))
    low_battery_v = getattr(Config("base"), "low_battery_v", None)
    with ExitStack() as stack:
        points = stack.enter_context(Reader("camera.points", keeptime=False))
        imu = stack.enter_context(Reader("imu.orientation", keeptime=False))
        drive_state = stack.enter_context(Reader("drive.state", keeptime=False))
        drive_status = stack.enter_context(Reader("drive.status", keeptime=False))

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

        drive = None
        if not args.dry_run:
            drive = stack.enter_context(Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False))
        led = stack.enter_context(Writer("led.ctrl", Type("led_ctrl"), keeptime=False))
        mode = "dry run" if args.dry_run else "rotate only" if args.rotate_only else f"v_max {cfg.v_max:.2f} m/s"
        print(f"[follow] follow active ({mode}, gap {args.gap:.2f} m) - stand in front of the robot", flush=True)
        try:
            control_loop(args, cfg, (points, imu, drive_state), drive, led, wheel_diam, robot_width)
        finally:
            if drive is not None:
                for _ in range(6):
                    write_twist(drive, 0.0, 0.0)
                    time.sleep(PERIOD)
                print("[follow] stopped; zero twist sent", flush=True)
            write_led(led, (0, 0, 0))


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

- [ ] **Step 8: Run the runner tests and syntax-check the runner**

Run: `uv run --extra dev python -m pytest tests/test_robot_follow_cli.py -q`

Expected: PASS — `11 passed`.

```bash
python -m py_compile scripts/robot_follow.py
```

- [ ] **Step 9: Run the whole suite (all waves are in now)**

Run: `uv run --extra dev python -m pytest -q`

Expected: PASS — `110 passed`.

- [ ] **Step 10: Commit only these files**

```bash
git add scripts/follow_perception.py tests/test_follow_perception.py scripts/robot_follow.py tests/test_robot_follow_cli.py
git commit -m "feat(follow): depth-only person clusters and runner" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- scripts/follow_perception.py tests/test_follow_perception.py scripts/robot_follow.py tests/test_robot_follow_cli.py
```

---

### Task 5: Robot gates (hardware; person at the e-stop)

**Files:**
- Modify: `docs/robot-facts.md` (gate results)
- Modify, only if a gate says so: `scripts/follow_perception.py` (`BASE_LEFT_SIGN`), `scripts/follow_core.py` (`FollowConfig.self_mask`, finally `v_max`), `scripts/robot_follow.py` (`WHEEL_ORDER`, `WHEEL_SIGNS`), `scripts/robot_dashboard.py` (`--follow-v-max` default), `tests/test_follow_perception.py`, `tests/test_robot_follow_cli.py`

**Interfaces:**
- Consumes: everything above; the robot reachable as `ssh bot` (or another alias from the dashboard's list), BBOS at `~/bbos`, `uv` at `~/.local/bin/uv`.
- Produces: recorded evidence per gate, and the final `v_max` default of 0.30 m/s.

Gate order: **G0 → G2 → G3 → G4a → G5 (0.15 m/s) → G6 → G4b → G5 again (0.30 m/s) → raise the default.** (G1, the TensorRT engine gate, no longer exists.) Any failed gate stops the sequence: record what happened, fix, and repeat that gate. The dashboard deploys the runner files itself; for G0/G2 copy them by hand as shown. Keep every CSV (`scp 'bot:/tmp/baymax_follow_*.csv' artifacts/follow/`); `artifacts/` is gitignored.

- [ ] **Step 1: G0: read-only probe and cluster check**

Start the BBOS depth daemon the way this robot normally starts it (it was **off** at the last probe; `camera.depth` and `camera.points` were not publishing) and write the exact method into `docs/robot-facts.md`. Then, with nobody in front of the robot:

```bash
mkdir -p artifacts/follow
scp scripts/probe_follow.py scripts/robot_follow.py scripts/follow_core.py scripts/follow_perception.py bot:/tmp/
ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/probe_follow.py --out /tmp/follow_probe'
ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/robot_follow.py --check'
```

Now one person stands about 1 m ahead and 0.5 m to the robot's **left**:

```bash
ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/probe_follow.py --out /tmp/follow_probe_left --person-left'
ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/robot_follow.py --check'
scp -r bot:/tmp/follow_probe bot:/tmp/follow_probe_left artifacts/
```

Pass: `camera.depth`, `camera.points`, `imu.orientation`, `drive.state`, and `drive.status` each report a nonzero `rate_hz`; with the area clear `--check` prints no clusters (or only ones you can name — note them); with the person there it prints one cluster at `forward` ≈ 0.9 and `left` ≈ +0.5 (after the sign fix below). Time the check: if `camera.points` arrives at N Hz, `--check` should print about N lines per second; much fewer means clustering is too slow on this CPU — stop and report.

- [ ] **Step 2: G0: apply what the probe found**

1. `cloud.verdict` in `artifacts/follow_probe_left/report.json` says whether base +x points left or right. If it says `+x is LEFT`, change `BASE_LEFT_SIGN = -1.0` to `BASE_LEFT_SIGN = 1.0` in `scripts/follow_perception.py` and update `test_base_frame_points_become_forward_left_up` to expect `[1.0, 0.2, 0.5]` and `test_perceive_finds_a_person_in_a_base_frame_cloud` to build `points_base` from `+left`.
2. If `cloud.self_points` in `artifacts/follow_probe/report.json` is above 0, convert `self_box_base_xyz_min/max` (base x, y, z) to one robot-local box with a 3 cm margin and set it as the `FollowConfig` default: `forward` = [y_min − 0.03, y_max + 0.03]; `left` = [−x_max − 0.03, −x_min + 0.03] when `BASE_LEFT_SIGN` is −1, or [x_min − 0.03, x_max + 0.03] when it is +1; `up` = [z_min − 0.03, z_max + 0.03]. For example `self_mask: tuple[...] = ((0.0, 0.33, -0.27, 0.27, 0.9, 1.5),)`.
3. Append a `## Person follow (gate results)` section to `docs/robot-facts.md` with the date and robot ID, and record: the depth-daemon start method; each topic's rate and the `camera.depth` shape; base +x direction and `BASE_LEFT_SIGN`; the self box and resulting `self_mask`; `Config("drive")` wheel diameter, width, speed clamps, and command timeout; `Config("base").low_battery_v`; which objects in the room `--check` reports as person-sized.
4. Run the suite and commit:

```bash
uv run --extra dev python -m pytest -q
git add docs/robot-facts.md scripts/follow_perception.py scripts/follow_core.py tests/test_follow_perception.py tests/test_robot_follow_cli.py
git commit -m "chore(follow): gate G0 results" -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

**An agent stops here.** G2–G6 need people in front of the robot and an e-stop operator.

- [ ] **Step 3: G2: dry run (never opens `drive.ctrl`)**

Tape floor marks straight ahead of the wheel axle at 0.6, 1.0, and 1.5 m, and at 1.0 m ±30°. The range is measured to the **front of the torso** (what the camera sees), so each person stands with the front of their torso directly above the mark (hold a straight board vertically against the tape to line it up). Then:

```bash
ssh -t bot '~/.local/bin/uv run --no-sync --project ~/bbos python /tmp/robot_follow.py --dry-run --no-heartbeat'
```

1. Person A stands at the 1.0 m mark. Within about a second the state line changes to `FOLLOWING` and the LED turns solid green.
2. Person A stands on each mark for 5 s; read `range` from the `FOLLOW_STATUS` lines.
3. Person A stands 0.3 m in front of a wall, then next to a chair, then next to any pillar or coat rack in the room. The first two must keep `FOLLOWING` on A; record whether the pillar/coat rack steals the lock.
4. Person B walks slowly past A, 0.3 m to A's side. `range` must stay on A.
5. Person A leaves the camera's view: `LOST` (blinking amber), and back to `FOLLOWING` when A returns within 10 s.
6. `Ctrl-C`, then copy the CSV and run `python scripts/follow_log_report.py artifacts/follow/<file>.csv`.

Pass: every mark within ±0.05 m; wall and chair cases hold; bystander rejected. If every range is off by a similar amount, **do not add an offset in code**: the point cloud is miscalibrated (a reflash has shifted another robot's cloud by 0.78 m). Recalibrate the depth pipeline the BBOS way, repeat G0 and G2, and record it.

- [ ] **Step 4: G3: rotate only**

```bash
python scripts/robot_dashboard.py --follow-rotate-only
```

Open <http://127.0.0.1:8020>, stand about 1.2 m in front, press **Follow me**, then walk a slow arc ±60° around the robot at 1.0–1.5 m for 60 s. Press **Stop action**, copy the CSV, and run the report.

Pass: `bearing_within_10deg_fraction` ≥ 0.95, `omega_sign_agreement` ≥ 0.9, and no `odometry-mismatch` exit. If `omega_sign_agreement` is low or the runner exits with `odometry-mismatch`, set `WHEEL_ORDER = (1, 0)` in `scripts/robot_follow.py`, record it, and repeat G3.

- [ ] **Step 5: G4a: follow at 0.15 m/s**

`python scripts/robot_dashboard.py` (default cap 0.15 m/s) on an open floor with at least 4 m clear ahead. Run two sessions (each start writes its own CSV):

1. **Stroll:** lock at 1.0 m; the person stands 20 s, then shuffles away at about 0.1 m/s (one short step every 2 s) for 30 s, then stops on a tape mark. When the robot has settled, tape-measure from the wheel axle to the front of the person's torso. Move the slider to 0.7 m and confirm it settles there too. Stop.
2. **Step-backs:** lock at 1.0 m; the person takes one quick 0.5 m step back, waits 5 s, five times. Stop.

Pass: session 1 `in_band_fraction` ≥ 0.95; the tape agrees with the logged `range` within 0.05 m; `v_sign_agreement` ≥ 0.9; `last_rule` is `stop`. In session 2 the robot is back in the band within 5 s of every step. If `v_sign_agreement` is low or the runner exits with `odometry-mismatch` (G3 already proved the turn direction), flip both wheel signs and swap the wheel order, which negates `v` and leaves `omega` as it was: set `WHEEL_SIGNS = (-1.0, -1.0)` and change `WHEEL_ORDER` from `(0, 1)` to `(1, 0)` or back. Record it, then repeat G3 and G4a.

- [ ] **Step 6: G5 at 0.15 m/s: obstacle**

While following a person strolling at about 0.1 m/s, a helper places a 30 cm box on the floor in the robot's path about 0.5 m ahead of it. Then remove it.

Pass: the robot stops before contact (state `BLOCKED`, LED solid amber); the report's `blocked_to_stop_s` ≤ 0.5; it resumes following about 0.5 s after the box is removed. Film it: the robot must be stopped within 0.3 s of the box entering the camera's view.

- [ ] **Step 7: G6: link loss**

While following at 0.15 m/s: (a) press `Ctrl-C` in the dashboard terminal; (b) restart, follow again, then turn the laptop's Wi-Fi off. Pass: both times the robot is stationary within 1.2 s, and the CSV's last rule is `stop` (a) or `heartbeat` (b).

- [ ] **Step 8: G4b and G5 at 0.30 m/s**

`python scripts/robot_dashboard.py --follow-v-max 0.30`. Repeat the G4a stroll session with the person walking slowly at up to 0.25 m/s (a normal step every 2–3 s), then repeat G5.

Pass: `in_band_fraction` ≥ 0.95 with the person at ≤ 0.25 m/s; no `not-upright` exit and no visible pitching or oscillation of the balancing base; G5 criteria as before.

- [ ] **Step 9: Record the gates and raise the default speed cap**

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

Run `uv run --extra dev python -m pytest -q` and expect PASS (`110 passed` during planning). Update the README sentence "Until gate G4b passes, the speed cap is 0.15 m/s" to "The speed cap is 0.30 m/s (0.15 m/s with `--follow-v-max 0.15` when bringing up a new robot)". Commit with the message `feat(follow): 0.30 m/s default after robot gates passed` and the trailer.

- [ ] **Step 10: Finish the branch**

Use superpowers:finishing-a-development-branch to decide how `feature/person-follow` is merged.
