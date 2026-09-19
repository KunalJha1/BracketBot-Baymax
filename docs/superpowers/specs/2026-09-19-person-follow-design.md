# Person-follow navigation — design

Status: design approved in conversation 2026-09-19; not implemented; no hardware evidence yet.
Revised 2026-09-19 after prototyping (see §11), then switched to **depth-only
perception** (no neural network; see §12). Implementation plans:
`docs/superpowers/plans/2026-09-19-person-follow.md` (Tasks 1–7 done as written),
`docs/superpowers/plans/2026-09-19-person-follow-depth-only.md` (the change and the rest).

## 1. Goal

BracketBot follows one chosen person and holds a runtime-adjustable following gap
(0.6–1.5 m, default 1.0 m) to within **±20 cm**, stopping for obstacles, never
reversing, and stopping safely on any loss of data, link, or balance.

## 2. Feasibility verdict

Feasible, with one physical limit.

| Factor | Evidence | Consequence |
|---|---|---|
| Range sensing | `camera.depth` (640×384, uint16 mm) and `camera.points` (base frame, IMU-pitch corrected by the depth daemon, same internal frame/timestamp as depth). Baseline 65 mm, depth fx ≈ 161.5 px. | At ~1.1 m camera-to-torso, ≈ ±3 cm per ¼ px disparity error; torso median + filtering → target ±5 cm. |
| Minimum stereo range | Head camera is 1.5 m above the floor. | Not a constraint: the floor at the robot's feet is ≥ 1.5 m from the camera. |
| Field of view | Rectified depth ≈ 126° × 109°; camera 1.5 m up, 33–35° down. | Whole person (feet to head) visible from 0.6 m to 1.5 m. |
| Latency | ~150 ms perception at 0.3 m/s. | ≈ 4.5 cm error contribution. |
| **Speed** | `drive.ctrl` linear clamp **0.3 m/s** (`docs/robot-facts.md`). People walk 1.2–1.4 m/s. | ±20 cm is achievable only while the person stands, turns, or moves ≤ ~0.25 m/s. At walking pace the gap opens and closes again when they slow. Raising the clamp is a separate safety decision, out of scope. |

## 3. Decisions

| Topic | Decision |
|---|---|
| Meaning of "20 cm" | Hold the configured gap to ±20 cm. |
| Target | A person, found as a person-sized cluster in the depth point cloud. No neural network, no camera image. |
| Gap | Adjustable at runtime, 0.6–1.5 m, default 1.0 m. |
| Lock-on | After Follow is pressed: the only person-sized cluster standing 0.5–2.0 m ahead within ±30° for 0.5 s. |
| Lost for 10 s | Return to SEARCHING and re-lock automatically on whoever stands in the start zone. |
| Obstacles | Stop-only corridor guard; no path planning. |
| Architecture | Everything closed-loop runs on the robot's Jetson; the dashboard only starts/stops, sets the gap, and sends a heartbeat. |
| Person comes closer than gap − 20 cm | Hold still. **No reversing in v1** (no rear sensing). |

## 4. Non-goals (v1)

- Keeping up with normal walking speed; changing the daemon speed clamp.
- Driving backward, steering around obstacles, map/SLAM-based planning.
- Following through doors/corners after the person leaves the camera's view.
- Re-identifying a person after a long absence (> 10 s) — whoever stands in the start zone is locked next.
- Telling a person apart from person-sized objects (pillars, coat racks, tall plants), or from a
  wall or another person within about 10 cm of them.
- Voice start/stop, sound cues (the speaker would be a second hardware writer).
- Arm motion of any kind during follow.
- Using facial-expression estimates for anything.

## 5. Architecture

```text
laptop: robot_dashboard.py ──SSH stdin (JSON lines: gap, heartbeat, stop)──┐
                          ◄──SSH stdout (FOLLOW_STATUS json, 5 Hz)──────┐  │
                                                                        │  ▼
robot (Jetson, bbos venv, files in /tmp):                          robot_follow.py (runner)
  camera.points ─────► follow_perception: person-sized clusters ──┐
  imu.orientation, drive.state (odometry) ────────────────────────┤
                                                                  ▼
                                      follow_core: Tracker → Controller → Supervisor
                                                                  │ 50 Hz
                                                                  ▼
                                                   drive.ctrl (sole writer), led.ctrl
```

### 5.1 Files

| File | Runs on | Responsibility | Depends on |
|---|---|---|---|
| `scripts/follow_core.py` | robot + laptop | Pure logic: data types, lock-on, tracker, controller, supervisor, obstacle corridor, state machine, command parsing. **No BBOS, no TensorRT imports.** | numpy |
| `scripts/follow_perception.py` | robot + laptop | Base-frame → robot-local conversion and person-sized cluster finding in the point cloud. Pure numpy. | numpy |
| `scripts/robot_follow.py` | robot | Runner: BBOS readers/writers, main loop, signals, pid file, stdin/stdout protocol, CSV log, zero-twist exit. Thin — no decisions. | bbos, numpy, the two modules above |
| `scripts/robot_dashboard.py` | laptop | Follow card, start/stop, gap slider, heartbeat, status parsing, resource exclusion, `--simulate` fake runner. | existing dashboard |
| `scripts/probe_follow.py` | robot | Read-only gate G0 probe: topic rates/fields, configs, self points, base x sign. | bbos |

The runner and both modules are deployed to `/tmp` by the existing `_deploy`
path, like `robot_base_mode.py`, and import each other by bare name (the
script's directory is on `sys.path`). The runner needs only BBOS and numpy, so it
runs in the robot's BBOS venv through the dashboard's existing
`remote_python_command`, exactly like `robot_base_mode.py`.

### 5.2 Base-frame convention

Follow `bbapps/nav/main.py`: base **+y forward, +x lateral**, z up. Perception
converts points to robot-local **(forward, left, up)** once
(`forward = y`, `left = BASE_LEFT_SIGN · x`, default −1 = +x points right); all
of `follow_core` works in (forward, left). Bearing is `atan2(left, forward)`:
positive = person on the left, and positive ω turns left. The sign of x is
verified at G0 and the ω sign at G3.

## 6. Components

### 6.1 Perception (`follow_perception.py`)

Depth only. Every new `camera.points` frame is converted to robot-local
`(forward, left, up)` (5.2) and searched for person-sized clusters:

1. **Crop:** keep points 0.10–2.0 m high, 0.3–3.5 m ahead, and within ±2.5 m
   to the side (drops floor, ceiling, and far clutter).
2. **Grid:** bin the kept points into a 10 cm floor grid over (forward, left); a
   cell with at least 3 points is occupied.
3. **Cluster:** group occupied cells that touch (8-neighbour connectivity).
4. **Person-sized:** footprint extent at most 0.8 m in both forward and left, at
   least 0.15 m in one of them; highest point at least 1.2 m; at least 60 points.
5. **Position:** per-axis median of the cluster's points 0.8–1.6 m high (the
   torso), or of all its points if fewer than 10 are in that band.

All values live in a frozen `ClusterConfig` in `follow_perception.py`. Output
per frame: `Perception(t, people, points)` with one
`PersonObservation(forward, left, score=1.0, hist=None)` per person-sized
cluster and the whole cloud in robot-local coordinates for the corridor check.
Pure numpy; a few milliseconds per frame on the Jetson CPU.

Known limits (accepted): any tall, narrow object (pillar, coat rack, tall plant)
is person-sized; a person within about 10 cm of a wall or another person merges
with it into one cluster that is either too wide (rejected) or displaced.

**What "range" measures:** the depth camera sees the front of the person, so the
cluster position (and therefore the held gap) is measured from the base origin to
the front of the torso, about 0.1 m nearer than the person's centre.

### 6.2 Tracker (`follow_core.Tracker`)

- **Odometry frame:** integrate robot motion from `drive.state.vel` (turns/s ×
  π·wheel_diam per wheel; wheel signs verified at G3/G4a).
  Person positions are transformed into this frame so that the robot's own motion is
  not mistaken for the person moving.
- **Lock-on (SEARCHING):** candidates within range 0.5–2.0 m and |bearing| ≤ 30°,
  associated frame-to-frame by nearest position (≤ 0.3 m). A candidate seen for
  at least 0.5 s and present in ≥ 80% of the frames of the last 0.5 s is locked.
  If two candidates qualify at once, lock neither and keep waiting.
- **Tracking:** constant-velocity Kalman filter on (x, y, vx, vy) in odometry frame
  (measurement σ 0.08 m, acceleration σ 1 m/s², position σ capped at 0.4 m so the gate
  stays bounded while coasting; on LOST the velocity is zeroed and position σ grows
  by 0.5 m).
  Association gate: Mahalanobis distance ≤ 3σ. (The tracker also supports a
  histogram appearance gate, but depth-only perception supplies no histogram, so
  association is by position and motion alone.) If two observations fall inside
  the gate with scores within 10% of each other, treat the frame as **ambiguous**:
  coast, don't update.
- Other people entering the start zone while one person is locked are ignored.
- **Output:** `Track(t, x, y, range, bearing, v_radial, age_since_update)` where
  `v_radial` is the person's velocity component along the robot→person direction
  (positive = moving away), or `None`.

### 6.3 Controller (`follow_core.control`)

```text
shrink(x, d) = 0 if |x| ≤ d else sign(x)·(|x| − d)     # continuous deadband
v      = (max(v_radial, 0) + k_r · shrink(range − gap, 0.05 m)) · cos(bearing)
v      = 0                       if |bearing| > 35°     # turn in place first
v      = clamp(v, 0, v_max)                             # never negative: no reversing
ω      = clamp(k_θ · shrink(bearing, 3°), −ω_max, ω_max)
```

The deadband is continuous (no step at its edge) so a balancing base does not
chatter around the gap.

Rate limiting (applied after the supervisor, inside the runner): linear +0.4 m/s²
accelerating, −0.8 m/s² braking; angular 1.5 rad/s². The drive daemon's own
S-curve limiter still applies on top.

### 6.4 Supervisor (`follow_core.Supervisor`)

Evaluated every 20 ms. Exit rules are checked in order and the first one wins.
Restrictions are all applied (each can only reduce motion); the status reports
the first that fired.

| # | Condition | Action |
|---|---|---|
| 1 | `stop` command, stdin EOF, SIGTERM/SIGINT/SIGHUP | Stop, exit (`stop`) |
| 2 | Heartbeat older than 1.0 s | Stop, exit (`heartbeat`) |
| 3 | \|roll\| or \|pitch\| ≥ 25° (same threshold as `robot_base_mode.py`) | Stop, exit (`not-upright`) |
| 4 | Wheel feedback opposite to the sent command for ≥ 0.5 s (\|v\| ≥ 0.05 with measured \|v\| ≥ 0.03, or \|ω\| ≥ 0.2 with measured \|ω\| ≥ 0.1). Catches a wheel sign/order bug that would otherwise turn feed-forward into positive feedback. | Stop, exit (`odometry-mismatch`) |
| 5 | Not tracking (SEARCHING or LOST) | v = ω = 0 (`no-track`) |
| 6 | `camera.points` older than 0.3 s (obstacle state unknown) | v = 0; ω allowed (`points-stale`) |
| 7 | No track update for > 0.3 s | v = ω = 0, ramped by the rate limiter (`track-stale`) |
| 8 | Obstacle in corridor (6.5) | v = 0; ω allowed (`blocked`) |
| 9 | range < 0.45 m | v = 0 (`min-range`) |
| — | Always | v clamped to [0, v_max], ω to ±ω_max |

No track update for > 1.0 s moves the state machine to LOST (6.6).

**Start preconditions** (runner refuses and exits non-zero with a reason):
IMU upright; `camera.points` fresh; no known `drive.ctrl` writer running
(`pgrep -af` for `greeter/main.py`, `nav/main.py`, `bbapps/teleop.py`,
`quest_teleop/main.py`, `leader_follower_teleop.py`, `live_inference.py`); `drive.status.voltage` ≥ `Config("base").low_battery_v`.

**Exit:** in `finally`, write a zero twist 6 times at 20 ms intervals with pacing
disabled (`keeptime=False`), set LEDs off, remove pid file.

### 6.5 Obstacle corridor

- Points from `camera.points` in base frame.
- Height band 0.05–1.70 m (drops floor and ceiling).
- Corridor: |x| ≤ robot_width/2 + 0.10 m (≈ 0.265 m); 0 < y ≤ 0.60 m.
- Exclude points within 0.35 m (horizontal) of the tracked person's position.
- Exclude a self-mask box around the robot's own body/wheels/arms measured at G0.
- Blocked when ≥ 30 remaining points; cleared after 0.5 s below the threshold
  (hysteresis).

### 6.6 State machine

| State | Robot | LED | Transitions |
|---|---|---|---|
| SEARCHING | stationary | slow blue pulse | lock → FOLLOWING |
| FOLLOWING | controller output | solid green | obstacle → BLOCKED; no update > 1 s → LOST |
| BLOCKED | v = 0, ω allowed | solid amber | corridor clear 0.5 s → FOLLOWING; no update > 1 s → LOST |
| LOST | stationary | blinking amber | locked track re-associates within 10 s → FOLLOWING; else → SEARCHING |

LED is written by the runner through `led.ctrl`; the dashboard disables its LED
actions while follow is active.

### 6.7 Parameters

All in one frozen `FollowConfig` dataclass in `follow_core.py`.

| Name | Default | Notes |
|---|---|---|
| `gap_default` / `gap_min` / `gap_max` | 1.0 / 0.6 / 1.5 m | runtime `gap` commands clamped |
| `band` | 0.20 m | acceptance band only; not used in control |
| `deadband_range` | 0.05 m | |
| `k_r` | 0.8 s⁻¹ | |
| `v_max` | 0.15 m/s | G4b runs at 0.30 once G4a passes; the default becomes 0.30 only after G4b passes |
| `k_theta` | 1.5 s⁻¹ | |
| `deadband_bearing` | 3° | |
| `omega_max` | 0.8 rad/s | daemon clamp is 1.0 |
| `turn_in_place_bearing` | 35° | |
| `accel_up` / `accel_down` | 0.4 / 0.8 m/s² | |
| `alpha_max` | 1.5 rad/s² | |
| `min_range` | 0.45 m | |
| `heartbeat_timeout` | 1.0 s | |
| `perception_stale` / `lost_after` | 0.3 / 1.0 s | |
| `lost_timeout` | 10 s | |
| `upright_deg` | 25° | |
| `corridor_margin` / `corridor_length` | 0.10 / 0.60 m | |
| `corridor_min_points` | 30 | |
| `person_exclusion_radius` | 0.35 m | |
| `self_mask` | `()` | robot-body boxes in the depth cloud, filled in from G0 |
| `odom_mismatch_time` | 0.5 s | |
| `meas_sigma` / `accel_sigma` / `max_pos_sigma` / `lost_pos_sigma` | 0.08 m / 1.0 m/s² / 0.4 m / 0.5 m | tracker |
| `lock_window` / `lock_fraction` | 0.5 s / 0.8 | presence fraction over the window |
| `lock_range_min` / `lock_range_max` / `lock_bearing_max` | 0.5 / 2.0 m / 30° | start zone |

## 7. Dashboard ↔ runner protocol

**stdin (dashboard → runner), one JSON object per line:**

```json
{"type": "heartbeat"}
{"type": "gap", "m": 1.2}
{"type": "stop"}
```

Heartbeat every 250 ms. Unknown or malformed lines are logged and ignored. Out-of-range
gaps are clamped and the clamped value is reported back.

**stdout (runner → dashboard):** existing free-text `[follow] …` log lines, plus
at 5 Hz:

```text
FOLLOW_STATUS {"state":"FOLLOWING","range":1.07,"gap":1.0,"error":0.07,"bearing_deg":-4.2,"v":0.08,"w":-0.1,"blocked":false,"age_ms":60,"rule":"ok"}
```

**Dashboard behaviour:**
- Follow card: Start/Stop toggle (key `F`), gap slider 0.6–1.5 m (step 0.1),
  status line (state, range, gap error; while SEARCHING: "stand in front of the
  robot"), last reason for refusal/exit.
- Follow holds the `base`, `led`, and `camera` resources: it is mutually exclusive
  with Lean and every gesture/routine; LED actions are disabled while it runs. The
  dashboard refuses to start Follow while Lean is active (the Lean runner is the only
  `base.mode` writer, and its request expires to BALANCE within 0.25 s once stopped).
- Every stdin line (gap or heartbeat) refreshes the heartbeat timer.
- `Esc` / Stop sends `{"type":"stop"}` and then uses the existing pid-file
  SIGTERM path (`_request_remote_stop`).
- The SSH process is started with `stdin=PIPE`; the heartbeat thread stops when
  the process exits.
- `--simulate` uses an in-process fake runner that emits scripted `FOLLOW_STATUS`
  lines and honours gap/stop, so the UI and API are fully testable offline.

**Runner log:** CSV at 20 Hz to `/tmp/baymax_follow_<timestamp>.csv`: time, state,
range, bearing, gap, error, v_cmd, ω_cmd, v_sent, ω_sent, blocked, corridor
points, track age, supervisor rule fired.

## 8. Testing

All laptop tests run under the existing pytest suite; none require the robot.

1. **`tests/test_follow_core.py`**
   - Lock-on: a person in the start zone for ≥ 0.5 s locks; 0.3 s does not; two
     people in the zone lock neither; people outside the zone are ignored.
   - Association: a bystander crossing between robot and target does not take the
     track; ambiguous frames coast.
   - Controller: deadband, `cos(bearing)` scaling, turn-in-place above 35°,
     v never negative, clamps; rate limiter respects all three limits.
   - Supervisor: one test per row of 6.4, including priority (e.g. heartbeat loss
     beats an obstacle).
   - Corridor: person points excluded, self-mask excluded, floor excluded,
     hysteresis.
   - Protocol: gap clamping, malformed lines ignored, stop command.
2. **`tests/test_follow_sim.py`** — closed-loop 2D kinematic simulation
   (unicycle robot with the rate limits; observations at 15 Hz with 100 ms latency,
   ±5 cm range noise, ±2° bearing noise; `v_max` = 0.30 m/s, the target
   configuration after G4b). Scenarios and assertions:
   - Standing person at 1.0 m ± 0.5 m start: settles within ±5 cm.
   - Walking away at 0.20 m/s: ≥ 95% of samples after settling within ±20 cm.
   - Walking at 1.0 m/s for 3 s then stopping: gap opens, then returns into the band.
   - Side-step 1 m: turns in place, re-centres, no forward motion while |bearing| > 35°.
   - Person steps toward the robot: robot never reverses.
   - 2 s occlusion: coasts, ramps to zero, LOST, recovers within 10 s.
   - Obstacle appears: BLOCKED within one supervisor tick of corridor detection.
   - Stale/absent heartbeat: stops.
   This is **simulation evidence**, reported as such, never as hardware evidence.
3. **`tests/test_follow_perception.py`** — synthetic point clouds: a standing
   person is found at the right position; floor, a table, a wall, and a short box
   are not; two separated people give two clusters; a person far beyond 3.5 m and
   an empty cloud give none.
4. **`tests/test_robot_dashboard.py`** additions — Follow start/stop in simulate
   mode, exclusion with Lean/gestures/LED actions, gap validation, heartbeat thread
   lifecycle, status parsing.

## 9. Robot rollout gates

Every gate that can move the robot has a person at the physical e-stop. Results
are recorded in `docs/robot-facts.md` with date and robot ID.

| Gate | Procedure | Pass criterion |
|---|---|---|
| G0 — read-only probe | Depth daemon running? `camera.depth`/`camera.points` rate; base-frame x sign; drive clamp and command timeout from `Config("drive")`; self points (arms at home) inside the corridor; `robot_follow.py --check` lists clusters (none with the area clear, one near 1.0 m with a person there). | All values recorded; self-mask defined. |
| G1 | Removed with the switch to depth-only perception (there is no TensorRT engine). | — |
| G2 — dry run | `robot_follow.py --dry-run` (never opens `drive.ctrl`). Person stands with the front of their torso above taped marks 0.6 / 1.0 / 1.5 m from the wheel axle, straight ahead and ±30°; then 0.3 m beside a wall, next to a chair, and next to any pillar or coat rack; a bystander walks past 0.3 m to the side. | Range within ±5 cm of tape at every mark; lock-on works; the chair is not a cluster; the wall case still finds the person; bystander doesn't steal the track; which tall objects count as people is recorded. |
| G3 — rotate only | `--rotate-only` (v forced to 0). Person walks an arc around the robot. | Keeps person within ±10° bearing at walking pace. |
| G4a — follow, 0.15 m/s | Open floor. Person stands, steps back 0.5 m repeatedly, strolls slowly. | ≥ 95% of FOLLOWING samples within ±20 cm while the person moves ≤ 0.10 m/s; settled tape measurements agree with logged range within 5 cm. |
| G4b — follow, 0.30 m/s | Same, `v_max` = 0.30. | Same criterion for person speed ≤ 0.25 m/s; no visible balance instability. |
| G5 — obstacle | Place a 30 cm box in the corridor at 0.5 m while following. | BLOCKED within 0.3 s of the box entering view; no contact. |
| G6 — link loss | Kill the dashboard / drop Wi-Fi while following. | Robot stationary within 1.2 s. |

G2 must be repeated after any BBOS reflash or depth-daemon change: a reflash has
previously shifted another robot's point cloud by 0.78 m.

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| Person walks faster than 0.3 m/s | Documented limit; gap reopens; dashboard shows the error. |
| A pillar, coat rack, or bystander is person-sized | Start in open space; auto re-lock after 10 s LOST can pick them (user choice); G2 records which objects count. |
| Person merges with a wall or another person within about 10 cm | Cluster rejected or displaced → LOST; documented limit. |
| Point-cloud calibration drift | G2 tape check after every reflash. |
| Braking/acceleration pitch shakes the camera | Depth daemon compensates with IMU pitch; conservative ramps; check at G4. |
| Robot's own arms/wheels in the depth cloud | Self-mask from G0; arms must be idle and at home. |
| Another app writes `drive.ctrl` | Start precondition process check; dashboard exclusivity. |
| Stereo depth weak on untextured clothing or in poor light | Person cluster thins below 60 points → LOST; checked at G2. |

## 11. Revisions after prototyping (2026-09-19)

(The perception-related items below are superseded by §12.)

The core logic, perception decode, runner argument handling, and dashboard
integration were prototyped and tested before the plan was written. The
closed-loop simulation passes on 20 random seeds with 95th-percentile gap error
≤ 5 cm for walking speeds up to 0.25 m/s. That is **simulation evidence only**.
Changes from the first draft:

- Torso region is an axis-aligned rectangle, and the appearance histogram is
  4×4×4 HSV instead of 8×8 hue/saturation (black vs white clothing).
- Continuous deadbands for range and bearing.
- Supervisor gained `no-track` and `odometry-mismatch`; restrictions are applied
  together rather than first-match.
- The runner is a PEP 723 script run with `uv run --script`, mirroring the
  greeter's dependency set, because the BBOS venv is not known to contain
  TensorRT/PyCUDA.
- The fisheye fallback projects points into the raw eye instead of undistorting
  keypoints.
- The dashboard's Esc handler runs before the "focused input" early return, so
  Esc stops the robot even while the gap slider has focus.

## 12. Revision: depth-only perception (2026-09-19)

Requested after Tasks 1–7 of the first plan were implemented: follow without an
object detector. SLAM was considered: it localises the robot, not the person,
so it cannot replace person detection; it could replace wheel odometry, which is
not needed at these speeds and distances. Map-based trail following (the only
approach that needs SLAM) is out of scope because corners are.

What changed:
- Perception is §6.1's depth clustering; the YOLO pose engine, keypoints, hand
  raise, torso colour histogram, `camera.rect`, and the fisheye fallback are gone.
- Lock-on is presence in a start zone (0.5–2.0 m, ±30°, 0.5 s) instead of a raised
  hand. `PersonObservation` loses `hand_raised`.
- After 10 s LOST the robot re-locks automatically (user's choice over stopping).
- The runner needs only BBOS + numpy and runs in the BBOS venv like the Lean runner;
  gate G1 and the TensorRT/fisheye tooling are removed.
- Clustering measured about 9 ms per frame on a laptop for a worst-case 40k-point
  cloud; the Jetson CPU is expected within the 50 Hz loop's budget (checked at G0).
- Bystander rejection relies on position and motion only. In the closed-loop
  simulation with no appearance cue, a bystander crossing in front and one passing
  0.3 m beside the target kept the right person in 20/20 seeds each — simulation
  evidence only; it does not model clusters merging.

## 13. Known follow-ups (open after the final review, 2026-09-19)

Not defects that block the first hardware bring-up, but the next session should
take them in this order:

1. **Tests assert protection the depth-only design no longer has.** The tracker
   tests still feed clothing histograms (`hist=onehot(...)`), a path depth-only
   perception never produces, so the suite implies a bystander protection that is
   not live. Re-run those cases with `hist=None` or mark them as covering the
   dormant hook.
2. **Nothing tests `robot_follow.control_loop`.** The stdin/stop/EOF handling, the
   per-tick drive write, and the zero-twist-on-exit `finally` are the safety glue
   and need a fake-BBOS reader/writer harness.
3. **Zero-twist invariant has one hole:** if the `led.ctrl` writer fails to open
   after `drive.ctrl` succeeded, the drive writer is released without an explicit
   zero (harmless in practice — nothing was written and the daemon times out in
   0.1 s — but the invariant should be literally true).
4. **Dormant code:** `timestamp_to_seconds`, `band`, `score` (always 1.0), and the
   histogram hook (`hist_*` config, `hist_distance`) are unused by the depth-only
   path. Keep them only as documented hooks, or delete them with their tests.
5. **The CSV log is never flushed** (`/tmp` is tmpfs on the Jetson): a periodic
   `flush()` protects gate evidence if a session ends hard.
6. **Dashboard refusal reasons** ("Return to balance mode before following") are
   returned by the API but never shown; spec §7 asks the Follow card to show the
   last refusal reason.
7. **Commit hygiene at merge:** commit `e3fa11f` has its `Co-Authored-By` line
   inside the subject, and `ac49bd8` names a different Claude model. A squash
   merge removes both; an interactive reword fixes them if the history is kept.
