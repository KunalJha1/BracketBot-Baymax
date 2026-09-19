# Person-follow navigation — design

Status: design approved in conversation 2026-09-19; not implemented; no hardware evidence yet.
Revised 2026-09-19 after prototyping (see §11). Implementation plan:
`docs/superpowers/plans/2026-09-19-person-follow.md`.

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
| Target | A person. |
| Gap | Adjustable at runtime, 0.6–1.5 m, default 1.0 m. |
| Lock-on | The person who raises a hand (wrist above nose) for ~0.5 s after follow mode starts. |
| Obstacles | Stop-only corridor guard; no path planning. |
| Architecture | Everything closed-loop runs on the robot's Jetson; the dashboard only starts/stops, sets the gap, and sends a heartbeat. |
| Person comes closer than gap − 20 cm | Hold still. **No reversing in v1** (no rear sensing). |

## 4. Non-goals (v1)

- Keeping up with normal walking speed; changing the daemon speed clamp.
- Driving backward, steering around obstacles, map/SLAM-based planning.
- Following through doors/corners after the person leaves the camera's view.
- Re-identifying a person after a long absence (> 10 s) — they raise a hand again.
- Voice start/stop, sound cues (the speaker would be a second hardware writer).
- Arm motion of any kind during follow.
- Using facial-expression estimates for anything.

## 5. Architecture

```text
laptop: robot_dashboard.py ──SSH stdin (JSON lines: gap, heartbeat, stop)──┐
                          ◄──SSH stdout (FOLLOW_STATUS json, 5 Hz)──────┐  │
                                                                        │  ▼
robot (Jetson, bbos venv, files in /tmp):                          robot_follow.py (runner)
  camera.rect ───────► follow_perception: pose engine → persons ──┐
  camera.points ─────► follow_perception: torso range/bearing ────┤
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
| `scripts/follow_perception.py` | robot (decode also on laptop) | Pose engine wrapper (TensorRT via pycuda, imported lazily), pure pose-output decode + NMS, person → base-frame torso position from `camera.points`. | numpy; tensorrt/pycuda on robot only |
| `scripts/robot_follow.py` | robot | Runner: BBOS readers/writers, main loop, signals, pid file, stdin/stdout protocol, CSV log, zero-twist exit. Thin — no decisions. | bbos, the two modules above |
| `scripts/robot_dashboard.py` | laptop | Follow card, start/stop, gap slider, heartbeat, status parsing, resource exclusion, `--simulate` fake runner. | existing dashboard |
| `scripts/probe_follow.py` | robot | Read-only gate G0 probe: topic rates/fields, configs, self points, base x sign. | bbos |
| `scripts/check_follow_alignment.py` | laptop | Gate G0: overlays depth edges on the `camera.rect` frame. | vision extra |
| `scripts/build_pose_engine.sh` | robot | Gate G1: builds the TensorRT engine with `trtexec` and records versions. | JetPack |

The runner and both modules are deployed to `/tmp` by the existing `_deploy`
path, like `robot_base_mode.py`, and import each other by bare name (the
script's directory is on `sys.path`). The runner is a PEP 723 script launched
with `uv run --script`; its dependency block mirrors `bbapps/greeter/main.py`,
the app already proven to run TensorRT through PyCUDA on this robot.

### 5.2 Base-frame convention

Follow `bbapps/nav/main.py`: base **+y forward, +x lateral**, z up. Perception
converts points to robot-local **(forward, left, up)** once
(`forward = y`, `left = BASE_LEFT_SIGN · x`, default −1 = +x points right); all
of `follow_core` works in (forward, left). Bearing is `atan2(left, forward)`:
positive = person on the left, and positive ω turns left. The sign of x is
verified at G0 and the ω sign at G3.

## 6. Components

### 6.1 Perception (`follow_perception.py`)

- **Model:** committed `yolo11n-pose.pt` → ONNX (opset 17, done locally) →
  TensorRT FP16 engine **built on the Jetson** (`trtexec`), stored at
  `~/.cache/baymax/yolo11n-pose.engine` with a sidecar recording JetPack,
  TensorRT, input size, precision, and the checkpoint SHA-256. Never copied
  between machines (see `docs/yolo-robot-port.md`).
- **Input image:** `camera.rect` (rectified left eye), expected to be pixel-aligned
  with `camera.depth` up to a scale factor. Fallback if G0 shows it is not: the left
  half of `camera.head.rgb`, with each `camera.points` point projected into that raw
  fisheye eye (`cv2.fisheye.projectPoints` with the left intrinsics, the stereo
  rectification `R1`, and `Config("depth").camera_to_base_3x4`).
- **Decode:** output `(1, 56, 8400)` → boxes, confidence, 17 COCO keypoints
  (x, y, conf); confidence ≥ 0.40; NMS IoU 0.5. Pure numpy, unit-tested on
  synthetic tensors and cross-checked against Ultralytics' own postprocessing of
  the committed checkpoint (skipped when Ultralytics is not installed).
- **Person position:** collect `camera.points` whose `mask` pixel lies inside the
  torso rectangle (bounding rectangle of shoulders 5/6 and hips 11/12, at least 25%
  of the box wide; the middle third of the box if keypoints are low-confidence) and
  whose height is 0.2–2.0 m. Require ≥ 40 points; take the per-axis median →
  `(forward, left)`, `range = hypot`, `bearing` per 5.2.
  Depth and points frames must share a timestamp; the RGB frame used for detection
  must be within 50 ms of it, otherwise the observation is dropped.
- **Hand raised:** either wrist (9/10) with conf ≥ 0.5 is above the nose (0) in the
  image by at least 10% of the person's box height, nose conf ≥ 0.5.
- **Torso appearance:** 4×4×4 hue/saturation/value histogram (64 bins) of the torso
  rectangle, L1-normalised. Value is included because hue/saturation alone cannot
  tell black clothing from white.

Output per frame: `Perception(t, people, points)` with
`PersonObservation(forward, left, score, hand_raised, hist)` per located person and
the whole cloud in robot-local coordinates for the corridor check.

### 6.2 Tracker (`follow_core.Tracker`)

- **Odometry frame:** integrate robot motion from `drive.state.vel` (turns/s ×
  π·wheel_diam per wheel; wheel signs verified at G3/G4a).
  Person positions are transformed into this frame so that the robot's own motion is
  not mistaken for the person moving.
- **Lock-on (SEARCHING):** candidates within range 0.5–2.5 m and |bearing| ≤ 60°.
  A candidate with `hand_raised` in ≥ 80% of observations over a 0.5 s window,
  associated frame-to-frame by nearest position (≤ 0.3 m), is locked. If two
  candidates qualify at once, lock neither and keep waiting.
- **Tracking:** constant-velocity Kalman filter on (x, y, vx, vy) in odometry frame
  (measurement σ 0.08 m, acceleration σ 1 m/s², position σ capped at 1 m so the gate
  stays bounded while coasting; on LOST the velocity is zeroed and position σ grows
  by 0.5 m).
  Association gate: Mahalanobis distance ≤ 3σ **and** histogram Bhattacharyya
  distance ≤ 0.4 against a slowly updated reference (α = 0.05, updated only on
  unambiguous matches). If two observations fall inside the gate with scores within
  10% of each other, treat the frame as **ambiguous**: coast, don't update.
- Hand raises by other people while locked are ignored.
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
`quest_teleop/main.py`, `leader_follower_teleop.py`, `live_inference.py`); `drive.status.voltage` ≥ `Config("base").low_battery_v`;
pose engine loads and one warm-up inference completes.

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
| `meas_sigma` / `accel_sigma` / `max_pos_sigma` / `lost_pos_sigma` | 0.08 m / 1.0 m/s² / 1.0 m / 0.5 m | tracker |
| `lock_window` / `lock_fraction` | 0.5 s / 0.8 | |

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
FOLLOW_STATUS {"state":"FOLLOWING","range":1.07,"gap":1.0,"error":0.07,"bearing_deg":-4.2,"v":0.08,"w":-0.1,"blocked":false,"age_ms":60}
```

**Dashboard behaviour:**
- Follow card: Start/Stop toggle (key `F`), gap slider 0.6–1.5 m (step 0.1),
  status line (state, range, gap error), last reason for refusal/exit.
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
   - Lock-on: raised hand ≥ 0.5 s locks; 0.3 s does not; two simultaneous raisers
     lock neither; a bystander's raise while locked is ignored.
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
3. **`tests/test_follow_perception.py`** — pose decode on a stored ONNX Runtime
   output fixture from the committed checkpoint; torso-median range on a synthetic
   point cloud; timestamp-mismatch rejection.
4. **`tests/test_robot_dashboard.py`** additions — Follow start/stop in simulate
   mode, exclusion with Lean/gestures/LED actions, gap validation, heartbeat thread
   lifecycle, status parsing.

## 9. Robot rollout gates

Every gate that can move the robot has a person at the physical e-stop. Results
are recorded in `docs/robot-facts.md` with date and robot ID.

| Gate | Procedure | Pass criterion |
|---|---|---|
| G0 — read-only probe | Depth daemon running? `camera.depth`/`camera.points` rate; `camera.rect` shape and alignment with depth; base-frame x sign; wheel/`drive.state.vel` sign; drive clamp and command timeout from `Config("drive")`; self points (arms at home) inside the corridor. | All values recorded; self-mask defined. |
| G1 — pose engine | Build engine with `trtexec` on the Jetson; run decode on a real frame. | Warm latency ≤ 50 ms; person detected with keypoints. |
| G2 — dry run | `robot_follow.py --dry-run` (never opens `drive.ctrl`). Person stands at taped marks 0.6 / 1.0 / 1.5 m straight ahead and ±30°. | Range within ±5 cm of tape at every mark; lock-on works; bystander crossing doesn't steal the track. |
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
| `camera.rect` absent or misaligned with depth | Fisheye-eye fallback path (6.1); decided at G0. |
| Point-cloud calibration drift | G2 tape check after every reflash. |
| Braking/acceleration pitch shakes the camera | Depth daemon compensates with IMU pitch; conservative ramps; check at G4. |
| Robot's own arms/wheels in the depth cloud | Self-mask from G0; arms must be idle and at home. |
| Another app writes `drive.ctrl` | Start precondition process check; dashboard exclusivity. |
| TensorRT/pycuda API differences on the installed JetPack | Resolved at G1 before any motion work. |
| Poor lighting, back-lit person, loose clothing hiding keypoints | Box-based torso fallback; LOST handling; out-of-scope beyond that. |

## 11. Revisions after prototyping (2026-09-19)

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
