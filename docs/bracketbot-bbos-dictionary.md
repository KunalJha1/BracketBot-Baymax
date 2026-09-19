# BracketBot / BBOS reverse-engineering dictionary

**Research snapshot:** 2026-09-19<br>
**Purpose:** a practical field guide to the public BracketBot history, the newer `bbos` runtime, surviving examples, 2026 hackathon projects, recurring mistakes, and the highest-value things to build next.

> This is a synthesis of public repositories and preserved forks, not official current documentation. The central `BracketBotOS` source and the old `docs.bracket.bot` site are not publicly available at the time of this audit. Treat every numeric limit, field layout, and hardware behavior as version- and robot-specific until checked against the installed daemon source and `Config(...)` on the target robot.

## 1. The short answer

The import the user remembered is almost certainly:

```python
from bbos import Reader, Writer, Type, Config
```

I found **no** `RequestType`, `Request Type`, or `request_type` symbol in the old repos, surviving BBOS app forks, copied examples, or the ten 2026 projects. `Type("drive_ctrl")` and similar calls are probably what was remembered as “request type.”

BBOS is best understood as a typed, shared-memory topic bus:

```text
hardware daemon -> Writer("sensor.topic", Type("schema")) -> Reader(...) -> app
app -> Writer("actuator.ctrl", Type("schema")) -> Reader(...) -> hardware daemon
```

The most important operating rules are:

1. A topic has **one writer and many readers**. A second live writer raises a `RuntimeError` identifying the owner PID.
2. Use `with writer.buf() as b:` for a complete, atomic frame. Do not assume `Writer` and `Reader` are symmetric.
3. `Reader.ready()` means a fresh sample is available; then read `reader.data[...]`.
4. Default `keeptime=True` can pace a loop to the schema period. Adding another `sleep()` can make control too slow.
5. Base velocity commands expire quickly. Publish continuously and explicitly command zero on exit.
6. Closing an arm control writer makes the arm daemon disable torque. A held object can fall.
7. Arm topic positions are motor turns, not URDF radians. Convert with the correct arm's `Config` methods.
8. Configuration and calibration are part of the API. Never copy one robot's calibration onto another.
9. A completed command is not proof of a completed physical task. Verify possession, placement, freshness, and frame/epoch.
10. The best reusable pattern is one hardware-owning process with queues/adapters behind it—not a new writer in every UI handler, tool call, or script.

## 2. Provenance and confidence labels

This guide uses these labels:

| Label | Meaning |
| --- | --- |
| **Observed** | Present in more than one public BBOS example or hardware project. |
| **Source-checked report** | A project says it checked deployed BBOS daemon source, but that source is no longer public. |
| **Legacy** | From the 2024–2025 capstone stack; it may not apply to the 2026 BBOS platform. |
| **Project-reported** | A team observed it on one robot/version; useful evidence, not a universal guarantee. |
| **Inferred** | Strongly suggested by examples, but the core implementation could not be inspected. |

The strongest sources recovered were:

- the archived [BracketBotCapstone organization](https://github.com/BracketBotCapstone), [quickstart](https://github.com/BracketBotCapstone/quickstart), website design logs, and [MCP bridge](https://github.com/BracketBotCapstone/bracketbot-mcp);
- surviving public forks of `BracketBotApps` and `BracketBotAI`;
- a later [Jetson setup gist](https://gist.github.com/raghavauppuluri13/62eb0db4c162e6fa77a654fcce17609b) that names `BracketBotOS`, `BracketBotApps`, and `BracketBotAI`;
- a [stereo calibration recovery gist](https://gist.github.com/raghavauppuluri13/82cb051f093d21a6502fe6402e207201);
- 2026 projects that copied BBOS examples or documented direct inspection of a deployed robot.

## 3. BracketBot lineage

### 3.1 2024–2025 capstone generation

The original public organization now explicitly identifies itself as an archived university capstone. Its six visible repositories are:

| Repository | What matters |
| --- | --- |
| [`quickstart`](https://github.com/BracketBotCapstone/quickstart) | Original setup and direct-hardware Python examples. The README points to a docs site that is now unavailable. |
| [`bracketbot-mcp`](https://github.com/BracketBotCapstone/bracketbot-mcp) | MCP tools that proxy local FastAPI robot servers. Useful product idea; weak safety/auth model. |
| [`BracketBotCapstone.github.io`](https://github.com/BracketBotCapstone/BracketBotCapstone.github.io) | Architecture, hardware, control, mapping, and build design logs. |
| [`.github`](https://github.com/BracketBotCapstone/.github) | Organization profile. |
| [`dora`](https://github.com/BracketBotCapstone/dora) | Upstream fork, not original BracketBot application logic. |
| [`rules_ros2`](https://github.com/BracketBotCapstone/rules_ros2) | Upstream fork, not original BracketBot application logic. |

The legacy architecture used small Python packages/nodes, MQTT on the robot, generated message classes, and Rerun visualization. Design logs describe LQR balance/control, localization, mapping, IMU and RealSense camera packages. The mapping work referenced PyWaveMap; localization evolved from an early particle-filter description toward an on-manifold EKF description.

### 3.2 Successor BBOS generation

Later public traces name a successor organization and three repositories:

- `BracketBotOS`: the local platform/runtime and `bbos` Python package;
- `BracketBotApps`: teleop, navigation, Quest, voice, calibration, inference, and diagnostic examples;
- `BracketBotAI`: higher-level AI experiments.

The original successor repos are no longer generally accessible, but public forks and copied example trees survive. Installation snippets consistently point `uv` at a local editable path rather than PyPI:

```toml
# /// script
# dependencies = ["bbos", "numpy"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
```

Some earlier copies use `/home/bracketbot/BracketBotOS`. Therefore:

- `bbos` is not a normal public PyPI dependency;
- the path differs across robot images;
- inspect the actual robot before copying a PEP 723 block;
- `uv run your_app.py` is the common execution pattern.

## 4. BBOS core dictionary

### `Writer(topic, Type(schema), keeptime=True)`

Creates the exclusive producer for a topic.

```python
import numpy as np
from bbos import Writer, Type

with Writer("drive.ctrl", Type("drive_ctrl")) as drive:
    with drive.buf() as frame:
        frame["twist"] = np.array([0.05, 0.0], dtype=np.float32)
```

Observed behaviors:

- `writer.buf()` yields a complete writable frame and commits it on context exit.
- Direct `writer["field"] = value` appears in projects, but `buf()` is the safer pattern for multi-field messages.
- `Writer` does not expose the same `.data` interface as `Reader`.
- A second writer for the same live topic fails rather than silently arbitrating.
- `writer.ready()` appears in examples as a readiness/backpressure check before some writes.
- Default timing may be enforced when the buffer context exits.
- Use `keeptime=False` for write-once/event channels or when one explicit control clock owns pacing.

### `Reader(topic, Type(schema)=optional, keeptime=True)`

Attaches to a topic without taking ownership.

```python
from bbos import Reader

with Reader("camera.head.jpeg", keeptime=False) as camera:
    while True:
        if camera.ready():
            n = int(camera.data["jpeg_len"])
            encoded = bytes(camera.data["jpeg"][:n])
```

Observed behaviors:

- Call `ready()` before consuming a new sample.
- Read fields through `reader.data["field"]`.
- A source-checked guide reports that `ready()` handles writer death and recreation.
- `aligned_to=other_reader` appears in synchronized camera/depth usage.
- `keeptime=False` is common for polling in an independently paced loop.
- One team reported that the first `Reader` created in a process could remain unready while a newly constructed reader worked. Treat this as a version-specific defect to diagnose, not desired API behavior.

### `Type("schema_name")`

Selects a registered shared-memory schema. It is not a request class.

Known names include:

```text
drive_ctrl       arm_state       arm_ctrl        arm_torque
arm_target       led_ctrl        speaker_audio   dataset_flag
quest_controllers quest_haptic   quest_joystick  quest_link
```

Inferred responsibilities of a type registration:

- field names, shapes, and dtypes;
- memory layout;
- nominal/realtime period;
- perhaps timestamp/state metadata.

Do not invent a type name because a topic exists. When examples omit `Type(...)` on a `Reader`, the reader can apparently discover the topic's registered schema.

### `Config("daemon")`

Loads registered configuration for a daemon or hardware subsystem.

Observed configs include:

```text
arm_left arm_right base drive cam_head cam_left cam_right depth
mic speaker quest usb
```

Observed uses:

- sample rate, channel count, and chunk size;
- wheel diameter and robot width;
- transforms such as `T_base_cam`;
- arm DOF, home pose, joint names, URDF path, limits, low-pass settings;
- `q2urdf(...)`, `urdf2q(...)`, and an IK object;
- depth and camera parameters.

Rule: read configuration from the installed robot. Numeric values in examples are diagnostics, not an API contract.

### Supporting modules

| Symbol | Observed purpose |
| --- | --- |
| `bbos.time.Loop` | Periodic-loop timing. |
| `bbos.time.Realtime` | Realtime scheduling/timing helper. |
| `bbos.tf.rot`, `trans`, `rmat_to_quat` | Transform helpers. |
| `bbos.app_manager.start_app`, `stop_app`, `get_status` | App lifecycle. |
| `bbos.functional.curry` | Functional helper used by IK code. |
| `bbos.register`, `realtime`, `state` | Type/config registration decorators or helpers seen in preserved daemon examples. |

## 5. Timing, ownership, and lifecycle

### 5.1 One writer means one hardware owner

Treat these as scarce leases:

```text
drive.ctrl
arm_left.ctrl       arm_left.torque
arm_right.ctrl      arm_right.torque
speaker.audio
led.ctrl
dataset.flag
```

Do not let an HTTP request, MCP tool, LLM tool, UI widget, and teleop process each open their own writer. Put one owner behind a queue:

```text
UI / voice / MCP / agent
          |
          v
 command arbiter + cancellation + deadman
          |
          v
 one BBOS hardware-owner process
```

This also solves priority: E-stop > manual teleop > supervised autonomy > idle.

### 5.2 `keeptime` is easy to misuse

A source-checked project reports:

- `Writer(..., keeptime=True)` is the default;
- `buf()` paces to the type's declared period;
- an extra `sleep()` stacks on top of that;
- timing state can be shared per process, so multiple paced writers can interact.

Practical rule:

- select one control clock;
- make other event/side-channel writers unpaced;
- measure actual publish frequency;
- never assume a loop is 50 Hz because the code says `sleep(0.02)`.

### 5.3 Startup and shutdown are state transitions

For arms, a robust startup pattern recovered from `homing.py` and later projects is:

1. Open state reader and control/torque writers only after ownership is confirmed.
2. Disable torque.
3. Wait for a fresh state sample.
4. Copy the live pose into the control command.
5. Flush that command briefly.
6. Enable torque.
7. Allow the daemon's mode transition to settle.
8. Begin a bounded trajectory.

This avoids enabling against a stale setpoint and causing a jump.

Shutdown depends on whether a load is held. Closing the writer or cutting torque is an emergency behavior, not a load-preserving stop. A safe loaded stop must keep the control owner alive, hold a verified pose, report its state, and wait for a deliberate release/park plan.

## 6. Topic and payload dictionary

The following table combines preserved examples and source-checked project notes. “Fields” means observed access, not an exhaustive schema.

| Topic | Type / fields observed | Meaning and cautions |
| --- | --- | --- |
| `drive.ctrl` | `drive_ctrl`; `twist[2]`, sometimes `twist_torque[2]` | `[linear m/s, yaw rad/s]`; publish continuously. |
| `drive.state` | `vel[2]`, `ctrl[2]`, `iq[2]` | Wheel feedback. `vel` was treated as motor turns/s in teleop/delivery code. |
| `base.mode` | `mode`, `lean_angle_deg` | Balance/lean/special mode. Do not confuse `twist` field with a mode named “twist.” |
| `arm_left.state`, `arm_right.state` | `arm_state`; `pos`, `vel`, `torque`, `temp`, `current`, each 8-wide in newer examples | State in motor-side units; position is motor turns. |
| `arm_*.ctrl` | `arm_ctrl`; `pos`, `vel`, `tau`, `alpha` | Position/velocity/feedforward command. Smooth and range-check before writing. |
| `arm_*.torque` | `arm_torque`; `enable`, `tau_mode`, `compliance_mode`, `axis_aligned`, `force_only`, `j0_homing`, `calibrating` | Mode and torque enable. Do not toggle casually under load. |
| `arm_*.target` | `arm_target`; `xyz`, `quat`, `grip`, `tracking` | Teleop/dataset target record. One guide reports the arm daemon itself does not consume it. |
| `camera.head.jpeg`, `camera.left.jpeg`, `camera.right.jpeg` | `jpeg`, `jpeg_len` | Encoded JPEG. Older code may use `bytesused`; check schema. |
| `camera.rect` | commonly `rgb` | Rectified stereo image. Pair with aligned point/depth data when possible. |
| `camera.depth` | `depth`, `timestamp` | Depth image; examples treat values as millimetres before conversion. |
| `camera.points` | `points`, `mask`, `num_points`, `timestamp` | Base-frame point cloud with pixel mask/index correspondence. Verify alignment empirically. |
| `camera.*.status` | version-specific | Camera health/status topic seen in newer projects. |
| `imu.orientation` | `rpy` | Newer example says order became roll, pitch, yaw on 2026-08-15; older daemon returned pitch, roll, yaw under the same field. Schema drift warning. |
| `imu.raw` | `accel`, `gyro` | Raw inertial arrays. |
| `slam.pose` | `pos`, `quat` | Base pose in a SLAM/map-related frame. Transform explicitly before planning. |
| `mapping.grid2d` | `grid`, `origin` | Traversability grid. A newer navigator treats `1` as floor and `2` as obstacle. Verify installed semantics. |
| `mapping.voxels` | `coords`, `colors`, `num_voxels` | Live voxel map. |
| `mapping.reproject` | `reprojecting` | Map rebuild in progress. |
| `mapping.rebuild` | `count`, `frame`, `num_moved`, `num_emptied`, `num_filled`, `timestamp` | Rebuild completion diagnostics. |
| `speaker.audio` | `speaker_audio`; `audio` | PCM16 frames. One deployed schema used mono 16 kHz `(1600, 1)` chunks every 100 ms; always read `Config("speaker")`. |
| `mic.audio` | `audio` | Microphone samples. Read `Config("mic")` for shape/rate. |
| `mic.ref_level` | `dbfs` | Reference signal level. |
| `led.ctrl` | `led_ctrl`; `rgb[3]`, `brightness`, `period_ms` | LED output. Some apps refresh it continually. |
| `led.state` | `rgb` | LED state feedback. |
| `wakeword.state` | `active` | Wakeword detection. Implement edge detection to avoid repeated triggers. |
| `quest.controllers` | `quest_controllers`; transforms, buttons, triggers, sticks | Quest controller input. |
| `quest.haptic` | `quest_haptic`; `hand`, `frequency`, `amplitude`, `duration` | Controller haptics. |
| `quest.link` | `connected` | Quest connection state. |
| `quest.joystick` | `left`, `right` | Joystick values. |
| `dataset.flag` | `dataset_flag`; `prefix`, `name`, `text`, `toggle_episode`, `drop_episode` | Dataset episode control. Exclusive writer. |
| `usb.tree` | `json`, `shape_hash` | USB topology snapshot. |
| `leader_left.state`, `leader_right.state` | `pos` | Leader-arm state in follower teleop. |

Legacy topics seen in older successor forks include `camera.jpeg`, `localizer.pose`, `mapping.voxels`, `speakerphone.speaker`, `speakerphone.mic`, `transcript`, `so101.state`, `so101.ctrl`, `so101.torque`, `imu.data`, and `led_strip.ctrl`. Their names should not be mixed with the newer schema without an adapter.

## 7. Hardware facts and units recovered from deployed examples

These were source-checked on one 2026 robot and are useful for understanding the system, but must be queried again on the target:

### Base

| Item | Reported value |
| --- | --- |
| `drive.ctrl` period | 10 ms |
| Expected host command rate | 50 Hz |
| Command timeout | 0.1 s |
| Linear clamp | 0.3 m/s |
| Yaw clamp | 1.0 rad/s; one drive config exposed 0.9 rad/s |
| Robot width | 0.3275 m |
| Wheel diameter | 0.165 m |

The daemon reportedly clamps finite commands and zeros on missing, stale, unreadable, or non-finite commands. Applications should still validate inputs and send an explicit zero in `finally`.

### Arms

- Two 8-DOF sides are addressed as `arm_left` and `arm_right`.
- J0 is a vertical lift stage, not a revolute shoulder.
- J7 is the gripper.
- State/control topic positions are motor turns.
- `Config("arm_left").q2urdf(...)` and `.urdf2q(...)` convert to/from URDF space.
- The two arms are not mirrored by one sign flip. J0, joint signs, and gripper sign differ.
- Per-robot calibration was stored under the installed BBOS tree as `ranges.calibration.json`.
- One deployed config used a 0.0465 m J0 wheel radius.
- Control/state examples used a 15 ms topic period; the daemon reportedly ran its inner loop faster.
- The daemon clipped a new target around the live pose and then to calibrated range. That is a backstop, not trajectory planning.

Coordinate convention reported by the newer examples:

```text
base frame: +x forward, +y left, +z up
quaternion convention in action examples: XYZW
```

Always annotate frames and quaternion ordering in interfaces. Several simulation projects used internal WXYZ while exposing XYZW publicly.

## 8. Minimal patterns worth copying

### Read-only probe

```python
import time
from bbos import Reader

with Reader("arm_left.state", keeptime=False) as state:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if state.ready():
            print(state.data["pos"].copy())
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("No fresh arm_left.state sample")
```

Read-only tools should open no writers. That property should be testable.

### Base command with explicit stop

```python
import numpy as np
from bbos import Writer, Type

with Writer("drive.ctrl", Type("drive_ctrl")) as drive:
    try:
        for _ in range(50):
            with drive.buf() as b:
                b["twist"] = np.array([0.05, 0.0], np.float32)
    finally:
        with drive.buf() as b:
            b["twist"] = np.zeros(2, np.float32)
```

This is only a transport example. Real motion also needs ownership, obstacle checks, cancellation, localization freshness, and an operator-approved test area.

### Audio framing

```python
import numpy as np
from bbos import Config, Writer, Type

cfg = Config("speaker")
silence = np.zeros((cfg.chunk_size, cfg.channels), dtype=np.int16)
with Writer("speaker.audio", Type("speaker_audio")) as speaker:
    with speaker.buf() as b:
        b["audio"] = silence
```

Do not send WAV/MP3 container bytes into the PCM field. Decode and resample first; the daemon may perform a separate hardware-rate conversion.

### IK conversion sequence

```python
live_motor = state.data["pos"].copy()
q = cfg.q2urdf(live_motor)
cfg.ik.reset(list(q[:7]))
solution = cfg.ik.solve(list(position_xyz), list(quaternion_xyzw))
if solution is None:
    raise RuntimeError("unreachable")
q[:7] = solution[:7]
target_motor = cfg.urdf2q(q)
target_motor[7] = live_motor[7]
```

Then validate the complete path, calibration margin, collision/workspace constraints, speed, cancellation, and arrival. Never step directly to the result.

## 9. Safety and correctness checklist

Before any motion:

- identify the robot image/version and BBOS path;
- inspect the installed schema/config/daemon source if permitted;
- list processes that may own the required topics;
- run a read-only health check first;
- confirm calibration belongs to this robot;
- confirm frames, units, quaternion ordering, and sign conventions;
- require an explicit `--execute` or equivalent gate;
- validate the entire path, not only the endpoint;
- enforce workspace, rate, acceleration, collision, freshness, and timeout limits;
- keep the physical E-stop reachable;
- define exactly what happens on Ctrl-C, exception, disconnect, stale sensor, and writer loss;
- do not test a loaded-arm shutdown over people, feet, boards, or fragile objects.

After every action:

- compare requested and measured state;
- obtain new task evidence after motion completes;
- distinguish “command accepted,” “motion arrived,” “object held,” and “task verified”;
- preserve timestamps and frame/map epoch;
- stop after bounded retries and ask for help.

## 10. Recurring BBOS failure modes

| Failure | Why it happens | Better design |
| --- | --- | --- |
| `RuntimeError` opening a writer | Another process owns the topic. | Inspect ownership; route commands through the designated owner. |
| Control loop runs at half speed | Automatic pacing plus manual sleeping, or multiple paced writers. | Use one measured clock and unpaced side channels. |
| Arm jumps when torque enables | Stale command buffer. | Disable, sample live pose, seed/flush command, then enable. |
| Arm goes limp on script exit | Daemon deadman sees the control writer disappear. | Park deliberately; never equate writer close with load-preserving stop. |
| Command appears to do nothing | Stale target, no fresh writer/reader state, clipping, or a daemon mode transition. | Verify measured convergence and retry from measured state, not the old intended target. |
| Camera decodes garbage | Assumed length/field name or copied full fixed buffer. | Use the installed `jpeg_len`/equivalent field. |
| Depth and RGB disagree | Different frames/timestamps or changed daemon intrinsics. | Align readers, compare timestamps, and validate correspondence live. |
| Robot drives briefly then stops | Base deadman expires. | Publish continuously above the timeout frequency. |
| Robot keeps driving on UI loss | App lacks its own deadman. | Add server-side command age, disconnect stop, and explicit zero. |
| IK succeeds inside the chassis/table | Joint reachability is not collision-free workspace. | Chassis exclusion, full-path collision, surveyed environment. |
| “Successful” pick holds nothing | Completion inferred from commands or idealized simulation attachment. | Use current/position/contact/vision evidence and post-action rescan. |
| Map treats blank space as free | Missing/omitted observations collapsed into free space. | Preserve unknown as unknown and report coverage. |
| Left-arm command reused on right | Mirroring assumed to be one sign change. | Use each arm's own config and conversion functions. |

## 11. Audit of the original capstone work

### What it did well

- Modular mechanical design using extrusion, clamps, printable mounts, commodity hoverboard motors, ODrive, Pi-class compute, and an E-stop.
- Clear subsystem decomposition: control, localization, planning, mapping, IMU, camera, messages, and visualization.
- Reproducible examples covering drive, camera/depth, localization, segmentation, YOLO, Whisper, TTS, realtime voice, and Rerun.
- A physical calibration step and a quick keyboard-drive smoke test.
- A clever MCP proof of concept for multi-robot and AI control.
- Published CAD and a prebuilt ARM wheel for a difficult visualization dependency.

### What it did poorly or dangerously

- The docs link now fails, leaving the code as the documentation.
- The setup script was highly invasive: apt changes, shell modifications, global venv activation, group changes, SSH enablement, downloaded installers, service creation, firewall changes, and reboot.
- It used curl-to-shell installers and patched installed Python package source with `sudo sed`.
- Many dependencies were unpinned.
- The setup's confirmation default proceeded automatically after a short timeout.
- Mosquitto exposed an anonymous WebSocket listener and opened its firewall port.
- The shell prompt ran Python/hardware inspection on every prompt.
- The quickstart examples were monolithic direct-device scripts rather than a stable SDK contract.
- Calibration state could be lost on reflash; an open PR specifically requested separate motor direction calibration.
- MCP movement used local HTTP GET-style action endpoints, hardcoded ports, and no visible authentication, lease, cancellation receipt, or stop-on-client-loss contract.

### What to capitalize on

Keep the approachable examples, cheap modular hardware, Rerun visibility, and AI-tool interface. Replace the installer and control boundary with:

- idempotent versioned provisioning;
- pinned dependencies and image manifests;
- authenticated localhost-by-default services;
- capability discovery instead of port assumptions;
- a lease/arbiter for motion ownership;
- deadman and cancellation semantics at every network boundary;
- calibration backup/restore with robot identity and schema version;
- automated post-flash health checks.

## 12. Audit of the ten 2026 projects

Counts below are repository snapshots, not quality scores. “BBOS files” means Python files importing `bbos`; copied reference examples can inflate it.

| Project | Concept | Python / test / BBOS files | What it did well | Main limitation | Reusable opportunity |
| --- | --- | ---: | --- | --- | --- |
| [Crafter](https://github.com/CalvinDobbs/crafter) | Minecraft blueprint to physical box build | 106 / 13 / 35 | Best operational contracts: truthful preflight, world-model freshness/epochs, unknown-not-free mapping, cancellation/possession distinctions, deep BBOS notes, offline fake-BBOS tests. | Live physical action provider was intentionally incomplete; many BBOS files are copied references. | Use its action/provider contracts, evidence model, bounded JSON, source-checked motion/audio guides, and fake-BBOS test strategy. |
| [Butler Bot](https://github.com/EliteAtlantico/Butler-Bot-) | Household fetch/tidy robot in MuJoCo | 95 / 29 / 0 | Strongest simulation evaluation: deterministic benchmarks, CI, watchdog, phone takeover/E-stop, rich failures, mapping and grasp checks. | No BBOS/hardware integration; simulator ground truth and welded/idealized assumptions can hide deployment problems. | Port its arbiter, benchmarks, tool-result explanations, duplicate-failure guard, and manual takeover—not its sim state assumptions. |
| [Gambit](https://github.com/GautamM303/-Gambit-) | Physical chess opponent | 42 / 7 / 17 | Excellent honesty: hardware-unverified label, dry-run default, camera verification before state advance, full-path IK before motion, retry/help path, calibration checklist. | Hardware backend had not been executed; base channel/reach mismatch; torque-off “E-stop” can drop/sag a load. | Reuse its dry-run/calibrate/check/execute progression and physical-vs-logical state invariant. |
| [LegoBot](https://github.com/chhabra-anirudh/legobot) | Prompt/image to block structure | 40 / 9 / 6 | Real BBOS arm bridge, measured reach maps, calibration margins, honest idealized-attachment labels, deterministic compiler checks, excellent handoff notes. | No possession signal, live vision, full-arm collision, or verified real build; one robot-specific Reader quirk workaround. | Highest-value manipulation work: add current-plus-vision grasp evidence and turn the bridge into a reusable trajectory executor. |
| [BracketBot Waiter Delivery](https://github.com/SK3720/BracketBot-Project) | Voice-triggered tray delivery | 35 / 2 / 31 | Most direct real-robot workflow: dual-arm holding, ArUco approach, odometry backup, speech/audio, measured convergence retries, practical field tuning. | Large monolithic script, hardcoded poses/paths/tags/thresholds, hand-copied deployment, very little automated testing, subprocess ownership complexity. | Extract its proven hardware primitives into services and replayable integration tests; preserve the empirical lessons, discard the monolith. |
| [Mr. Clean / RL-BOT](https://github.com/Karan-Gupta07/Mr.-Clean-Bracketbot-Cleaning-Robot) | Room cleaning in MuJoCo | 80 / 13 / 0 | Full simulated mission, sensors at realistic rates, inflated A*, replanning, multiple manipulation policies, offline planner option. | No real sensors or BBOS; station dispatch and simulator state constrain generality. | Use the mission/evaluation harness to test the real adapter and compare classical, learned, and agent planners. |
| [SoleMates](https://github.com/selvxhini-10/SoleMates) | Sock pairing and sorting | 31 / 7 / 1 | Deterministic classical baseline, global matching instead of greedy pairing, calibration checks, guarded hardware backend that refuses to fake support. | No motor control, cloth physics, collision-aware planning, or visual post-action verification. | Reuse confidence-aware matching and human-confirmation path as a task layer above a real BBOS action provider. |
| [AOT BracketBot](https://github.com/rodeanmoradi/bots2026) | Adaptive action-observation therapy | 27 / 0 / 0 | Clear ROS 2/MoveIt architecture, normalized movement scoring, adaptive coaching loop. | No BBOS, no tests counted, sim/visualization first; medical/therapy claims require stronger validation. | Build a BBOS-to-ROS bridge and preserve the scoring/adaptation engine as a separate, testable layer. |
| [Educational Pong](https://github.com/JinchengLikesPlanes/UTWAT_Hachathon_2026_BracketBot) | Browser lessons for balance, vision, and RL | 37 / 9 / 0 | Excellent teaching progression, reproducible failure modes, seeded tests, connects PID→vision→policy conceptually. | Browser/MuJoCo results do not deploy to BBOS; simplified dynamics/policy. | Turn it into a BBOS-backed sandbox/replay viewer with logged robot data and strict no-motion classroom mode. |
| [BigFluffyRobot](https://github.com/cchryx/BigFluffyRobot) | Browser-driven boxing/puppeteering simulator | 11 / 3 / 0 | Strong lightweight teleop: a 300 ms deadman, one in-flight request, reachable-envelope clamping, proportional retargeting, record/replay, headless HTTP tests, vendored browser vision assets, and a clean driver abstraction. | `BracketDriver` was explicitly unimplemented; no BBOS or physics/contact/balance integration. The root README documents only the bundled URDF, while the actual project story is split across `PLAN.md` and `TELEOP.md`. | Port the tested driver boundary and deadman to the central BBOS owner; reuse its browser-side perception and deterministic replay, with stricter acceleration/balance limits. |

## 13. Cross-project lessons

### The best ideas to copy

1. **Truthful capability gates.** Crafter and SoleMates refuse hardware execution when required evidence/providers are missing.
2. **Dry run first.** Gambit and LegoBot separate planning/IK validation from writer creation and motion.
3. **One verifiable owner.** Newer BBOS examples reveal why shared writer ownership must be centralized.
4. **Manual takeover and deadman.** Butler Bot keeps a phone operator in the loop and applies a server watchdog.
5. **Measure, do not narrate.** Butler Bot commits benchmark results; LegoBot separates idealized attachment from contact evidence; Gambit verifies the board after every robot move.
6. **Deterministic fallbacks.** Several projects can run without a paid model, which makes CI and debugging possible.
7. **Unknown stays unknown.** Crafter does not convert missing map cells, depth, possession, or stale observations into confident facts.
8. **Bounded retry with reasons.** Butler Bot prevents repeated identical failures; other projects ask for help rather than looping forever.
9. **Artifacts and handoffs.** Status documents, calibration files, run JSON, traces, Rerun recordings, and benchmark baselines preserve what actually happened.

### The repeated bad patterns

- polished simulation described too close to physical capability;
- endpoint IK used as a substitute for full-path collision checking;
- hardcoded robot poses and thresholds embedded in giant scripts;
- `sleep()`-driven orchestration without measured state transitions;
- command completion treated as task completion;
- no possession or placement evidence;
- no frame/epoch/freshness metadata;
- direct BBOS writers scattered through UI, agent, audio, and action code;
- weak or absent tests around actual BBOS adapter behavior;
- copying a shared `~/bbapps` deployment by hand;
- credentials, calibration, code, and service state managed separately with no manifest;
- emergency behavior that stops software but may mechanically drop a load.

## 14. What to build to capitalize on the gaps

### Priority 0: a public, versioned BBOS contract

Create a repository that can be generated from the installed runtime and contains:

- package version and robot image version;
- all topic names, type names, field dtypes/shapes, and declared periods;
- all config keys with units;
- ownership/deadman semantics;
- frame tree and quaternion convention;
- compatibility notes and schema changelog;
- tiny read-only examples plus separately gated motion examples;
- a simulator/fake implementation of the same interface.

This is the biggest leverage point because every project reverse-engineered the same facts.

### Priority 1: a hardware-owner and command arbiter

One process should own BBOS writers and expose typed commands:

```text
submit -> admitted/rejected -> running -> holding/completed/failed/cancelled
```

Each action receipt should include:

- immutable command ID;
- accepted parameters and frame/epoch;
- owner/priority/lease expiry;
- start and terminal timestamps;
- latest heartbeat separately from terminal time;
- measured arrival/error;
- task evidence and its observation time;
- cancellation/stop result;
- reason codes.

The arbiter must support manual preemption and a load-preserving hold state.

### Priority 2: a calibrated action library

Promote proven fragments into tested primitives:

- `drive_velocity`, `drive_distance`, `rotate`, `stop`;
- `enable_from_live_pose`, `home`, `park`, `hold`;
- `move_joint_path`, `move_cartesian_path`;
- `open_gripper`, `close_until_contact`, `release`;
- `play_pcm`, `set_led`;
- camera/depth synchronized capture.

Every primitive should run against fake BBOS in CI and have a hardware acceptance test that requires an explicit operator gate.

### Priority 3: physical evidence

Build reusable estimators for:

- gripper contact from J7 position, motor current, and commanded position;
- slip during carry;
- object disappearance from source and appearance at destination;
- post-place support/stability;
- localization/map epoch consistency;
- sensor and command freshness.

No task-level `success=True` should exist without named evidence.

### Priority 4: deployment and recovery

- versioned robot app bundles instead of hand-copying into `~/bbapps`;
- read-only inventory command for daemon/app/topic owners;
- calibration backup tied to robot serial and schema hash;
- safe update with rollback;
- post-flash camera, IMU, drive, arm, audio, and disk/network tests;
- structured logs and downloadable run bundles;
- secret storage outside repositories with explicit permissions.

### Priority 5: secure AI and remote control

Modernize the old MCP idea with:

- authentication and authorization;
- localhost-only default binding;
- capability discovery;
- writer lease and command expiry;
- rate/velocity/workspace bounds below the model layer;
- human approval for hazardous capabilities;
- stop on disconnect;
- auditable tool receipts;
- camera privacy controls;
- no GET request that causes movement.

## 15. Recommended software shape

```text
                           read-only observations
 cameras / IMU / map / state ------------------------------+
                                                            v
 UI / voice / MCP / planner -> intent -> safety supervisor -> action state machine
                                         |                  |
                                         | reject/reason    v
                                         +------------ command arbiter
                                                            |
                                      one writer per topic  v
                                                          BBOS
                                                            |
                                             measured state/evidence
                                                            v
                                                 verifier + run log
```

Separation of concerns:

- **Planner:** proposes intent, never writes hardware topics.
- **Safety supervisor:** validates capability, world freshness, frames, limits, and conflicts.
- **Action state machine:** owns bounded transitions and cancellation behavior.
- **Command arbiter:** owns BBOS writers and priorities.
- **Verifier:** decides task truth from fresh observations, not from command return codes.
- **Recorder:** stores schemas, configs, calibration hashes, commands, state, and evidence.

## 16. Definition of done for a real BracketBot behavior

A feature is not “working on BracketBot” until all of these are true:

- runs with the installed BBOS version and records that version;
- has a read-only preflight;
- has an explicit hardware execution gate;
- respects single-writer ownership;
- specifies units, frames, signs, and timing;
- defines startup, normal stop, cancel, exception, and disconnect behavior;
- validates a full trajectory and environmental constraints;
- measures arrival from state feedback;
- verifies task outcome from fresh physical evidence;
- has bounded retry and human-help behavior;
- has fake-BBOS unit/integration tests;
- has a documented supervised hardware test;
- produces a run artifact detailed enough to reproduce a failure;
- never claims simulation evidence as hardware evidence.

## 17. Source catalog

### Reuse and licensing warning

The surviving `BracketBotApps` fork includes an MIT license and the surviving `BracketBotAI` fork includes GPL-3.0. The ten audited 2026 repositories and the four original capstone source repositories did **not** contain a license file in the checked snapshots. Public visibility is not permission to copy: learn from their architecture, link to them, and seek permission before incorporating unlicensed code or assets. Also inspect vendored models, meshes, media, and upstream forks separately because their licenses can differ from the enclosing repository.

### Historical and platform sources

- [BracketBotCapstone organization](https://github.com/BracketBotCapstone)
- [Legacy quickstart](https://github.com/BracketBotCapstone/quickstart)
- [Legacy MCP bridge](https://github.com/BracketBotCapstone/bracketbot-mcp)
- [Legacy website/design logs](https://github.com/BracketBotCapstone/BracketBotCapstone.github.io)
- [Current BracketBot site](https://bracketbot.com)
- [Surviving BracketBotApps fork](https://github.com/fayaz-rafin/BracketBotApps)
- [Surviving BracketBotAI fork](https://github.com/sharisseji/BracketBotAI)
- [Jetson/BBOS setup trace](https://gist.github.com/raghavauppuluri13/62eb0db4c162e6fa77a654fcce17609b)
- [Stereo calibration recovery trace](https://gist.github.com/raghavauppuluri13/82cb051f093d21a6502fe6402e207201)

### 2026 project sources

- [Crafter](https://github.com/CalvinDobbs/crafter)
- [Butler Bot](https://github.com/EliteAtlantico/Butler-Bot-)
- [SoleMates](https://github.com/selvxhini-10/SoleMates)
- [BracketBot Waiter Delivery](https://github.com/SK3720/BracketBot-Project)
- [LegoBot](https://github.com/chhabra-anirudh/legobot)
- [Mr. Clean / RL-BOT](https://github.com/Karan-Gupta07/Mr.-Clean-Bracketbot-Cleaning-Robot)
- [Educational Pong](https://github.com/JinchengLikesPlanes/UTWAT_Hachathon_2026_BracketBot)
- [Gambit](https://github.com/GautamM303/-Gambit-)
- [AOT BracketBot](https://github.com/rodeanmoradi/bots2026)
- [BigFluffyRobot](https://github.com/cchryx/BigFluffyRobot)

## 18. Recovery limitations

- The old documentation URL is inaccessible and its relevant pages were not recoverable from the exact archived paths checked.
- The public successor organization no longer exposes the original `BracketBotOS`, `BracketBotApps`, or `BracketBotAI` repositories used by the setup trace.
- No public PyPI package named `bbos` was available.
- The core `bbos` implementation could not be independently audited; behavior is reconstructed from examples and source-checked project notes.
- Some 2026 repositories contain copied reference code, so import counts do not equal original platform integration.
- Hardware claims are intentionally separated from simulation, mock, copied-example, and hardware-unverified claims.
- Public source was synthesized rather than copied wholesale; consult the linked repository and its license before reusing code.

## 19. First practical next steps

1. On the actual robot, record `python -c 'import bbos; print(bbos.__file__)'`, the git/image version, and the available daemon/config tree—without opening motion writers.
2. Generate a machine-readable inventory of topics, schemas, periods, configs, app owners, and calibration hashes.
3. Run read-only probes for arm state, drive state, cameras, IMU, audio config, and map/SLAM freshness.
4. Implement fake BBOS and the one-owner arbiter before building another task demo.
5. Port one tiny behavior end to end: read-only preflight → explicit execute → bounded slow motion → measured arrival → explicit stop → fresh verification → run bundle.
6. Only then attach voice, MCP, or an LLM planner.
