# Follow mode after a pull

A pull installs code; it cannot establish camera orientation, identify the
robot's body in a depth cloud, or prove that following works in a particular
room. The runner now refuses motion without a valid robot-specific calibration.
Calibration persists outside the repository and outside `/tmp`, so neither
pulling nor dashboard deployment overwrites it.

The detector still identifies **person-sized shapes**, including some plants,
pillars, and coat racks. Start with one person in clear space. This change does
not add semantic person recognition or SLAM.

## 1. Check the code on the laptop

From a clean checkout of the branch containing the readiness fix:

```sh
git pull --ff-only
uv sync --extra dev --locked
uv run python -m compileall -q scripts bbapps
uv run python -m pytest -q
uv run python scripts/robot_dashboard.py --simulate
```

The dashboard is at http://127.0.0.1:8020/. The GitHub `Pull checks` workflow runs
compilation and the hardware-free suite on Linux and Windows with Python 3.10
and 3.13. Repository administrators should require those checks before merging
to `main`; adding the workflow alone does not enable branch protection.

## 2. Probe the robot without moving it

From the laptop, with the configured SSH alias `bot`:

```sh
scp scripts/probe_follow.py scripts/robot_follow.py scripts/follow_core.py scripts/follow_perception.py scripts/follow_calibration.py bot:/tmp/
ssh bot '~/bbos/.venv/bin/python /tmp/probe_follow.py --out /tmp/follow_probe'
ssh bot '~/bbos/.venv/bin/python /tmp/robot_follow.py --check'
scp -r bot:/tmp/follow_probe artifacts/
```

The robot needs BBOS and numpy in `~/bbos/.venv`. If that interpreter is absent,
use the existing BBOS environment through
`~/.local/bin/uv run --no-sync --project ~/bbos python` instead. Do not install a
different BBOS version just to satisfy the laptop's dependencies.

The probe saves current `idx_2d`, legacy `mask`, or index-free clouds. Neither
pixel-index field is required by the follow runner. Its `nearby_points` and
`nearby_box_base_xyz_*` fields are geometric candidates, **not proof of robot
body points**. Inspect the saved cloud and the physical scene. Do not exclude
the whole nearby box automatically: that can conceal actual obstacles.

If depth topics are absent, resolve the installed robot's depth-daemon setup
and record its actual start command in `docs/robot-facts.md`. This repository
does not have enough information to start every BBOS version automatically.

## 3. Calibrate once on this robot

With a person about 1 m ahead and 0.5 m to the robot's left, and other objects
out of that measurement region, run:

```sh
ssh bot '~/bbos/.venv/bin/python /tmp/probe_follow.py --out /tmp/follow_probe_left --person-left'
scp -r bot:/tmp/follow_probe_left artifacts/
```

Use the report's `robot_id` (the robot hostname) and measured `left_sign`.
An inconclusive result is not a sign measurement; repeat with an isolated
person. The median-based probe can be contaminated by furniture.

Create `~/.config/baymax/follow.json` **on the robot**. This template is
deliberately incomplete and must not be treated as measured calibration:

```json
{
  "schema_version": 1,
  "robot_id": "REPLACE_WITH_ROBOT_HOSTNAME",
  "evidence": "",
  "left_sign": null,
  "self_mask": null,
  "wheel_order": [0, 1],
  "wheel_signs": [1, 1],
  "motion_speed_limit": 0.15
}
```

Fill `evidence` with the date, calibration observations, and where the probe
artifacts are stored. `left_sign` is `-1` or `1`. `self_mask` is a list of
physically reviewed body boxes, each ordered as:

```text
[forward_min, forward_max, left_min, left_max, up_min, up_max]
```

All bounds are metres. From a verified base-frame body box, forward is base y,
up is base z, and left is `left_sign * base_x`; reorder the left bounds after
negation. Use multiple tight boxes if needed. A small margin may cover sensor
noise, but verify that a real obstacle just beyond the body remains visible.
Use `[]` only when inspection confirms no body exclusion is needed. Both
person detection and obstacle counting use these same boxes.

The wheel settings start with the existing commissioning assumption: entries
0/1 are left/right and positive means forward. Verify them during the supervised
turning/slow-drive gates; correct them in this file, never in source defaults.
The file allows supervised commissioning; it is not a certificate that those
gates have passed. Keep the speed limit at 0.15 during initial tests. To test
0.30, complete the lower-speed gates first and explicitly update the profile;
record the higher-speed results before treating that speed as validated.

Keep the profile on the robot; do not copy another robot's profile. After a
camera remount, robot hostname change, BBOS reflash, depth-calibration change,
or body/arm configuration change, remove or rename the profile and repeat the
affected calibration and dry-run checks. These physical changes are not
automatically detected by hostname matching.

## 4. Check readiness, then run supervised gates

```sh
ssh bot '~/bbos/.venv/bin/python /tmp/robot_follow.py --preflight'
```

`PREFLIGHT OK` means the profile matches this robot, input samples are usable,
the robot is upright, battery checks pass, and no recognized competing drive
process was found. It opens **no writers** and does not prove detection accuracy.
The slow battery topic gets a five-second startup window. During following,
stale IMU/wheel feedback and invalid depth data stop the runner; its exit path
sends zero twist. The daemon's own command timeout remains necessary when a
process stalls or is killed without cleanup.

Then follow [the handover's physical sequence](follow-mode-test-handover.md):
G2 dry run, G3 turning, G4a slow following, G5 obstacle, G6 link loss, then G4b
higher speed and G5 again. G2 never opens `drive.ctrl`; it can write LEDs.
Have the required person and e-stop operator present for motion tests.

```sh
ssh bot '~/bbos/.venv/bin/python /tmp/robot_follow.py --dry-run --no-heartbeat'
```

`--check` and `--dry-run` can run without a profile, but explicitly report
`UNCALIBRATED`; their assumed-coordinate results do not qualify the robot for
motion. A present but invalid profile is rejected even in diagnostic modes.

For ordinary use, start the laptop dashboard. It deploys all four runner modules
and reads the robot's persistent calibration automatically. There is no need
to patch constants after every pull. Custom runner invocations may supply
`--calibration /path/to/profile.json`.

## Evidence to retain

Record the commit, robot ID, BBOS/depth version, actual depth start command,
profile, probe reports/clouds, marked-distance observations, false detections,
and gate CSVs. The automated tests exercise both schema variants, profile
refusals, and the actual runner's stop/cleanup paths through simulated BBOS
readers and writers. They do not establish real braking distance, perception
latency, person identity, or obstacle detection reliability.
