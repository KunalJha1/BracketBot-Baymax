# Robot facts (bracketbot-184, measured 2026-09-19)

Read-only probe of the live robot. Raw data: `artifacts/probe/` (probe report + one still per camera) and
`artifacts/robot/` (copied BBOS daemon configs, URDF + meshes, calibration). Re-run with
`scripts/probe_cameras.py` (see "Reproduce" below). Nothing here moved the robot.

## Access

`ssh bot` → `bracketbot@192.168.55.1` (USB link to the Jetson; key auth). Ubuntu 22.04, aarch64, BBOS at `~/bbos`,
apps at `~/bbapps`.

## Cameras

| Topic | Size | Rate | Notes |
|---|---|---|---|
| `camera.head.jpeg` / `camera.head.rgb` | 2560×960 (two 1280×960 eyes side by side) | ~29 Hz delivered (`Config` rate 60, `decimate` 2) | Fisheye. Fields: `jpeg_len`, `jpeg` (4 MB buffer), `timestamp`; raw: `rgb` (960, 2560, 3) |
| `camera.left.jpeg` | 640×480 | 30 Hz | Left **wrist** camera, looks along the gripper jaws |
| `camera.right.jpeg` | 640×480 | 30 Hz | Right wrist camera |

Not running at probe time: `camera.points`, `camera.depth`, `camera.rect` (depth daemon off).

**Head camera mounting** (`Config("depth").camera_to_base_3x4`): 1.55 m above the base origin, pitched **33° down**,
roll −1°. Rotation rows: `[1.000, 0.017, 0]`, `[0.010, −0.545, 0.839]`, `[0.015, −0.839, −0.545]`.

**Head intrinsics** (`artifacts/robot/depth/cache/stereo_calibration_fisheye.yaml`, fisheye model, per 1280×960 eye):
left fx = fy = 447.13, cx = 618.11, cy = 497.87, k = [0.1287, −0.0281, 0, 0]; right fx = 446.87, cx = 616.58,
cy = 498.80. Stereo baseline 65.0 mm.

At probe time the head camera saw the ceiling (`artifacts/probe/camera_head_jpeg.png`): the robot was tilted back, not
balancing. When upright, 33° down from 1.55 m puts the image centre about 2.4 m ahead on the floor. A table at 0.74 m
in front of the robot sits in the lower part of the image. Check this with a real frame before labelling.

## Arms

- 8 DOF per side, joints `lj0..lj6, left_left_gripper` / `rj0..rj6, right_left_gripper`. J0 is a prismatic lift.
- Gripper: revolute jaw pair (`*_left_gripper`, with `*_right_gripper` mimicking it), black foam pads (see the wrist stills).
- IK: `cfg.ik` = Rust `hybrid_ik` (`libhybrid_ik_lib.so`, aarch64 only), `base_link="arm_base"`, `ee_link="left_eef"`/`"right_eef"`,
  tolerances 5 mm / 0.01 rad, 5 RelaxedIK iterations per call, nominal elbow j3 = 1.5708.
- Control period 15 ms (`arm_ctrl`/`arm_state` are `@realtime(ms=15)`); daemon inner `dt` 6.7 ms.
- Home (turns): left `[0, 0, 0, 0.25, 0, 0, 0, -0.15]`, right `[0, 0, 0, -0.25, 0, 0, 0, 0.10]`.
- Current limits (A): J0 and J4–J7 13.2, J1–J3 1.96. Torque constants: J1–J3 2.50 Nm/A, others 0.204.

**Motor turns ↔ URDF** (`artifacts/robot/arm_*/constants.py`), linear, no robot needed:

```python
# left:  q = turns * 2π ; q[0] *= 0.0465 ; q[7] *= -1 ; q[:7] *= [-1, 1, -1, 1, 1, -1, -1]
# right: q = turns * 2π ; q[0] *= -0.0465 ; q[7] *= +1 ; q[:7] *= [-1, -1, -1, -1, 1, -1, 1]
```

**Policy scaling** (`bbapps/inference/bracketbot_adapter.py`): per joint, `(turns − cal_min) / (cal_max − cal_min)` clipped to
[0, 1], then ×200 − 100 for J0–J6 and ×100 for the gripper. `cal_min`/`cal_max` are in
`artifacts/robot/arm_*/ranges.calibration.json` (these belong to **this** robot only).

## Base

- Modes (`Writer("base.mode", Type("base_mode"))`, fields `mode` uint8, `lean_angle_deg` float32): 0 BALANCE (default),
  **1 LEAN**, 2 TWIST. Positive lean = top tilts forward. Range 1–15°, default 4°.
- The request **expires after 0.25 s** and the base falls back to BALANCE, so publish it continuously (the type is 100 ms).
- `drive.ctrl` `twist` = [m/s, rad/s], 10 ms type, 0.1 s timeout, limits 0.3 m/s and 1.0 rad/s. Wheel 0.165 m.

## Recorder (`dataset` daemon)

- Started and stopped by `dataset.flag` (`prefix`, `name`, `text`, `toggle_episode`, `drop_episode`).
- Records `drive.*`, `imu.*`, `arm_*.state`, `arm_*.ctrl`, `arm_*.target`, `quest.joystick`, leader arms, and the cameras
  `head`, `arm_left`, `arm_right` (JPEG → 30 fps MKV per camera).
- Uploads `datasets/<prefix>/<name>/episodes/epNNNNNN_<uid>_<chunk>.npz` and `.../video/<cam>_ep..mkv` via bb-server (key in `/etc/BB_API_KEY` on the robot).

## Reproduce

```sh
scp scripts/probe_cameras.py bot:/tmp/ && ssh bot 'export PATH=$HOME/.local/bin:$PATH; cd /tmp && \
  uv run --no-sync --project ~/bbos --with pillow python probe_cameras.py'
scp -r bot:/tmp/camera_probe artifacts/probe
```

Always pass `--no-sync`: without it `uv` re-syncs the robot's BBOS environment.
