# SLAM + possible-person-on-ground safety pipeline

## What the robot can do now

The robot already runs the three foundations needed for this feature:

| Capability | Live source | Verified on `bracketbot-184` |
| --- | --- | --- |
| Localize | `slam.pose` | Yes; position, XYZW quaternion, map epoch |
| Map traversable space | `mapping.grid2d` | Yes; 1500×1500 unknown/floor/obstacle grid |
| Align vision with 3D | `camera.rect` + `camera.points` | Yes; 512×384, equal timestamps, pixel-index correspondence |

`bbapps/emotion_greeter` now joins those sources with YOLO11 pose output. It
associates confident shoulder/hip/limb keypoints with aligned depth, checks
whether the torso and most of the visible body are low and extended in 3D,
and transforms the median body position into the current SLAM map.

The observation is deliberately named **possible person on ground**, not
"fall detected." A static camera observation cannot prove that a fall occurred,
why somebody is on the floor, or whether they need medical assistance.

## Runtime flow

```text
camera.rect.left ──> YOLO pose ──> 2D joints
       │                                │
       └── camera.points.idx_2d ────────┘
                         │
                         v
              base-frame 3D joints
                         │
       slam.pose ────────┴──> map coordinate + map epoch
                         │
                         v
         2 s persistence / 2 s positive-clear latch
                         │
             /tmp/bracketbot_ground_alert.json
                         │
                         v
        navigation cancels autonomous route and commands zero
```

The alert file is an atomic, versioned handoff between perception and the one
process that owns `drive.ctrl`. Perception never writes a drive command. The
navigator cancels rather than pauses a route, so a cleared observation cannot
cause a surprise automatic restart beside someone on the floor. Manual
teleoperation remains an explicit operator override.

The vision dashboard on port 8018 shows aligned-depth health, the ground-safety
state, confirmed tracks and their map coordinates, known SLAM map area, map
epoch, camera annotations, and timing metrics.

## Live verification on `bracketbot-184`

Verified on 2026-09-19 without commanding robot motion:

| Gate | Result |
| --- | --- |
| Rectified RGB/depth correspondence | Pass; exact matching timestamps over repeated frames |
| Continuous vision service | Pass; roughly 3.8 scans/s with aligned depth |
| Live mapping telemetry | Pass; map epoch and known/floor/obstacle area reported |
| Standing person negative case | Pass; 7–11 grounded keypoints, torso 1.25–1.34 m, state clear |
| Seated person negative case | Pass; torso 0.87 m, 25% low joints, state clear |
| Person map annotation | Pass; base-frame depth observation transformed into current SLAM coordinates |
| Interlock clear + synthetic alert parsing | Pass on robot; malformed updates also stay latched in unit tests |
| Deliberately lying positive case | **Not yet run** |
| Live autonomous drive cancellation | **Not yet run**; requires cleared test area and e-stop operator |

The last two gates are required before claiming end-to-end fallen-person safety.

## Response policy

A confirmed observation should cause this deterministic sequence:

1. Stop and cancel autonomous base motion.
2. Preserve the map coordinate, camera timestamp, confidence evidence, and map
   epoch in the alert record.
3. Notify the operator and ask the person a neutral question such as, "Are you
   okay?" Do not claim they fell and do not diagnose them.
4. Keep a standoff distance. Do not drive over, touch, lift, or manipulate the
   person autonomously.
5. Escalate to a configured human contact only under an explicit product policy
   with consent, authentication, rate limiting, and a clear false-alarm path.
6. Require positive clear evidence and a new operator navigation command before
   continuing.

Emergency-service calling is intentionally not implemented. That requires a
separate, jurisdiction-aware and user-consented escalation design; a pose model
must never place that call by itself.

## Run and inspect

SLAM, mapping, and depth must be active before the app starts:

```sh
cd ~/bbapps/emotion_greeter
/usr/bin/python3 main.py
```

Inspect the machine-readable state locally on the robot:

```sh
curl -sS http://127.0.0.1:8018/api/status
cat /tmp/bracketbot_ground_alert.json
```

## Required physical calibration before relying on it

The integration and geometry are implemented, but the score thresholds still
need a labelled robot-specific validation set. Do this with the base stationary,
an operator at the physical e-stop, and no autonomous navigation:

1. Record synchronized rectified RGB, points, SLAM pose, and emitted assessment
   for standing, sitting, crouching, kneeling, exercising, partially occluded,
   and deliberately lying poses.
2. Include varied body sizes, clothing, lighting, floor materials, orientations,
   and distances throughout the usable depth range.
3. Measure person-detection recall separately from posture classification. A
   posture classifier cannot recover a person YOLO failed to see.
4. Tune for high recall, then report false alerts per hour and missed grounded
   people by scenario. Keep `unknown` when depth or torso evidence is insufficient.
5. Test the full stop path with wheels lifted or in a cleared test area: inject
   a synthetic confirmed alert, verify `drive.ctrl` becomes zero, verify the
   route is cancelled, and verify clearing does not auto-resume.
6. Only then run a slow, supervised approach-to-standoff trial. A future
   approach action must target a collision-checked standoff pose, never the
   person's map cell.

Until that dataset and stop-path test pass, this is a working prototype and
operator aid—not a certified safety system.

For an image-free telemetry capture, run this once per staged pose on the robot
or through an SSH port forward:

```sh
python3 scripts/record_ground_pose_dataset.py \
  --url http://127.0.0.1:8018/api/status \
  --label standing --seconds 15
```

Repeat with `sitting`, `crouching`, `kneeling`, `lying`, and `empty`. The JSONL
records contain the label, 3D evidence, map pose/epoch, and timing health, but no
image, expression estimate, or identity data.

## Current limitations

- YOLO can miss severe occlusion, blankets, unusual poses, or people partly
  outside the rectified field of view.
- Stereo depth can fail on dark, reflective, textureless, or distant surfaces.
- Track IDs are session-local and may change after occlusion.
- Map coordinates inherit SLAM drift and loop-closure changes; `map_epoch` lets
  consumers reject stale coordinates.
- Thresholds are engineering defaults, not validated clinical or functional-
  safety limits.
- Automatic approach is intentionally disabled pending physical validation.
