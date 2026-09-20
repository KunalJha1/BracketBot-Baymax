# Ground check-in

The **Check on person** dashboard button starts one bounded approach. The
existing YOLO pose/depth detector must have confirmed one possible person on
the ground for two seconds. The robot slowly faces and approaches that track,
stops outside its body envelope, waits for stationary wheels, and says exactly:

> hello specimen, are you in trouble

It completes after speaking once. Another button press is required to repeat.

## Automatic check-in

The voice assistant (`bbapps/greeter/local_assistant.py`) also starts the same
approach by itself. `GroundAlertWatcher` polls `/tmp/bracketbot_ground_alert.json`
twice a second, and when it holds one fresh, confirmed alert it starts the
`check-on-person` action: the assistant says it is coming over, runs
`robot_follow.py --ground-approach --no-speech` with the same heartbeat as
"follow me", and speaks the check-in line itself on arrival (it owns the only
`speaker.audio` writer). Every limit below still applies, and the neck flashes
red/blue for as long as the alert holds.

- One attempt per alert episode. Arriving, a refused start, and a spoken
  "stop" all end the episode; nothing restarts until the alert has been clear
  for 10 seconds.
- It waits while another voice action or a wake-word turn is in progress, and
  does nothing for two simultaneous alerts or a vision file older than 1 second.
- "Hey BracketBot, stop" cancels it like any other action.
- `--no-auto-ground-check` on the assistant turns it off; the dashboard button
  is unaffected.

## What counts as lying down

`ground_safety.assess_ground_pose` needs a low torso, most joints near the
floor and a long footprint, and additionally a lying posture: the highest
shoulder at most 0.45 m up, shoulders within 0.30 m of hip height (a
near-horizontal trunk), and, when the head has depth, a head at most 0.55 m up.
Someone sitting on the floor with their legs out passes the first three tests
but not these, and reads as `clear` with the reason
`low but trunk upright: sitting or crouching, not lying`.

The depth cloud ends about 1.7 m out, so most people on the floor have no depth
on their joints. For those, `assess_ground_pose_monocular` slides every joint
down its camera ray onto the floor (camera model fitted from `camera.points`,
0.02 px error) and checks bone lengths: a body really lying there keeps human
proportions, while a standing, seated or crouching one comes out metres long
and reads `clear`. Reasons from this path start with `mono:`. Detection gaps
under 0.7 s no longer restart the 2 s hold, tracks survive 8 missed frames, and
a confirmed alert nobody has seen for 15 s is dropped instead of latching
forever.

## Setup

1. Deploy the updated vision files to the existing robot app directory:

   ```sh
   scripts/bot push bbapps/emotion_greeter/main.py bbapps/emotion_greeter/ground_safety.py bbapps/emotion_greeter/floor_roi.py --to '~/bbapps/emotion_greeter/'
   ```

   Restart the vision service using the robot's normal service workflow while
   the robot is idle. Its existing model assets/dependencies are unchanged.
   The atomic `/tmp/bracketbot_ground_alert.json` must now contain
   `approach_schema_version: 1`, `session_id`, `observations`, and
   `depth_aligned`. Old or unavailable producers are refused before opening
   the drive writer. The dashboard only deploys runners into `/tmp`; it does
   not replace or restart this persistent vision service.

2. Retain the robot-specific `~/.config/baymax/follow.json` calibration described
   in [follow readiness](follow-readiness.md). The runner still checks wheel
   signs, depth axes, self mask, battery, balance, and conflicting drive owners.
   The navigation app must release drive ownership before this mode can start;
   its normal ground-alert stop interlock is unchanged.

3. Ensure `espeak-ng` is installed on the robot. Speech uses existing local
   synthesis and BBOS speaker helpers, with no LLM or network dependency. The
   line is rendered before opening the drive writer, so a missing speech
   dependency refuses the start. An available speaker writer is reserved until
   the approach ends. If the voice assistant owns it, the runner checks its
   speech relay before moving and requests the line on arrival. A successful
   playback receipt is required before reporting completion.

   When using the voice assistant, deploy these updates too, then restart that
   service while idle (the dashboard does not restart persistent services):

   ```sh
   scripts/bot push bbapps/greeter/local_assistant.py bbapps/greeter/speech_relay.py --to '~/bbapps/greeter/'
   ```

   An older, busy, or unavailable relay refuses the start with a visible error.
   Stop cancels a queued request or stops streaming its remaining audio chunks;
   audio already buffered by the speaker may still drain. Occupied LEDs are
   left with their owner. An idle person tracker is allowed to stay running;
   BBOS still refuses a second `drive.ctrl` writer if it is actively turning.

4. Run `python3 scripts/robot_dashboard.py` and open
   <http://127.0.0.1:8020/>. **Check on person** owns the same exclusive base
   session and heartbeat as Follow. Keep the page open. **Stop check-in** or
   **Esc** cancels both motion and speech. The following-distance slider is
   disabled for this action.

## Controller and stop behavior

`scripts/ground_approach.py` uses the same `FollowController` and `FollowConfig`
PID tuning as normal following in `scripts/follow_core.py`. Current gains
`(P, I, D)` are `(0.6, 0.3, 0.15)` for range and `(1.2, 0.2, 0.12)` for bearing.
Future changes to that shared tuning also apply to ground check-in.
Ground approach limits remain 0.05 m/s forward, 0.20 rad/s yaw, 0.04 m/s²
acceleration, and 0.30 rad/s² angular acceleration. It never reverses and first
aligns within 15° before advancing.

The shared controller subtracts a continuous deadband from each error, filters
the derivative (0.25 s for range, 0.15 s for bearing), and only accumulates the
integral near the target (0.3 m range error, 20° bearing error). Integral output
contributions are bounded at 0.2 m/s and ±0.15 rad/s; the final creep-speed caps
still apply, with anti-windup at saturation. Ground mode resets both PIDs when
motion is inhibited or the control clock has a long gap. The desired distance
to the body centre is `body_radius + 0.6 m`; angular target-velocity feedforward
is zero because this is a ground-pose approach. It also inherits the runner's
background depth processing and calibrated output turning sign (`omega_sign`).

The envelope is the furthest depth-associated joint from the median body
position plus 0.25 m. Its radius can grow but cannot shrink during an attempt.
The robot stops 0.6 m outside it (above the 0.45 m range at which the supervisor
blocks forward motion), with a 5 cm arrival tolerance, then requires
wheel speed below 0.015 m/s and yaw speed below 0.04 rad/s for 0.6 seconds.
This margin is based on visible joints; occluded limbs and depth errors still
require physical evaluation.

Only one alert with a current matching pose can establish a target. The latest
controller can bridge an unavailable observation for at most 1.0 s using wheel
odometry, while the camera capture must remain at most 2.0 s old. New observations
require a file published within 1.0 s. Once those limits expire, or depth is older
than 0.5 s, it commands zero, including rotation. A changed vision session/track ID or a
position jump over 0.4 m terminates the attempt. Total duration is bounded to
120 seconds. Safety stops bypass acceleration smoothing.

The local obstacle corridor extends 0.85 m forward, covers robot width plus
0.20 m on each side, and includes points from 0.06 to 1.70 m above the fitted floor.
Ten points block motion immediately; the person's points are never removed
from obstacle checking. Sparse clouds cannot establish clearance. This is a
local approach controller, not a route planner; it waits at obstructions.

## Validation

Laptop tests cover the producer/consumer handoff, stale and ambiguous targets,
PID limits, heading direction, obstacle braking, identity changes, standoff,
wheel settling, one-shot speech, cancellation, and dashboard mutual exclusion.
`--simulate` exercises the dashboard flow without connecting to hardware.

`tests/test_ground_check_integration.py` runs the real ground assessment,
confirmation timer, atomic file publisher, depth processing, control runner,
PID and speech handoff against synthetic joints, a ramped floor and simulated
wheel feedback. It checks standalone and shared-speaker completion plus obstacle,
depth loss, vision loss, heartbeat loss and cancellation. Separate tests check
speaker receipts/cancellation, busy motor ownership, the dashboard launch command,
and imports from freshly copied deployment bundles. These do not validate camera
inference, wheel calibration, physical clearance, or audible output on the robot.

After dashboard connection has copied the runner bundle, a read-only robot
diagnostic is:

```sh
scripts/bot py /tmp/robot_follow.py --ground-approach --dry-run --no-heartbeat --no-led
```

This does not open motor, LED, or speaker writers.
Keep the robot stationary for this diagnostic because no motion compensates
the target range. Existing `--preflight` and `--rotate-only` options also work
with `--ground-approach`; live motion still requires dashboard heartbeats.

Hardware approach and voice playback have not been validated by the laptop
tests. Complete the positive lying-pose and stop-path trials in
[ground safety](slam-ground-safety.md), then tune and test in a clear area with
an operator at the physical stop. Neither detection nor this question infers
that a person fell or needs medical treatment.

## Seeing a body on the floor at range

`camera.rect` is 512x384 and the pose model runs at 320 px, so a body lying a few
metres out is a handful of pixels and was not detected at all (`people=0`) while
the same model on the raw 1280x960 eye found it. The CPU is saturated (a 640 px
pass costs ~0.5 s), so `bbapps/emotion_greeter/floor_roi.py` instead crops the raw
eye to the floor 1.6-6 m ahead and runs the same 320 px model on that: raw-frame
pixel density at the small model's cost. There is one pose pass per frame. Every
third frame is the floor crop; once somebody is `checking` or `alert`, every
frame goes to whichever view sees them (`next_view_focus`), and the other view's
people are carried forward for up to 1 s. Crop detections are mapped into
`camera.rect` pixels (`RAW_TO_RECT`, fitted from 3.6k SIFT matches, 0.56 px), so
tracking, depth lookup and the floor-plane test are unchanged.

Iterate offline, not on the robot: record with `scripts/ground_record.py`
(read-only), then run `scripts/ground_replay.py RECORDING --sheet out.jpg`,
which pushes every frame through the same detection, merge, tracking, floor
test and alert hold as the robot.

## Why the approach used to stay still

Two things held it at zero even with a valid alert, both found with
`robot_follow.py --ground-approach --dry-run --no-heartbeat` against a synthetic
alert file:

- The depth cloud's floor is a ramp (~0.10 m per metre). Ground mode did not
  level it, so ~11,000 floor points filled the obstacle corridor and the state
  was permanently `BLOCKED`. `ground_perception` now levels the floor like normal
  follow, and the obstacle cut is 6 cm (levelled floor noise stops at 5 cm).
- Vision frames are 0.3-1.5 s old and drop out now and then, and each expiry
  reset the acceleration ramp. The loop now pins the person in the odometry
  frame and dead-reckons from wheel feedback between frames; a missing
  observation is bridged for 1 s, frames up to 2 s old are accepted, and a
  pinned target that moves more than 0.4 m aborts with `target-jumped`.

The robot still needs about a metre of genuinely clear floor ahead: a table or
chair within 0.85 m is a real `BLOCKED`.

