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

1. Deploy both updated vision files to the existing robot app directory:

   ```sh
   scripts/bot push bbapps/emotion_greeter/main.py bbapps/emotion_greeter/ground_safety.py --to '~/bbapps/emotion_greeter/'
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
   dependency refuses the start.

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

Only one alert with a current matching pose is eligible. A latched alert with
unknown/missing depth, stale/future camera time, malformed data, or multiple
alerts commands zero. Vision older than 1.0 s or depth older than 0.30 s also
commands zero, including rotation. A changed vision session/track ID or a
position jump over 0.4 m terminates the attempt. Total duration is bounded to
120 seconds. Safety stops bypass acceleration smoothing.

The local obstacle corridor extends 0.85 m forward, covers robot width plus
0.20 m on each side, and includes points from 0.03 to 1.70 m above the floor.
Ten points block motion immediately; the person's points are never removed
from obstacle checking. Sparse clouds cannot establish clearance. This is a
local approach controller, not a route planner; it waits at obstructions.

## Validation

Laptop tests cover the producer/consumer handoff, stale and ambiguous targets,
PID limits, heading direction, obstacle braking, identity changes, standoff,
wheel settling, one-shot speech, cancellation, and dashboard mutual exclusion.
`--simulate` exercises the dashboard flow without connecting to hardware.

After dashboard connection has copied the runner bundle, a read-only robot
diagnostic is:

```sh
~/bbos/.venv/bin/python /tmp/robot_follow.py --ground-approach --dry-run --no-heartbeat
```

This does not open `drive.ctrl` or play speech (it does display status LEDs).
Keep the robot stationary for this diagnostic because no motion compensates
the target range. Existing `--preflight` and `--rotate-only` options also work
with `--ground-approach`; live motion still requires dashboard heartbeats.

Hardware approach and voice playback have not been validated by the laptop
tests. Complete the positive lying-pose and stop-path trials in
[ground safety](slam-ground-safety.md), then tune and test in a clear area with
an operator at the physical stop. Neither detection nor this question infers
that a person fell or needs medical treatment.
