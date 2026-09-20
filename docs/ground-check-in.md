# Ground check-in

The **Check on person** dashboard button starts one bounded approach. The
existing YOLO pose/depth detector must have confirmed one possible person on
the ground for two seconds. The robot slowly faces and approaches that track,
stops outside its body envelope, waits for stationary wheels, and says exactly:

> hello specimen, are you in trouble

It completes after speaking once. Another button press is required to repeat.
The action never starts on boot or as a side effect of a navigation alert.

## Setup

1. Deploy both updated vision files to the existing robot app directory:

   ```sh
   scp bbapps/emotion_greeter/main.py bbapps/emotion_greeter/ground_safety.py bot:~/bbapps/emotion_greeter/
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

`scripts/ground_approach.py` contains separate range and bearing PID controllers
with bounded integrals, filtered derivatives, anti-windup, and reset on motion
inhibition. Default gains `(P, I, D)` are `(0.12, 0.01, 0.025)` for range and
`(0.65, 0.015, 0.05)` for bearing. These are initial conservative gains, not
hardware-tuned values. Limits are 0.05 m/s forward, 0.20 rad/s yaw, 0.04 m/s²
acceleration, and 0.30 rad/s² angular acceleration. It never reverses and first
aligns within 15° before advancing.

The envelope is the furthest depth-associated joint from the median body
position plus 0.25 m. Its radius can grow but cannot shrink during an attempt.
The robot stops 1.0 m outside it, with a 5 cm arrival tolerance, then requires
wheel speed below 0.015 m/s and yaw speed below 0.04 rad/s for 0.6 seconds.
This margin is based on visible joints; occluded limbs and depth errors still
require physical evaluation.

Only one alert with a current matching pose is eligible. A latched alert with
unknown/missing depth, stale/future camera time, malformed data, or multiple
alerts commands zero. Vision older than 0.65 s or depth older than 0.30 s also
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
