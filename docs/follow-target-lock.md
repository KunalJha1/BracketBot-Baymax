# Follow target retention and turning

The reported failure was Dashboard Follow / `--rotate-only` losing a person or
stopping its turn. These changes address reproducible weaknesses in that runner;
the physical failure's exact cause is not yet established. SSH to the supplied
robot address timed out, so no test logs were read and no motion was performed.

## What changed

- Initial acquisition requires one current candidate and at least four distinct
  observations over the existing half-second window. A second person entering
  the start zone resets acquisition. Missing candidates, duplicate frames, and
  long frame gaps cannot manufacture a lock from old observations.
- When `camera.points.colors` contains aligned uint8 RGB triplets, each cluster
  carries a torso clothing descriptor. Soft chromaticity/brightness bins reduce
  sensitivity to modest exposure changes. The descriptor remains anchored to
  the original target rather than gradually learning a bystander.
- Spatial and clothing matches are considered together. Two similarly plausible
  matches pause tracking, including the case where one is exactly at the
  predicted position. A missing colour cue never silently weakens an existing
  appearance-based match.
- After uncertainty or a gap longer than 0.3 s, resuming needs the original
  clothing cue and at least three consistent observations spanning 0.25 s.
  Without an appearance cue, the robot can follow a continuously visible target
  geometrically, but cannot resolve identity after an ambiguous crossing or a
  longer dropout: stop and restart Follow with one intended target in front.
- After the existing LOST recovery window expires, Follow remains stopped and
  requires a restart. It never automatically selects a replacement person.
- Turning adds the target's estimated angular motion to the bearing correction.
  This reduces the lag when someone walks an arc around the robot. It accounts
  for the robot's own forward speed; speed/turn limits remain unchanged.
- Uncertain or stale target locations no longer carve a target-shaped hole in
  obstacle counting. Existing acceleration limits govern the commanded stop;
  this is not a claim of instantaneous physical braking.

The detector still finds shapes, not semantic humans. Similar clothes, major
lighting changes, merged depth clusters, a person against a wall, bad wheel
calibration, camera delay, and occlusion remain real limitations. This is a
lightweight colour cue, not neural re-identification or SLAM.

## Check on the actual robot

Use the deployment and calibration commands in [Follow readiness](follow-readiness.md).
Run `--check` first: every candidate now includes `appearance: true/false`.
If it is false, the RGB field is absent or unsupported. Do not claim the
clothing-based recovery path has been validated on that robot.

Repeat G2 stationary/dry-run tests, then G3 rotation with the e-stop operator,
then the established G4a–G6 sequence at 0.15 m/s. Include a slow continuous arc,
sideways walking, a brief occlusion, a differently dressed bystander crossing,
two similarly dressed people crossing, and a long loss. In ambiguous cases the
desired outcome is to stop, not to keep following at any cost. Stop/restart is
the deliberate way to choose a new target.

CSV logs now include `association` and `candidates` (positions and appearance
availability from the last processed frame). The dashboard explains
`ambiguous`, `confirming`, `identity-required`, and `restart-required` states.
Other useful evidence is `people`, `track_age`, `perception_ms`, `omega`,
`measured_omega`, and `rule`. These distinguish absent detections, rejected
matches, sensor/heartbeat stops, and wrong-direction feedback.

## Offline evidence

`tests/test_follow_target_lock.py` adds regressions for acquisition ambiguity,
missing/duplicate frames, clothing matching, conservative recovery, long-loss
retention, calibrated body-mask alignment, and continuous movement. The old
one-metre instantaneous position jump now correctly expects a stop without
identity evidence; physically continuous sideways motion is tested separately.

In a unicycle simulation with 10 Hz observations, 100 ms camera delay, noisy
measurements, 150 ms velocity response lag, and a person walking a 0.35 rad/s
arc at 1.2 m radius, the 95th-percentile bearing error across seeds 0/7/19 was
15.23 degrees with the prior bearing-only turn controller and 2.46 degrees with
angular feed-forward. These are simulation results, not measured robot results.

```sh
uv run --extra dev python -m pytest -q tests/test_follow_target_lock.py tests/test_follow_sim.py
```

For a more capable later version, combine a semantic person detector with an
appearance re-identification tracker and explicit target selection. Such a
system still needs uncertainty handling; a persistent tracker ID alone is not
proof of identity. See the [Ultralytics tracker documentation](https://docs.ultralytics.com/modes/track/).
