# Follow mode — handover for robot testing

**Current setup instructions:** [Follow readiness](follow-readiness.md). Use the
per-robot JSON calibration and `--preflight` workflow there before these physical
gates. It supersedes the old instructions to edit source constants or turn an
entire nearby cloud into a self mask. The historical laptop test count below
is not the current suite count.

Everything in this branch has been tested on a laptop only: 118 automated tests
pass, including a closed-loop simulation. **Nothing has run on the robot.** This
page is what you need to change that. Design:
`docs/superpowers/specs/2026-09-19-person-follow-design.md`.

## What it is meant to do

Press **Follow me** in the dashboard and stand 0.5–2 m in front of the robot.
The only person-sized shape in that zone for half a second is locked on, and the
robot then holds the slider's distance (0.6–1.5 m, default 1.0 m) to within
±20 cm while you stand, turn, or walk slowly. It finds you from the depth
camera's 3D points alone — no neural network and no camera image.

## Voice: "follow me"

Say **"follow me"** (also "come with me", "walk with me"). The assistant turns to
face you, then starts `~/bbapps/follow/robot_follow.py` with
`--no-led --ignore-writer person_tracker.py` and feeds it heartbeats; say **"stop"**
or **"stop following me"** to end it. If the assistant dies the runner stops within 1 s.
Deploy the runner with `scripts/bot push scripts/robot_follow.py scripts/follow_*.py --to /home/bracketbot/bbapps/follow`
(`run_robot_local_voice.sh` does it too).

Speed comes from two PIDs in `follow_core.FollowController`: range error -> v
(`v_kp/v_ki/v_kd`) and bearing -> omega (`w_kp/w_ki/w_kd`), then the rate limiter.
Defaults are deliberately gentle: 0.15 m/s, 0.5 rad/s, 0.25 m/s^2. Jerky -> lower
`accel_up`/`alpha_max` or `*_kp`; trails a steady walker -> raise `v_ki`; overshoots
the gap -> raise `v_kd`. Check a change with `pytest tests/test_follow_sim.py` first.

## Expected behaviour that is not a bug

- **A pillar, coat rack, or tall plant is person-sized.** Start in open space.
- **Standing within about 10 cm of a wall or another person** merges you with it:
  the robot loses you (state LOST).
- **0.3 m/s cap.** It cannot keep up with normal walking; it catches up when you
  stop. Nothing above 0.30 m/s is allowed.
- **Range is measured to the front of your torso**, about 10 cm nearer than your
  centre — that is the surface the camera sees. Tape marks line up with the front
  of the torso, not the toes.
- **After losing you for 10 s** it re-locks onto whoever then stands in front.
- It never drives backward, and stops for anything in a 0.6 m corridor ahead.

## Before you start

1. **A person stays at the physical e-stop** for every step that can move the
   base (G3 onward). 4 m of clear floor ahead. No arm gestures running; the base
   in BALANCE mode, not Lean.
2. Laptop: Python 3.10+, `uv`, and SSH to the robot as `bot`, `botwifi`, or
   `bracketbot@bracketbot-184.local`. Robot: BBOS at `~/bbos`, `uv` at
   `~/.local/bin/uv`.
3. **The depth daemon must be publishing** `camera.points` and `camera.depth`. It
   was off the last time this robot was probed, and nobody has recorded how it is
   started here. Find out, start it, and write the command into
   `docs/robot-facts.md` — the rest cannot run without it.
4. Check out this branch and confirm the laptop side is healthy:

```bash
uv run --extra dev python -m pytest -q          # expect 118 passed
python scripts/robot_dashboard.py --simulate    # open http://127.0.0.1:8020, no robot involved
```

## How to stop the robot

- **Esc** or the **Stop action** button in the dashboard.
- Closing the dashboard, or losing Wi-Fi: the runner stops itself within ~1 s.
- Closing the browser tab: same, within ~1 s (the heartbeat stops).
- The **physical e-stop** — always the fallback.

## The test sequence

Follow `docs/superpowers/plans/2026-09-19-person-follow-depth-only.md`, Task 5.
It has the exact commands and the pass criterion for each gate. Summary:

| Gate | What it does | Moves the robot? |
|---|---|---|
| G0 | Read-only probe; `robot_follow.py --check` lists what it sees as person-sized | no |
| G2 | Dry run: full logic, never opens the drive writer; tape marks at 0.6/1.0/1.5 m; wall, chair, pillar and bystander checks | no |
| G3 | Turning only: it faces you as you walk an arc | yes (turns) |
| G4a | Following at 0.15 m/s: standing, step-backs, slow stroll, slider change | yes |
| G5 | Obstacle: a 30 cm box placed in its path | yes |
| G6 | Link loss: Ctrl-C the dashboard; then drop Wi-Fi | yes |
| G4b | Repeat at 0.30 m/s, then G5 again | yes |

Stop at the first gate that fails, record what happened, and don't move to the
next one. Two failures have known fixes written into Task 5: wheel
direction (G3/G4a) and a constant range offset (G2 — do not patch it in code;
the point cloud needs recalibrating).

## What to send back

- `artifacts/follow_probe/report.json` and `artifacts/follow_probe_left/report.json`
- The `--check` output with the area clear, and with a person at 1 m
- Every runner CSV (`scp 'bot:/tmp/baymax_follow_*.csv' artifacts/follow/`) plus
  `python scripts/follow_log_report.py <csv>` for each
- Which objects in the room registered as person-sized, and any gate where the
  robot behaved unexpectedly (what you saw, and the CSV covering it)
- Video of G5 if you can — the obstacle stop is the hardest one to judge from logs

## Known open items (not blockers, listed in the spec §13)

Some tests still assert a clothing-colour bystander check that the depth-only
design no longer has; the runner's main loop has no automated test; the CSV log
is not flushed periodically. None of these affect what the gates measure.
