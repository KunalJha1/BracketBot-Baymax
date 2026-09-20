# Fall detection concept check

Question asked: can the BracketBot use the committed `yolo11n-pose.pt` to spot a
person who has fallen on the floor and is in pain?

Reproduce everything below with:

```sh
python scripts/fall_concept_check.py --verbose
```

It needs only NumPy. It does not open a camera or move the robot.

## Verdict

Split the question in two, because the two halves have opposite answers.

| Half | Verdict |
| --- | --- |
| "person is on the floor" | **Feasible now.** Most of the code already exists and the geometry separates falls from normal posture. |
| "person is in pain" | **Not feasible as vision.** Do not build it. Ask the person instead. |

The honest product is **"possible person on the ground"** plus a spoken check-in,
not a pain classifier. That is also what `ground_safety.py` already says in its
own module docstring, and the measurements below back it up.

## What already exists

This is further along than it looks:

- `yolo11n-pose.pt` is a **pose** checkpoint, not a plain detector. It gives 17
  COCO keypoints per person, which is what makes posture geometry possible at
  all. A bounding box alone would be far weaker.
- `bbapps/emotion_greeter/ground_safety.py` already implements the whole
  decision layer: depth association (`keypoints_in_base_frame`), the posture
  test (`assess_ground_pose`), and a hold/clear debouncer (`GroundAlertTracker`).
- `docs/yolo-robot-port.md` already proved the checkpoint loads, exports to ONNX
  and runs — on a dev machine, not on the Jetson.

What had **never** been tested is whether `assess_ground_pose` actually
discriminates. It had no test coverage. That is what this concept check adds.

## What was measured

Twelve labelled synthetic COCO-17 skeletons were placed in the robot base frame
using the real camera geometry from `docs/robot-facts.md` (head camera 1.55 m up,
pitched 33° down, per-eye fx = fy = 447.13). Each was run through the **shipped**
decision code by two evidence paths.

| scenario | truth | depth path | monocular path |
| --- | --- | --- | --- |
| fallen supine @2 m | on floor | hit | hit |
| fallen supine @3 m | on floor | hit | hit |
| fallen on side @2.5 m | on floor | hit | hit |
| fallen prone, legs occluded | on floor | hit | hit |
| standing @2 m | normal | ok | ok |
| standing @3.5 m | normal | ok | ok |
| sitting in a chair | normal | ok | ok |
| crouching | normal | ok | ok |
| bending over to pick something up | normal | ok | ok |
| **sitting on the floor, cross-legged** | normal | **false alarm** | ok |
| **lying on a 0.45 m sofa** | normal | ok | ok (unknown) |
| **plank / push-up** | normal | **false alarm** | **false alarm** |

Totals: both paths caught **4 / 4** on-floor cases. Depth produced 2 false alarms
out of 8 normal postures; monocular produced 1.

The easy negatives — standing, sitting, crouching, bending — are cleared
comfortably, not marginally. Standing scores 0.12 against a 0.62 threshold. That
is the result that makes the approach viable.

## The three failure modes, and what to do about each

### 1. Sitting on the floor reads as a fall (depth path)

Scored **0.6249** against a threshold of **0.62**. It fails by four
ten-thousandths. This is not a robust pass/fail — it is a coin flip that happened
to land wrong, and real keypoint noise will make it flip either way.

Someone sitting on the floor genuinely is on the floor, so this is arguably a
labelling problem rather than a detector bug. But an alert here is still wrong,
and the margin means the current threshold is not defensible as tuned.
**Action:** treat the 0.55–0.75 score band as "ask, don't alert" rather than
trying to tune the threshold to a number that separates these by luck.

### 2. A plank or push-up reads as a fall (both paths)

Scored **0.877** — confidently wrong, and it survives the 2-second temporal hold
because someone holding a plank stays there. Neither geometry nor debouncing
fixes this. It needs either motion history (a fall is a fast transition, a plank
is entered slowly) or the verbal check-in below. **Action:** accept it and let
the check-in resolve it.

### 3. Depth may not be running at all

`docs/robot-facts.md` records that at probe time `camera.points`, `camera.depth`
and `camera.rect` were **not running** — the depth daemon was off. Everything in
`ground_safety.py` requires depth, so in that state it returns `unknown` forever
and detects nothing.

The monocular path in the harness is the fallback: assume the lowest visible
joint touches the floor, intersect that pixel ray with the floor plane to get
range, then lift the rest of the skeleton onto the vertical plane at that range.
It reuses the identical `assess_ground_pose` thresholds, so the comparison
isolates the evidence source.

It works — 4/4, and it beat the depth path on false alarms — but it has a
dangerous failure that had to be fixed. A person on a **sofa or bed** breaks the
floor-contact assumption, and the path scored that at **1.0**: maximally
confident, completely wrong. Exactly the alert you least want at 3 a.m.

The fix is an anthropometric scale gate. When the assumption is violated the
whole recovered body inflates uniformly:

| segment | real fall | person on a sofa | adult limit |
| --- | --- | --- | --- |
| shoulder → hip | 0.42–0.50 m | **0.72 m** | 0.62 m |
| hip → knee | 0.37–0.45 m | **0.64 m** | 0.55 m |
| knee → ankle | 0.38–0.42 m | **0.61 m** | 0.52 m |

A 0.72 m torso is not a person. So the monocular path now refuses to score a body
it could not have recovered correctly and returns `unknown` instead of `alert` —
converting a confident false alarm into an honest abstention. That is the right
failure direction for a safety system.

This gate does **not** rescue the plank case (its segments are plausible), and it
is not meant to.

## Runtime cost is not a problem

Measured on this dev machine (Windows x86, **CPU only**, torch 2.14.0+cpu,
ultralytics 8.4.156, Python 3.12). Warm median over 12 runs after 3 warm-up
passes, on a random frame the size of one 1280×960 stereo eye:

| input | imgsz | warm median | fps |
| --- | --- | --- | --- |
| one stereo eye | 640 | 57.0 ms | 17.5 |
| one stereo eye | 480 | 45.9 ms | 21.8 |
| one stereo eye | 320 | 30.4 ms | 32.9 |

The checkpoint also re-verified clean: task `pose`, classes `{0: person}`,
2,874,462 parameters, SHA-256 matching `docs/yolo-robot-port.md`.

Fall detection does not need a high frame rate — `GroundAlertTracker` uses a
2-second hold, so 5–10 Hz is ample. Even unaccelerated CPU inference clears that
by a wide margin, and the Jetson has CUDA. **Do not spend effort on TensorRT for
this feature until a measurement on the robot says you need it.**

Caveat: this is x86 CPU, not the Jetson. It bounds the problem rather than
answering it. The port gate in `docs/yolo-robot-port.md` is still unexecuted on
real hardware.

## Why "in pain" should not be built as vision

Two independent blockers, either one sufficient.

**The model has no pain class.** `enet_b0_8_best_afew.onnx` is an AffectNet
8-class model: anger, contempt, disgust, fear, happiness, neutral, sadness,
surprise. Pain is not among them and is not a subset of them — clinically it is
scored from facial action units (PSPI), a different construct on a different
dataset. Reading "pain" out of a "sadness" logit would be inventing a medical
signal.

**The face is not there to classify.** From the measured intrinsics:

| range | face width | YuNet detection | expression usable |
| --- | --- | --- | --- |
| 0.8 m | 84 px | likely | yes |
| 1.5 m | 45 px | likely | no |
| 2.5 m | 27 px | unreliable | no |
| 3.0 m | 22 px | unreliable | no |

The expression model needs roughly 80 px of face, which this camera only supplies
within about **0.84 m**. A fallen person is first seen at 2–3 m, where the face is
20–35 px — detectable at best, not classifiable. And a person face-down or turned
away presents no face at all, which is precisely the posture that matters most.

## Recommended architecture

Detect posture, then **ask**. The robot already has a speaker, mic, wakeword and
TTS; the answer to "are you okay?" is a far better distress signal than any
facial inference, and it keeps a human in the loop.

```
YOLO11-pose  ──▶  posture evidence  ──▶  hold 2 s  ──▶  spoken check-in  ──▶  escalate
                  (depth, else                            "Are you okay?"      notify a
                   monocular + scale gate)                                      human
```

1. **Observe.** Run pose, label the observation `possible_person_on_ground` with
   its score. Never call it a fall and never call it pain.
2. **Debounce.** `GroundAlertTracker`, 2 s hold. Already written.
3. **Check in.** Speak. A plank, a yoga session and someone sitting on the floor
   all answer "I'm fine" — which is what resolves failure modes 1 and 2 that no
   amount of threshold tuning will.
4. **Escalate on silence.** No response is the actual alarm condition. Unresponsive
   is both easier to measure and more clinically meaningful than "looks like pain".
5. **Keep the action gate separate.** Per `docs/yolo-robot-port.md`, detection must
   not autonomously trigger movement or contact. Approaching a person who may be
   injured is a safety decision, not a detector output.

If you later want a physiological signal, note that rPPG (contactless heart rate)
was added in commit `d769a90` and is no longer in the tree. It is a more defensible
distress cue than facial pain — but it needs a still face at close range, so it
lands after approach, not at detection time.

## How to verify this on the robot

`scripts/fall_check_frame.py` is the bridge from the synthetic harness to real
frames. It runs the same `assess_ground_pose` decision code, but on real YOLO
keypoints and with the **fisheye** camera model instead of a pinhole
approximation. It is read-only and never commands motion.

Work through these in order. Each stage is useless until the one above it passes.

### Stage 0 — connect, and confirm you have the right robot

`bot` and `botwifi` are SSH aliases that need a `~/.ssh/config`; without one they
do not resolve at all. Each robot also answers to its own mDNS name,
`bracketbot-<unit>.local`, over both IPv4 and IPv6 — that name follows the robot
across DHCP leases, so it is more reliable than a hard-coded IP.

If you are already working on the robot directly, skip to stage 1.

**Check the host key before trusting it.** On a shared network with numbered
BracketBots, a `.local` name can belong to someone else's unit. Read the robot's
own fingerprint from its console or a known-good link:

```sh
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub   # run ON the robot
```

and only then accept the matching key on the laptop. Do not blanket-disable host
key checking.

### Stage 1 — prove the model runs on the Jetson

```sh
scp yolo11n-pose.pt scripts/check_yolo_runtime.py bot:/tmp/
ssh bot 'cd /tmp && ~/.local/bin/uv run --no-sync --project ~/bbos \
  python check_yolo_runtime.py --model yolo11n-pose.pt --device 0'
```

Expect task `pose`, classes `{0: person}`, 2,874,462 parameters and the SHA-256
in `docs/yolo-robot-port.md`. If Ultralytics or PyTorch is missing from `~/bbos`,
stop here — that is the real port question, and nothing downstream matters yet.

### Stage 2 — capture real frames, with consent

```sh
ssh bot '~/.local/bin/uv run --no-sync --project ~/bbos python capture_head.py \
  --out ~/head_frames --count 40 --every 1.0'
scp -r bot:head_frames ./head_frames
```

Capture the postures that the harness says are hard, not just easy ones: someone
lying on the floor, **sitting** on the floor, on a sofa, and doing a plank.

### Stage 3 — run the decision logic on those frames, off the robot

```sh
python scripts/fall_check_frame.py --image 'head_frames/*.jpg' --annotate out/
```

This is where the concept check either holds up or does not. Look at `out/` and
check that the skeletons land on the person, then compare verdicts against the
table above. Disagreement here means the thresholds need real data, not that the
approach is wrong.

### Stage 4 — live on the robot, robot stationary

```sh
ssh bot 'cd /tmp && ~/.local/bin/uv run --no-sync --project ~/bbos \
  python fall_check_frame.py --live --count 60 --device 0'
```

(Copy `fall_check_frame.py`, `ground_safety.py` and the checkpoint to `/tmp`
first.) This adds `GroundAlertTracker`, so you see `checking` → `alert`
transitions rather than single-frame verdicts. Record the per-frame latency.

**Keep the robot stationary for this.** Detection is an observation; approaching
a person who may be injured is a separate decision that needs its own gate.

### What would falsify the approach

Be willing to conclude it does not work. Stop and reconsider if:

- a standing person scores anywhere near the 0.62 threshold on real frames (in
  simulation standing scores 0.12, so real noise should not close that gap);
- keypoint confidence on a genuinely fallen, partly occluded body is too low to
  leave four usable joints;
- the monocular path cannot find a real floor contact, making every verdict
  `unknown`.

## Before this ships

The measurements above use synthetic skeletons and a pinhole camera model. They
prove the **decision logic** discriminates. They do not prove perception on the
robot. Still required:

- [ ] Run `scripts/check_yolo_runtime.py` on the Jetson itself — the port gate in
      `docs/yolo-robot-port.md` is still unexecuted on real hardware.
- [x] Replace the pinhole projection with the real **fisheye** model (k =
      [0.1287, −0.0281, 0, 0]). Done in `scripts/fall_check_frame.py` via
      `cv2.fisheye.undistortPoints`. This mattered more than expected: at the
      frame edge the fisheye and pinhole rays differ by **20.9°**, and the frame
      edge is exactly where a body on the floor appears. The undistorted centre
      ray independently lands 2.386 m ahead on the floor, matching the 2.4 m in
      `docs/robot-facts.md`.
- [ ] Pick **one eye** of the 2560×960 stereo frame. Never infer on the
      side-by-side image.
- [ ] Confirm whether the depth daemon runs in the deployed configuration. That
      single fact decides whether the depth or monocular path is primary.
- [ ] Re-run the scenario table against real recorded falls, with consent. Keypoint
      confidence on a real occluded body is the biggest unmodelled risk.
- [ ] Measure pose latency on the Jetson. One successful inference proves
      compatibility, not real-time suitability. The CPU numbers above suggest
      there is plenty of headroom, but they were not measured on the robot.
