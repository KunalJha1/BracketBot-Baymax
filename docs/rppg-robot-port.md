# rPPG on the head camera: port gate

Contactless heart rate (remote photoplethysmography) estimates a pulse from the
tiny colour changes skin shows as blood volume rises and falls. `rppg.py` holds
the signal chain; `scripts/robot_rppg.py` runs it against the robot's head
camera.

**This is not a medical device and must never be presented as one.** It is a
demo-grade estimate from a camera. It must not diagnose, triage, screen, or
gate any robot action, and it must not run on someone who has not agreed to it.
See "Consent and safety" below.

## Verdict

The signal chain is verified locally on synthetic and webcam data. It is **not
yet verified on the physical robot**, because the robot was offline during this
pass, and there is one unresolved risk that only a real run can settle:
**this app cannot lock the head camera's exposure.**

Run `scripts/robot_rppg.py --check` on the robot first. It answers the camera
questions without attempting a measurement.

## Why exposure is the deciding risk

The pulse changes skin brightness by roughly 0.1–1%. Auto-exposure, auto-gain,
and auto-white-balance corrections are larger than that, and they arrive as
sharp steps that a spectrum reads as fake peaks. On a laptop, `laptop_rppg.py`
switches the camera's own auto-exposure off through OpenCV and then sets
exposure itself, from the face (see the next section).

On the robot it cannot. The camera daemon owns the device; `robot_rppg.py`
holds a `Reader` and nothing else, by design, so it cannot reconfigure the
camera out from under every other app on the robot.

This leaves three outcomes, to be told apart by the `lum_drift_pct` figure that
`--check` reports (peak-to-peak ROI luminance over the sample, as a percentage
of the mean):

| `lum_drift_pct` | Reading | Action |
| --- | --- | --- |
| < 1% | The daemon is effectively holding exposure | Proceed; POS should work |
| 1–3% | Mild drift | POS may still hold; treat low-SNR results as failures, not readings |
| > 3% | Active AE hunting | Fix exposure in the camera daemon config, or do not ship the feature |

POS (the default method) is specifically designed to survive some illumination
drift, which is why it is the default and why `--method green` exists in the
laptop tool as the visible counter-example. It is not magic: a large enough AE
step still destroys the estimate.

If exposure has to be fixed, it belongs in the camera daemon's own config
(`Config("cam_head")`), as a deliberate, reviewed robot-wide change — not as
something a scan script toggles at runtime.

## Metering the face instead of the frame

Switching auto-exposure off answers half the question. The other half is what
value to switch it off *at*, and a camera's own AE is the wrong thing to ask,
because it meters the whole frame: it spends the sensor's range on a bright
window or a white wall and leaves the face wherever it lands. A face down at
15% of full scale carries its pulse in two or three counts, and quantisation
plus sensor noise then cost more SNR than any of the filtering downstream wins
back. A face with blown highlights is worse: a clipped pixel has had its
modulation removed by the sensor, so the skin gate is the only thing that can
help, by throwing those pixels away.

So `rppg.FaceExposure` closes a slow loop on the skin pixels the estimate is
actually taken over (`rppg.skin_luma_stats`, over the same mask as the mean
RGB). It aims for a mean of about 52% of full scale, with a 6% dead band, and
it steps exposure only while the face is outside that band or clipping at
either rail. Once inside it holds for 1.5 s and declares itself settled, after
which it stops touching the camera until the skin drifts more than 11% off
target — someone switching a lamp on, or stepping into the shade. Corrections
smaller than 4% are not taken at all, and a reversal halves the step size, so
the loop cannot start hunting the way the AE it replaces does.

Every move it does make is a step in the raw trace, orders of magnitude bigger
than a pulse. The callers therefore drop the buffer the step would land in
(`tracker.clear()`, keeping the tracking lock and the stacked spectrum, because
the rate did not change — only the scale did) and refill it. That cost is the
reason the loop is built to be reluctant: in practice it converges in two or
three moves in the first couple of seconds and is silent for the rest of a scan.

Which knob moves depends on the camera, and the units are not portable:

- `rppg.OpenCVExposure` (UVC webcams). DirectShow and Media Foundation report
  exposure as log2 seconds, where one count is a whole stop; V4L2 reports
  `exposure_time_absolute` in 100 µs units, which scale with the light. A
  request too small for a whole stop, or one that hits the end of the range,
  goes to sensor gain instead — gain is noise, but a face pinned at the bottom
  of the range has no pulse left to be noisy about.
- `rppg.Picamera2Exposure` (the Pi camera in `open_camera`). Exposure only ever
  moves in whole multiples of the mains-flicker period (8333 µs for 60 Hz,
  10000 µs for 50 Hz), because a part-cycle exposure samples the lamp's ripple
  instead of averaging it away and the beat lands in the pulse band. The
  remainder of each change goes to analogue gain, and the longest flicker-safe
  exposure is preferred so as little gain as possible is used.

`robot_rppg.py` does not run any of this. The head camera is a shared topic and
the scan holds a `Reader` only, so there is no control to drive and none should
be: retuning a camera every other service reads is exactly the robot-wide change
that belongs in the daemon config. What the loop contributes there is the
readout — the skin brightness and clipping figures in the debug view say
whether a scan was taken on a well-exposed face or on one the camera left in the
dark, which is the difference between a weak reading and a reading to repeat in
better light.

## Camera geometry

Derived from `docs/robot-facts.md` (head camera 1.55 m up, pitched 33° down,
fisheye, per-eye fx = fy = 447.13, cy = 497.87, 1280×960 per eye). **Confirm
against a real frame before trusting it.**

A standing adult's face (≈1.60 m) lands in the upper quarter of the frame at
every useful distance, so it is not clipped:

| Distance | Face row (of 960) |
| --- | --- |
| 0.6 m | ≈203 |
| 1.0 m | ≈218 |
| 2.0 m | ≈229 |

Face width falls off with distance, and rPPG needs skin pixels:

| Distance | Face width |
| --- | --- |
| 0.5 m | ≈139 px |
| 0.7 m | ≈99 px |
| 1.0 m | ≈69 px |
| 1.5 m | ≈46 px |

`FaceROI` rejects a face whose forehead-plus-cheek mask falls under 400 pixels.
That puts the expected working range at roughly **0.5–1.0 m**, matching the
50–70 cm the laptop tool asks for. `--check` reports the measured
`face_width_px` so this can be replaced with a real number.

Two further geometric notes: the head topic is 2560×960, two 1280×960 eyes side
by side, and `HeadCamera` splits it and uses the left eye — never infer on the
full stereo image. The lens is fisheye (k = [0.1287, −0.0281, 0, 0]); distortion
is mild near the centre and rPPG averages over a region rather than measuring
distances, so rectification is not required for the signal.

## Raw versus JPEG

The default reads `camera.head` (raw RGB). JPEG is lossy in exactly the way
that hurts here: chroma subsampling and DCT quantisation discard small colour
changes, which is the entire signal. `--jpeg` exists for the case where only
the encoded topic is available, and any result it produces should be treated as
weaker evidence.

## Which peak is the heart rate

A camera sees the pulse *waveform*, not a sine. The dicrotic notch puts a
second harmonic in the spectrum that is often taller than the fundamental, so
the tallest in-band peak is frequently 2× the real rate — and, because it is
the same peak in every window, the old aggregate called that doubled rate
confident. On synthetic traces with a harmonic-heavy waveform the chain
reported 143.6 BPM for a 72 BPM pulse.

`fundamental()` now takes the tallest peak and looks at half its frequency. If
that half is still a plausible rate (≥ 42 BPM) and carries a peak of its own —
at least 20% of the main peak's power, and at least 9 dB over the in-band noise
floor — the half is the fundamental and the tall line is its harmonic. The
thresholds sit in a wide gap measured on synthetic windows: genuine 2× locks put
the half-peak 11–19 dB over the floor, while windows whose peak was already the
fundamental put it below 7 dB.

That floor test has one blind spot, found while testing the whole-scan stack
below and fixed with it. "9 dB over the in-band noise floor" is not scale-free:
on a weak scan the floor itself rises, until a perfectly good fundamental sits
under the bar and the estimate silently doubles. Real 66 BPM traces whose
half-line held 68% of the peak's power cleared the floor by only 8.5 dB and
were reported as 132. A half carrying 40% or more of the peak's power is now
taken as the fundamental on that share alone, because noise near half the rate
is a few percent of the peak and never half of it.

The shortcut needs one guard. A loud artefact near half the rate can carry the
same share as a real fundamental — a dim 124 BPM trace had one at 57.6 BPM
holding 65%. The two are told apart by where they sit: a genuine second
harmonic is at *exactly* twice its fundamental, while an unrelated line merely
lands nearby. 57.6 is four BPM off the 61.5 that would be half, so the shortcut
requires the half-line within 0.02 Hz of exactly half, and the floor test still
judges the weaker, further-off halves it was written for.

The trade is deliberate and worth stating: a genuine rate near the top of the
band whose half collides with a strong artefact can be halved. That costs an
occasional reading in a range this demo barely serves, where lock-in was costing
a systematic 2× on ordinary resting rates. Watch for it if the scan is ever
pointed at someone exercising.

## Reading the rate off the whole scan, not the best window

A single 10 s window resolves the pulse only when the pulse is strong. That is
the wrong assumption for this camera: at a metre, under a ceiling light, with
exposure the app cannot lock, the pulse sits near the noise floor and *no*
window is clearly right. The old chain gated each window on SNR and aggregated
the survivors, so a scan where every window was marginal produced nothing at
all — on the benchmark's `far-face` family, four scans in nine returned no
reading.

`SpectrumStack` folds every analysed window's spectrum onto one grid and
averages them. The noise differs in each window, the rate does not, so the
floor falls and the pulse stays: a rate too weak to win any single window can
still be the tallest line in the average. Each window is normalised by its own
in-band power before it is added, so a window that caught a motion jerk cannot
shout down the clean ones, and weighted by its SNR so a clear window counts for
more. Windows below the per-window gate are stacked too — rescuing a weak scan
is the point, and the SNR weight already discounts them. Consecutive windows
overlap by 90%, so their noise is correlated and the floor falls more slowly
than it would for independent samples; the gain is real but it is not the
square root of the window count.

The reported rate is then read off the stack, and two numbers decide whether it
is reported at all, because a scan can fail in two different ways:

- **`snr_db`** asks how far the line stands above the broad in-band noise
  floor, and catches a scan whose spectrum is flat. Below −3 dB, no reading.
- **`margin_db`** asks how far it beats the best rival line elsewhere in the
  band — counting a rate's own harmonic and half as the same candidate, not as
  rivals — and catches the other case: a spectrum with several comparable
  lines, where the tallest wins by chance and would move somewhere else on the
  next scan. Below 2.5 dB, no reading.

The two were measured separately on synthetic scans. Correct readings beat
their best rival by 5.4 dB at the 10th percentile and 10.6 dB at the median;
traces with no pulse in them top out at 3.6 dB whatever their SNR says. Adding
the rival test is what let the SNR floor be loosened enough to rescue weak
scans without also accepting noise.

Three other things guard the number the robot says out loud:

- **Gaps.** Frames dropped for motion or a lost face leave holes, and
  interpolating across one invents a low-frequency swing sitting right where the
  pulse lives. `analyze()` takes the longest stretch with no hole over 0.5 s,
  falling back to the whole span when no stretch is long enough to use.
- **Aggregation.** The windows that passed the SNR gate are still combined with
  an SNR-weighted median, kept as `window_bpm`: an independent view of the same
  scan, from a different part of the evidence. The laptop tool and the robot
  drive the same tracker, so their readouts agree.
- **Confidence.** `confident` is deliberately harder to earn than a reading.
  The stacked spectrum must clear 2 dB, at least three windows must have passed
  the gate, they must agree within 6 BPM of each other *and* within 6 BPM of the
  stacked rate, and they must span at least 6 s of the scan. That last agreement
  is easy to forget and matters: the windows agreeing among themselves says
  nothing about whether they agree with the number being reported, and on the
  synthetic set one scan splits the two by 53 BPM. Consecutive 10 s windows overlap by 90%, so their agreeing with each
  other is weak evidence on its own; a noise peak that wins only the last few
  windows no longer passes. When a scan is not confident, the robot says the
  signal was weak and not to rely on the number — the correct outcome for a scan
  that weak, not a bug to tune away.

`scripts/rppg_benchmark.py` scores the chain over 99 synthetic scans. Six
families model the signal itself (clean, harmonic-heavy, auto-exposure steps,
motion, dropped frames, a pulse near the noise floor) and five more model what
this camera adds: continuous AE hunting it cannot lock out, a detector box that
wanders, a 15 fps stream, a small dim face at a metre, and a mains-flicker beat
inside the band.

Across that set, stacking took mean absolute error from 1.62 to 0.25 BPM, the
90th percentile from 0.92 to 0.36, and the worst single scan from 66.4 to 7.0.
The two families that model a distant or dim face improved most: `low-fps` from
5.83 to 0.17 BPM and `dim` from 9.14 to 0.60. Scans returning no reading at all
rose from five to nine, which is the intended trade — those are scans where the
stacked spectrum says there is nothing to report, and "no clear reading" is the
honest answer. No scan was both wrong by more than 5 BPM and marked confident,
before or after.

`python scripts/rppg_benchmark.py --roi` scores the other half: it renders
frames rather than traces — a pulsing face with hair across the forehead, a
specular highlight on one cheek, static shading, and a detector box that lands a
pixel or two off every frame — and runs them through `FaceROI`. Two notes on
reading it, both mistakes made while writing it. The pulse must be rendered into
the three channels at *different* depths, because POS works by projecting out
exactly the common-mode part that a flat modulation would be, and a face must be
shaded and textured, because a box wandering across a flat patch re-averages the
identical colour and costs nothing.

`tests/test_rppg.py` covers each of these behaviours. None of it is validation
against a real pulse: the benchmark is synthetic, so step 4 of the gate below
still has to happen.

## Which pixels get averaged

The stage before the signal chain decides what the signal is made of, and on
this camera it is worth as much as the chain. `FaceROI` does two things beyond
finding the face, both visible in the debug view below:

- **Steadying the box.** YuNet re-runs from scratch on every frame, so even
  someone holding still gets a box that moves a pixel or two. A real face is
  shaded and textured, so the ROI then averages slightly different skin every
  frame, and that wobble lands in the pulse band. `steady_box()` blends each box
  with the one before it, and takes a new box whole when its centre moves more
  than 35% of the box width — that is someone moving, and tracking must stay
  immediate.
- **Gating the skin.** The polygons are mostly skin by construction, but the
  minority that is not — a strand of hair, a glasses frame, a highlight on the
  nose bridge, a cheek edge catching background — carries no pulse, so it
  dilutes a signal that is already under 1% of the brightness, and it moves in
  and out of the mask as the box shifts. `skin_pixels()` keeps the pixels whose
  chroma is close to the face's *own* median and whose luminance is inside the
  4th–96th percentile.

The gate is deliberately relative rather than a fixed Cr/Cb skin range. A fixed
range is the usual way this is done and it is exactly where rPPG pipelines bake
in an assumption about skin tone; this one only asks that a pixel match the rest
of the face, so it behaves the same on dark skin as on light. That is tested,
not assumed. It does not remove the skin-tone limitation below, which is about
how much light returns from the dermis, not about which pixels are chosen.

**How much this is worth, measured rather than assumed: not much, on the rig
that can measure it.** `--roi` puts the two together at +0.6 dB of scan SNR on
a 70 px face and +0.1 dB on a 130 px one — real, in the direction claimed, and
an order of magnitude smaller than what stacking the spectra bought. Two honest
caveats in opposite directions. The rig understates the case, because its
contaminants are static: real hair sways, a real highlight slides across a
cheek as someone shifts, and both then modulate the average at frequencies a
static render never produces. But it is also the only measurement there is, and
it does not support describing the skin gate as a large accuracy win. What it
does support is the debug view, which needs the mask and the per-region counts
regardless, and the claim that neither change makes anything worse.

The stated reason to keep them is therefore margin rather than accuracy: both
act where the face is small and dim, which is where this camera puts a person
standing at a metre, and that is the regime where a scan stops returning a
reading at all. Whether that margin matters is a question for the robot, not
for a renderer.

## Seeing what the model sees

Every number above is a summary of a picture, and on a headless robot the
picture is what tells a scan that failed from a scan that never had a face to
work with. `robot_rppg.py --view` serves one on port 8021:

```sh
ssh bot 'cd /tmp && ~/.local/bin/uv run robot_rppg.py --view --monitor'
```

The left half is the left-eye frame with the model's own working drawn on it:
the detector's raw box, the steadied box the ROI actually used, the three skin
regions it offered, and which pixels inside them survived the skin gate — green
kept, red thrown away. The right half is the signal: the raw ROI green, the
pulse after POS and the bandpass, this window's spectrum, and the stacked
spectrum the reported rate is read off, with both SNRs and the rival margin.
`--monitor` repeats the scan until interrupted, which is the mode to tune in.

`rppg_view.py` draws it, and `laptop_rppg.py` draws through the same module, so
the picture on the robot is the picture on the laptop and a threshold tuned
against one is tuned against the other. The robot dashboard has a tile that
starts the view over SSH and embeds the stream, so it can be watched beside the
other controls.

Two costs are deliberate. The view redraws at 6 fps onto a 480-wide frame,
because the scan loop pays for every frame drawn and a 1280×960 eye composed
and encoded thirty times a second would cost the robot more than the
measurement does. And the page is unauthenticated — it is a debug view, not a
product — so it shows whoever is in front of the robot to anyone who can reach
the port. Serve it on a trusted network, only while someone is watching it, and
stop it afterwards. The dashboard tile stops it when switched off, and the page
dies with the scan process.

## Robot port gate

These steps are read-only apart from copying files into `/tmp`.

1. Copy the app, the signal chain, and the model:

   ```sh
   scp scripts/robot_rppg.py rppg.py rppg_view.py \
       assets/models/face_detection_yunet_2026may.onnx bot:/tmp/
   ```

   `rppg_view.py` is only needed for `--view`, but it is small and shipping it
   with the rest avoids an import error on the robot at the moment someone
   wants to look at why a scan is failing.

2. Check the camera and face gate with a willing person standing at about
   0.7 m, facing the robot:

   ```sh
   ssh bot 'cd /tmp && ~/.local/bin/uv run robot_rppg.py --check'
   ```

   Read `fps` (want ≥ 15), `eye_shape` (want `[960, 1280, 3]`),
   `face_frames` versus `frames`, `face_width_px`, and above all
   `lum_drift_pct` against the table above.

3. Only if the check passes, run a scan while the person holds still:

   ```sh
   ssh bot 'cd /tmp && ~/.local/bin/uv run robot_rppg.py --duration 20'
   ```

   If it reports no reading, or a rate that looks wrong, add `--view` and watch
   the picture rather than guessing from the JSON. The two most common answers
   are visible there directly: a mask sitting on hair or background rather than
   skin, and a stacked spectrum with no line standing out of the floor.

4. Compare against a reference — a smartwatch, a phone pulse app, or a
   MAX30102 — across several people, lighting conditions, and skin tones.
   A single agreeing reading is not validation.

Run it from `/tmp`: in `~`, the `bbos` project folder shadows the `bbos`
package.

## Known limits

- **Skin tone.** rPPG literature reports degraded performance on darker skin,
  because less light returns from the dermis. Validate across skin tones or
  state the limitation prominently.
- **Motion.** The robot must be stationary, holding balance only. Motion
  artefacts sit in the same frequency band as the pulse. The measurement drops
  frames where the nose moves more than 3% of face width and resets its window
  after half a second of motion.
- **High rates.** The second-harmonic check can halve a genuine rate near the
  top of the band if an artefact happens to sit at half of it. See "Which peak
  is the heart rate".
- **Weak signal.** A pulse near the noise floor can still lose to a noise peak.
  Stacking the scan's spectra and requiring the winning line to beat its best
  rival catches most of these as "no clear reading" instead, but not all: on
  the synthetic set one scan in ninety-nine is still read wrong, and it is
  reported as not confident rather than caught. Treat an unconfident reading as
  a failed scan.
- **Refusals are a result.** Making the chain honest about weak scans made it
  refuse more of them — nine of ninety-nine synthetic scans return nothing,
  against five before. A scan that produces no reading is working as intended;
  the thing to fix is the light, the distance, or the exposure, not the gate.
- **Timestamps.** Frames are stamped at read time, not with the camera's own
  `timestamp` field. `resample_uniform` absorbs the jitter, but using the
  camera timestamp would be more correct and is the obvious next refinement.
- **One face.** The landmarker is configured for a single face and the scan
  assumes the nearest one is the subject.
- **Lighting.** Mains flicker at 50/60 Hz aliases into the band if exposure is
  not an integer number of flicker cycles. Even, diffuse, front lighting.

## Consent and safety

The repository's existing vision guidance applies here with more force, because
a heart rate is health-adjacent data.

- Ask before scanning, every time, and make the capture state visible. The
  person must be able to refuse and to stop the scan.
- The script keeps no frames. Only per-frame mean skin RGB is held in memory
  for the length of the analysis window, and nothing is written to disk unless
  someone explicitly records a CSV with the laptop tool.
- A result must reach any assistant as an uncertainty-labelled observation, if
  at all — never as a trigger. The deterministic action gate stays separate.
- The robot must not describe a reading as a health finding, express concern
  about it, or suggest what to do about it. "I measured about 72 beats per
  minute, which cameras only estimate roughly" is the tone; anything about what
  that number means medically is out of bounds.
- Do not use it for screening, access control, stress or deception inference,
  or any decision about a person.
