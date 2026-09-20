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
locks exposure through OpenCV before measuring.

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

## Robot port gate

These steps are read-only apart from copying files into `/tmp`.

1. Copy the app, the signal chain, and the model:

   ```sh
   scp scripts/robot_rppg.py rppg.py assets/models/face_detection_yunet_2026may.onnx bot:/tmp/
   ```

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
