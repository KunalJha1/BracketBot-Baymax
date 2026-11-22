"""Synthetic accuracy benchmark for the rPPG chain. Runs anywhere: no camera.

    python scripts/rppg_benchmark.py            # the signal chain, from mean-RGB traces
    python scripts/rppg_benchmark.py --roi      # the ROI stage too, from rendered frames

Generates mean-ROI-RGB traces shaped like what the head camera actually delivers - a
PPG waveform with a strong second harmonic, slow illumination drift, auto-exposure
steps, low-frequency motion artefacts, dropped frames, and a pulse near the noise
floor - then scores what HeartRateTracker makes of them against the rate that went in.

This measures the signal chain, not the camera, the landmarker, or a real pulse. It
is how the thresholds in rppg.py were chosen and how a change to them is checked; it
is NOT validation of the feature, which needs a reference device and real people.
See docs/rppg-robot-port.md.

`--roi` renders frames instead of traces - a pulsing face with hair across the
forehead, a specular highlight on one cheek, and a detector whose box lands a
pixel or two off every frame - and runs them through FaceROI. That is the half
of the accuracy the trace benchmark cannot see: which pixels get averaged.

Deps: numpy, scipy, opencv-python for --roi.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rppg import HeartRateTracker, MIN_SECONDS  # noqa: E402


def make_trace(bpm, *, seconds=20.0, fps=30.0, seed=0, pulse_amp=0.004, harmonic=0.7,
               rsa_bpm=2.0, drift_pct=0.0, ae_steps=0, motion=0.0, drop_rate=0.0,
               noise=0.15, gaps=(), ae_hunt=0.0, roi_jitter=0.0, flicker_beat=0.0):
    """Return (t, rgb, true_bpm) for one synthetic scan.

    harmonic:  second-harmonic amplitude relative to the fundamental. Real PPG has a
               dicrotic notch, and on video the harmonic is often the taller line.
    rsa_bpm:   how far the instantaneous rate wanders (respiratory sinus arrhythmia).
    drift_pct: peak-to-peak slow illumination drift, percent of mean.
    ae_steps:  abrupt exposure steps of 0.5-2 % each.
    motion:    amplitude of a 0.2-0.6 Hz motion artefact, relative.
    drop_rate: fraction of frames dropped at random.
    gaps:      ((start_s, length_s), ...) hard holes, as the motion gate leaves.
    ae_hunt:   peak-to-peak percent of a continuous ~0.15 Hz auto-exposure oscillation.
               The robot cannot lock the head camera's exposure, so AE correction is a
               standing wave under the pulse, not the occasional step a laptop sees.
    roi_jitter: common-mode multiplicative noise from a detector box that moves a pixel
               or two per frame, changing which skin the mean averages over.
    flicker_beat: amplitude of a slow beat left by mains flicker that the exposure time
               does not divide evenly. It lands in the pulse band and looks like a peak.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.sort(np.arange(n) / fps + rng.uniform(-0.3 / fps, 0.3 / fps, n))
    t -= t[0]

    f0 = bpm / 60.0
    rsa = (rsa_bpm / 60.0) * np.sin(2 * np.pi * 0.25 * t)     # zero-mean over the scan
    phase = 2 * np.pi * np.cumsum(np.r_[0.0, np.diff(t)] * (f0 + rsa))
    pulse = np.sin(phase) + harmonic * np.sin(2 * phase + 0.7)
    pulse /= np.abs(pulse).max()

    base = np.array([175.0, 128.0, 112.0])
    weights = np.array([0.35, 1.0, 0.45])                     # green carries most of it
    sig = 1.0 + pulse_amp * pulse[:, None] * weights

    illum = np.ones(n)
    if drift_pct:
        illum *= 1 + (drift_pct / 200.0) * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6))
    if ae_hunt:
        illum *= 1 + (ae_hunt / 200.0) * np.sin(2 * np.pi * 0.15 * t + rng.uniform(0, 6))
    for _ in range(ae_steps):
        k = rng.integers(int(0.1 * n), int(0.9 * n))
        illum[k:] *= 1 + rng.choice([-1, 1]) * rng.uniform(0.005, 0.02)
    if flicker_beat:
        illum *= 1 + flicker_beat * np.sin(2 * np.pi * 1.9 * t + rng.uniform(0, 6))

    if motion:
        f = np.fft.rfftfreq(n, 1 / fps)
        m = np.fft.irfft(np.fft.rfft(rng.normal(0, 1, n)) * ((f > 0.2) & (f < 0.6)), n)
        sig *= 1 + motion * (m / (np.abs(m).max() + 1e-12))[:, None] * np.array([1.0, 0.9, 1.1])

    if roi_jitter:
        # A box that wanders by a pixel or two re-picks the averaged skin every frame:
        # all three channels move together, with no preferred frequency.
        illum = illum * (1 + roi_jitter * rng.normal(0, 1, n))

    rgb = base * sig * illum[:, None] + rng.normal(0, noise, (n, 3))

    keep = rng.random(n) > drop_rate if drop_rate else np.ones(n, bool)
    for start, length in gaps:
        keep &= ~((t >= start) & (t < start + length))
    return t[keep], rgb[keep], 60.0 * f0


def scan(t, rgb, period=1.0, **kw):
    """One estimate a second over the trace, the way a real scan paces itself."""
    tracker = HeartRateTracker(**kw)
    next_est = t[0] + MIN_SECONDS
    for now, sample in zip(t, rgb):
        tracker.add(now, sample)
        if now >= next_est:
            next_est = now + period
            tracker.update(now)
    return tracker.result()


RATES = [52, 58, 63, 72, 78, 88, 96, 108, 124]
FAMILIES = {
    "clean": lambda i: dict(seed=i),
    "harmonic-heavy": lambda i: dict(seed=i + 10, harmonic=1.6),
    "ae-steps": lambda i: dict(seed=i + 20, ae_steps=2, drift_pct=1.5),
    "motion": lambda i: dict(seed=i + 30, motion=0.004, drop_rate=0.05),
    "gaps": lambda i: dict(seed=i + 40, gaps=((7.0, 1.5), (13.0, 2.0)), drop_rate=0.02),
    "dim": lambda i: dict(seed=i + 50, pulse_amp=0.0018, noise=0.25),
    # Below here: what the robot head camera adds on top. Its exposure cannot be
    # locked, its face boxes wander, and a face at a metre is small and dim.
    "ae-hunting": lambda i: dict(seed=i + 60, ae_hunt=1.2, ae_steps=1),
    "roi-jitter": lambda i: dict(seed=i + 70, roi_jitter=0.0035),
    "low-fps": lambda i: dict(seed=i + 80, fps=15.0, noise=0.2),
    "far-face": lambda i: dict(seed=i + 90, pulse_amp=0.0022, noise=0.45, roi_jitter=0.0015),
    "flicker": lambda i: dict(seed=i + 100, flicker_beat=0.0012),
}


# ----------------------------------------------------------------------------
# ROI stage: rendered frames through FaceROI, not traces
# ----------------------------------------------------------------------------

FRAME_W, FRAME_H = 640, 480
SKIN_BGR = np.array([112.0, 128.0, 175.0])
# Per-channel pulse depth, in the frame's BGR order. Haemoglobin absorbs green
# hardest, so the three channels do NOT move by the same fraction - and POS works
# by projecting out exactly the common-mode part that a flat modulation would be.
# Rendering the pulse equally into all three channels makes POS cancel it.
PULSE_WEIGHTS_BGR = np.array([0.45, 1.0, 0.35])


class JitteringDetector:
    """A detector that finds the face every time but never puts the box in the same place.

    This is the honest part of a real detector: YuNet re-runs from scratch on
    every frame, so even a person holding perfectly still gets a box that moves
    a pixel or two. The ROI then averages slightly different skin each frame,
    and that wobble lands in the pulse band.
    """

    def __init__(self, box, rng, jitter_px=2.0):
        self.box = np.asarray(box, dtype=float)
        self.rng = rng
        self.jitter_px = jitter_px

    def setInputSize(self, size):
        self.size = size

    def detect(self, frame):
        scale = frame.shape[1] / FRAME_W
        x, y, w, h = self.box * scale
        dx, dy = self.rng.normal(0, self.jitter_px * scale, 2)
        row = [x + dx, y + dy, w, h,
               x + 0.3 * w, y + 0.4 * h, x + 0.7 * w, y + 0.4 * h,      # eyes
               x + 0.5 * w + dx, y + 0.55 * h + dy,                     # nose
               x + 0.35 * w, y + 0.78 * h, x + 0.65 * w, y + 0.78 * h,  # mouth corners
               0.99]
        return None, np.asarray([row], dtype=np.float32)


def skin_shading(box, rng, shade=0.12, speckle=0.03):
    """A static brightness field over the face: a real face is not one flat colour.

    This is what makes a wandering detector box cost anything. On a flat patch,
    a box that moves two pixels re-averages the identical colour and the wobble
    is free; on a face that is brighter at the forehead than the jaw and
    speckled at the pixel level, the same two pixels change the mean, and that
    change lands in the pulse band. Leaving it out is what made an earlier
    version of this benchmark unable to see the ROI stage at all.
    """
    import cv2

    x, y, w, h = (float(v) for v in box)
    rows, cols = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float32)
    down = np.clip((rows - y) / max(h, 1.0), 0.0, 1.0)
    across = np.clip((cols - x) / max(w, 1.0), 0.0, 1.0)
    field = 1.0 + shade * (0.5 - down) + 0.4 * shade * (across - 0.5)
    grain = rng.normal(0.0, speckle, (FRAME_H, FRAME_W)).astype(np.float32)
    return field + cv2.GaussianBlur(grain, (0, 0), 1.2)


def face_frame(base, box, pulse, rng, shading, pulse_amp=0.004, noise=0.8,
               hair=True, highlight=True):
    """One frame: skin whose colour carries the pulse, plus what is not skin.

    The contaminants are the ones that matter on the robot. Hair over the
    forehead and a blown-out highlight on a cheek sit inside the ROI polygons,
    carry no pulse at all, and dilute a signal that is under 1% of the
    brightness to begin with.
    """
    import cv2

    frame = base.copy()
    x, y, w, h = (int(v) for v in box)
    cv2.ellipse(frame, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360,
                tuple(SKIN_BGR * (1 + pulse_amp * pulse * PULSE_WEIGHTS_BGR)), -1)
    y0, y1 = max(0, y - 4), min(FRAME_H, y + h + 4)
    x0, x1 = max(0, x - 4), min(FRAME_W, x + w + 4)
    patch = frame[y0:y1, x0:x1].astype(np.float32) * shading[y0:y1, x0:x1, None]
    frame[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    if hair:
        for k in range(6):
            cx = x + int(w * (0.26 + 0.09 * k))
            cv2.line(frame, (cx, y + int(0.08 * h)), (cx + 3, y + int(0.34 * h)), (28, 30, 38), 2)
    if highlight:
        cv2.circle(frame, (x + int(0.30 * w), y + int(0.55 * h)), max(2, int(0.08 * w)),
                   (248, 250, 252), -1)
    # Sensor noise, over the face only: the background is never averaged, and paying
    # for noise across the whole frame is what makes this benchmark too slow to run.
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    patch += rng.normal(0, noise, patch.shape)
    frame[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return frame


def roi_scan(true_bpm, face_w, *, steady, skin_gate, seconds=20.0, fps=30.0, seed=0,
             harmonic=0.7):
    """Render a scan and run it through FaceROI and HeartRateTracker, as the robot does."""
    import rppg

    rng = np.random.default_rng(seed)
    box = (FRAME_W // 2 - face_w // 2, 90, face_w, int(face_w * 1.3))
    base = np.full((FRAME_H, FRAME_W, 3), 60, np.uint8)
    shading = skin_shading(box, np.random.default_rng(seed + 1000))
    roi = rppg.FaceROI(detector=JitteringDetector(box, rng), steady=steady, skin_gate=skin_gate)
    tracker = HeartRateTracker()
    next_est = MIN_SECONDS
    f0 = true_bpm / 60.0
    for i in range(int(seconds * fps)):
        t = i / fps
        pulse = np.sin(2 * np.pi * f0 * t) + harmonic * np.sin(4 * np.pi * f0 * t + 0.7)
        out = roi(face_frame(base, box, pulse, rng, shading), t * 1000)
        if out is None:
            continue
        tracker.add(t, out[0])
        if t >= next_est:
            next_est = t + 1.0
            tracker.update(t)
    return tracker.result()


ROI_RATES = [58, 72, 96]
ROI_VARIANTS = (("raw box, every pixel", False, False),
                ("steadied box only", True, False),
                ("skin gate only", False, True),
                ("steadied box + skin gate", True, True))


def roi_main():
    """Score the ROI stage by the signal it hands the chain, not only by the rate.

    On a clean synthetic scan every variant gets the rate right, so BPM error
    alone says very little: it reads near zero until a scan fails outright, and
    then jumps. What the ROI stage controls is how far the pulse stands above
    the noise in the trace it produces, and that is what decides whether a
    marginal real scan - a face a metre away under a ceiling light - lands on
    the reading side of the gate or the "no clear reading" side.
    """
    print("ROI stage: rendered frames through FaceROI (face width in the 640x480 frame)")
    print("Scan SNR is the stacked spectrum's: higher means more margin before a real")
    print("scan stops producing a reading at all.")
    print()
    print(f"{'variant':<28}{'face 130px':>24}{'face 70px':>24}")
    rows = []
    for label, steady, gate in ROI_VARIANTS:
        cells, snrs = [], []
        for face_w in (130, 70):
            errors, scan_snrs, misses = [], [], 0
            for seed, bpm in enumerate(ROI_RATES, start=1):
                result = roi_scan(bpm, face_w, steady=steady, skin_gate=gate, seed=seed)
                if result is None:
                    misses += 1
                else:
                    errors.append(abs(result["bpm"] - bpm))
                    scan_snrs.append(result["snr_db"])
            snrs.append(float(np.mean(scan_snrs)) if scan_snrs else float("nan"))
            cell = (f"{np.mean(errors):.2f} BPM, {snrs[-1]:+.1f} dB" if errors
                    else "no reading at all")
            if misses and errors:
                cell += f", {misses} miss"
            cells.append(cell)
        rows.append(snrs)
        print(f"{label:<28}{cells[0]:>24}{cells[1]:>24}")
    base, best = rows[0], rows[-1]
    print()
    print(f"Steadying the box and gating the skin together move the scan SNR by "
          f"{best[0] - base[0]:+.1f} dB at 130 px")
    print(f"and {best[1] - base[1]:+.1f} dB at 70 px. Mean over {len(ROI_RATES)} rates. "
          f"Synthetic frames, not a real camera:")
    print("see docs/rppg-robot-port.md.")


def main():
    errors, misses, confident_wrong = [], 0, []
    print(f"{'family':<16}{'MAE':>8}{'worst':>8}{'misses':>8}{'confidently wrong':>20}")
    for family, kwargs in FAMILIES.items():
        fam_err, fam_miss, fam_bad = [], 0, 0
        for i, bpm in enumerate(RATES):
            t, rgb, true_bpm = make_trace(bpm=bpm, **kwargs(i))
            result = scan(t, rgb)
            if result is None:
                fam_miss += 1
                continue
            err = abs(result["bpm"] - true_bpm)
            fam_err.append(err)
            if result["confident"] and err > 5:
                fam_bad += 1
                confident_wrong.append((family, true_bpm, result))
        errors += fam_err
        misses += fam_miss
        print(f"{family:<16}{np.mean(fam_err):8.2f}{np.max(fam_err):8.2f}"
              f"{fam_miss:8d}{fam_bad:20d}")

    e = np.array(errors)
    print(f"\n{len(e)} scans, {misses} with no reading at all")
    print(f"MAE {e.mean():.2f} BPM   median {np.median(e):.2f}   p90 {np.percentile(e, 90):.2f}"
          f"   max {e.max():.2f}")
    print(f"within 3 BPM {100 * np.mean(e < 3):.1f}%   within 5 BPM {100 * np.mean(e < 5):.1f}%"
          f"   confidently wrong {len(confident_wrong)}")
    for family, true_bpm, result in confident_wrong:
        print(f"  {family}: said {result['bpm']:.1f} for {true_bpm:.0f} BPM, "
              f"SNR {result['snr_db']:+.1f} dB over {result['span_s']} s")
    print("\nSynthetic only. Not a validation of the feature: see docs/rppg-robot-port.md.")


if __name__ == "__main__":
    if "--roi" in sys.argv:
        roi_main()
    else:
        main()
