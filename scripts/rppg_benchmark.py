"""Synthetic accuracy benchmark for the rPPG signal chain. Runs anywhere: no camera.

    python scripts/rppg_benchmark.py

Generates mean-ROI-RGB traces shaped like what the head camera actually delivers - a
PPG waveform with a strong second harmonic, slow illumination drift, auto-exposure
steps, low-frequency motion artefacts, dropped frames, and a pulse near the noise
floor - then scores what HeartRateTracker makes of them against the rate that went in.

This measures the signal chain, not the camera, the landmarker, or a real pulse. It
is how the thresholds in rppg.py were chosen and how a change to them is checked; it
is NOT validation of the feature, which needs a reference device and real people.
See docs/rppg-robot-port.md.

Deps: numpy, scipy (no mediapipe, no OpenCV).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rppg import HeartRateTracker, MIN_SECONDS  # noqa: E402


def make_trace(bpm, *, seconds=20.0, fps=30.0, seed=0, pulse_amp=0.004, harmonic=0.7,
               rsa_bpm=2.0, drift_pct=0.0, ae_steps=0, motion=0.0, drop_rate=0.0,
               noise=0.15, gaps=()):
    """Return (t, rgb, true_bpm) for one synthetic scan.

    harmonic:  second-harmonic amplitude relative to the fundamental. Real PPG has a
               dicrotic notch, and on video the harmonic is often the taller line.
    rsa_bpm:   how far the instantaneous rate wanders (respiratory sinus arrhythmia).
    drift_pct: peak-to-peak slow illumination drift, percent of mean.
    ae_steps:  abrupt exposure steps of 0.5-2 % each.
    motion:    amplitude of a 0.2-0.6 Hz motion artefact, relative.
    drop_rate: fraction of frames dropped at random.
    gaps:      ((start_s, length_s), ...) hard holes, as the motion gate leaves.
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
    for _ in range(ae_steps):
        k = rng.integers(int(0.1 * n), int(0.9 * n))
        illum[k:] *= 1 + rng.choice([-1, 1]) * rng.uniform(0.005, 0.02)

    if motion:
        f = np.fft.rfftfreq(n, 1 / fps)
        m = np.fft.irfft(np.fft.rfft(rng.normal(0, 1, n)) * ((f > 0.2) & (f < 0.6)), n)
        sig *= 1 + motion * (m / (np.abs(m).max() + 1e-12))[:, None] * np.array([1.0, 0.9, 1.1])

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
}


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
    main()
