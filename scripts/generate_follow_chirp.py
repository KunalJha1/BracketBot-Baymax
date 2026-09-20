#!/usr/bin/env python3
"""Render the "I'm following you" robot chirp to a WAV asset, once, ahead of time.

Three short sliding blips, synthesised here; no recorded audio is used. Kept
under half a second so the gaps between chirps leave room to hear a command.

    python3 scripts/generate_follow_chirp.py
"""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "bbapps" / "play_sound" / "wavs" / "follow_chirp.wav"
SPEAKER_SAMPLE_RATE = 16000       # the speaker's own rate; the player refuses others
# (start Hz, end Hz, seconds) per blip, separated by short silences.
BLIPS = ((900, 1500, 0.09), (1500, 1100, 0.07), (1200, 1900, 0.11))
GAP_S = 0.04
PEAK = 0.5


def blip(f0: float, f1: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SPEAKER_SAMPLE_RATE)) / SPEAKER_SAMPLE_RATE
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * seconds))
    # A little second harmonic reads as "robot" rather than "test tone".
    tone = np.sin(phase) + 0.3 * np.sin(2 * phase)
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.008)   # no clicks
    return tone * fade / 1.3


def main() -> int:
    gap = np.zeros(int(GAP_S * SPEAKER_SAMPLE_RATE))
    parts = []
    for spec in BLIPS:
        parts += [blip(*spec), gap]
    samples = (np.concatenate(parts[:-1]) * PEAK * 32767).astype("<i2")
    with wave.open(str(OUTPUT), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SPEAKER_SAMPLE_RATE)
        out.writeframes(samples.tobytes())
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
