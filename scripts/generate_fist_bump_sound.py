#!/usr/bin/env python3
"""Render the fist bump's "ba-la-la-la-la" to a WAV asset, once, ahead of time.

Spoken with the robot's own TTS voice (macOS ``say``), falling in pitch the way
the pulled-back, finger-wiggling hand does. No film audio is used.

    python3 scripts/generate_fist_bump_sound.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_tts_server import DEFAULT_VOICE, synthesize_wav  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "bbapps" / "play_sound" / "wavs" / "fist_bump_balalala.wav"
SPEAKER_SAMPLE_RATE = 16000       # the speaker's own rate; the player refuses others
# Each syllable steps the pitch baseline down for the falling sing-song.
TEXT = "Bah " + " ".join(f"[[pbas {pitch}]] la" for pitch in (52, 50, 48, 46, 44)) + "."


def main() -> int:
    OUTPUT.write_bytes(synthesize_wav(TEXT, DEFAULT_VOICE, 220, 0, SPEAKER_SAMPLE_RATE))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
