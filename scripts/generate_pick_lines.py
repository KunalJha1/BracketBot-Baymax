"""Render the pick runner's spoken lines to WAVs the robot speaker can play.

Same engine, voice and speaker rate as ``generate_line_assets.py`` (macOS
``say``); ``pick_lab.sh`` copies the results to the robot and
``pick_object.speak`` plays them. Run again after editing PICK_LINES:

    python3 scripts/generate_pick_lines.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_tts_server import DEFAULT_PITCH, DEFAULT_RATE, DEFAULT_VOICE, synthesize_wav  # noqa: E402

SPEAKER_SAMPLE_RATE = 16000
OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "pick"

# id -> what the robot says. The ids are what pick_object.speak() is given.
PICK_LINES = {
    "starting": "On it. Grabbing the can.",
    "next": "Got it. Next one.",
    "retry": "Missed that one. Let me look again.",
    "out-of-reach": "Yo, the other cans are out of reach. Slide them closer and I'll grab them.",
    "none": "I don't see a can I can reach.",
    "done": "All done. Everything is in the box.",
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in PICK_LINES.items():
        path = OUT_DIR / f"{name}.wav"
        path.write_bytes(synthesize_wav(text, DEFAULT_VOICE, DEFAULT_RATE, DEFAULT_PITCH,
                                        SPEAKER_SAMPLE_RATE))
        print(f"{path.name}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
