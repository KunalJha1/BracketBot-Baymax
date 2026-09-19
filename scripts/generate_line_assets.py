#!/usr/bin/env python3
"""Render BracketBot's fixed spoken lines to WAV assets, once, ahead of time.

Each line in ``scripts/canned_lines.py`` becomes
``assets/lines/<id>.wav`` in the same 16 kHz mono format as the existing sound
cues, so the dashboard's allowlisted sound path can play it with no network,
no speech recognition, and no model call.  The same audio is written into the
TTS bridge's cache, so a live spoken reply with identical text is free too.

Requires macOS ``say`` and ``afconvert`` (the same engine as the TTS bridge).

    python3 scripts/generate_line_assets.py
    python3 scripts/generate_line_assets.py --force --voice "Reed (English (US))"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canned_lines import LINES, LINES_DIR, ROOT  # noqa: E402
from local_tts_server import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_PITCH,
    DEFAULT_RATE,
    DEFAULT_VOICE,
    SpeechCache,
    synthesize_wav,
)


MANIFEST = LINES_DIR / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE)
    parser.add_argument(
        "--pitch", type=int, choices=range(-10, 11), default=DEFAULT_PITCH,
        metavar="{-10..10}",
    )
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE_DIR),
        help="also seed this TTS cache directory (use 'off' to skip)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-render lines whose recorded text and voice already match",
    )
    args = parser.parse_args()

    LINES_DIR.mkdir(parents=True, exist_ok=True)
    cache = SpeechCache(None if args.cache_dir.lower() == "off" else args.cache_dir)
    try:
        previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = {}

    entries: dict[str, dict[str, object]] = {}
    for line in LINES:
        recorded = previous.get(line.id, {}) if isinstance(previous, dict) else {}
        unchanged = (
            not args.force
            and line.path.is_file()
            and recorded.get("text") == line.text
            and recorded.get("voice") == args.voice
            and recorded.get("rate") == args.rate
            and recorded.get("pitch") == args.pitch
        )
        if unchanged:
            audio = line.path.read_bytes()
            status = "kept"
        else:
            try:
                audio = synthesize_wav(line.text, args.voice, args.rate, args.pitch)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"{line.id}: synthesis failed: {exc}", file=sys.stderr)
                return 1
            line.path.write_bytes(audio)
            status = "rendered"
        cache.put(line.text, args.voice, args.rate, args.pitch, audio)
        entries[line.id] = {
            "file": line.filename,
            "text": line.text,
            "voice": args.voice,
            "rate": args.rate,
            "pitch": args.pitch,
            "bytes": len(audio),
            "sha256": hashlib.sha256(audio).hexdigest(),
        }
        print(f"{status:9} {line.path.relative_to(ROOT)}  ({len(audio)} bytes)")

    MANIFEST.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"manifest  {MANIFEST.relative_to(ROOT)}")
    if cache.enabled:
        print(f"tts cache {cache.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
