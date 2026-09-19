#!/usr/bin/env python3
"""Natural local TTS bridge bound only to BracketBot's private USB link."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile


MAX_TEXT_LENGTH = 4096
DEFAULT_VOICE = "Eddy (English (US))"
DEFAULT_RATE = 178
DEFAULT_PITCH = 4


class TtsHandler(BaseHTTPRequestHandler):
    server_version = "BracketBotTTS/1.0"

    def do_POST(self) -> None:
        if self.path != "/tts":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("invalid request size")
            body = json.loads(self.rfile.read(length))
            text = str(body.get("text", "")).strip()
            if not text or len(text) > MAX_TEXT_LENGTH:
                raise ValueError("text must contain 1 to 4096 characters")
            audio = self.server.synthesize(text)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_error(400, str(exc))
            return
        except (OSError, subprocess.SubprocessError) as exc:
            self.send_error(503, f"speech generation failed: {exc}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def log_message(self, format: str, *args) -> None:
        print(f"[tts] {self.address_string()} {format % args}", flush=True)


class TtsServer(ThreadingHTTPServer):
    def __init__(self, address, voice: str, rate: int, pitch: int = DEFAULT_PITCH):
        super().__init__(address, TtsHandler)
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    def synthesize(self, text: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="bracketbot-tts-") as temp_dir:
            source = Path(temp_dir) / "speech.aiff"
            output = Path(temp_dir) / "speech.wav"
            speech = f"[[pbas {self.pitch:+d}]] {text}" if self.pitch else text
            subprocess.run(
                ["say", "-v", self.voice, "-r", str(self.rate), "-o", source, speech],
                check=True,
                capture_output=True,
                timeout=20,
            )
            subprocess.run(
                [
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@16000",
                    "-c",
                    "1",
                    source,
                    output,
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
            return output.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="BracketBot private-link TTS bridge")
    parser.add_argument("--host", default="192.168.55.100")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"installed macOS voice (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_RATE,
        help=f"speaking rate in words per minute (default: {DEFAULT_RATE})",
    )
    parser.add_argument(
        "--pitch",
        type=int,
        choices=range(-10, 11),
        default=DEFAULT_PITCH,
        metavar="{-10..10}",
        help=f"relative baseline pitch adjustment (default: +{DEFAULT_PITCH})",
    )
    args = parser.parse_args()

    server = TtsServer((args.host, args.port), args.voice, args.rate, args.pitch)
    print(
        f"Natural voice ready on http://{args.host}:{args.port}/tts "
        f"(voice={args.voice}, rate={args.rate}, pitch={args.pitch:+d})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
