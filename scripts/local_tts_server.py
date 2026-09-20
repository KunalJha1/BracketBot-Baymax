#!/usr/bin/env python3
"""Natural local TTS bridge bound only to BracketBot's private USB link.

Synthesis is cached on disk, keyed by the exact text and voice settings, so a
line the robot has already spoken is served from the cache instead of paying
for ``say`` plus ``afconvert`` again.  Rehearsed lines can be rendered ahead of
time with ``--prewarm`` or ``scripts/generate_line_assets.py``.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile


MAX_TEXT_LENGTH = 4096
# Eddy belongs to the macOS novelty voice family; it is the one that made the
# robot sound synthetic.  Samantha is the highest-quality en_US voice that
# ships by default.  An Enhanced or Premium voice downloaded from System
# Settings > Accessibility > Spoken Content sounds better still and can be
# selected with --voice without any other change.
DEFAULT_VOICE = "Samantha"
# A touch slower than conversational: across a room, through a small speaker,
# the extra gap between words does more for intelligibility than anything else.
DEFAULT_RATE = 160
# No artificial pitch shift: transposing the baseline is what gave the speech
# its chipmunk-like, obviously-machine timbre.
DEFAULT_PITCH = 0
# Render at the speaker's own rate so nothing is band-limited to a telephone
# bandwidth and then stretched back out on the robot.
DEFAULT_SAMPLE_RATE = 48000
SUPPORTED_SAMPLE_RATES = (16000, 22050, 24000, 32000, 44100, 48000)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bracketbot" / "tts"


def synthesize_wav(
    text: str,
    voice: str,
    rate: int,
    pitch: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """Render one line to mono PCM WAV with the macOS speech engine."""
    with tempfile.TemporaryDirectory(prefix="bracketbot-tts-") as temp_dir:
        source = Path(temp_dir) / "speech.aiff"
        output = Path(temp_dir) / "speech.wav"
        speech = f"[[pbas {pitch:+d}]] {text}" if pitch else text
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", source, speech],
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
                f"LEI16@{sample_rate}",
                "-c",
                "1",
                # Mastering-grade sample rate conversion; the robot's own
                # linear-interpolation resampler aliases audibly.
                "--src-complexity",
                "bats",
                "-q",
                "127",
                source,
                output,
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return output.read_bytes()


class SpeechCache:
    """Content-addressed WAV cache that degrades to no caching on any error."""

    def __init__(self, directory: str | Path | None):
        self.directory = Path(directory).expanduser() if directory else None
        self.enabled = self.directory is not None
        if self.enabled:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                print(f"[tts] cache disabled: {exc}", flush=True)
                self.enabled = False

    @staticmethod
    def key(text: str, voice: str, rate: int, pitch: int, sample_rate: int) -> str:
        identity = json.dumps(
            [text, voice, int(rate), int(pitch), int(sample_rate)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _path(
        self, text: str, voice: str, rate: int, pitch: int, sample_rate: int
    ) -> Path:
        return (
            self.directory
            / f"{self.key(text, voice, rate, pitch, sample_rate)}.wav"
        )

    def get(
        self, text: str, voice: str, rate: int, pitch: int, sample_rate: int
    ) -> bytes | None:
        if not self.enabled:
            return None
        try:
            return self._path(text, voice, rate, pitch, sample_rate).read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            print(f"[tts] cache read failed: {exc}", flush=True)
            return None

    def put(
        self,
        text: str,
        voice: str,
        rate: int,
        pitch: int,
        sample_rate: int,
        audio: bytes,
    ) -> None:
        if not self.enabled or not audio:
            return
        target = self._path(text, voice, rate, pitch, sample_rate)
        # Write through a sibling temporary file so a concurrent reader never
        # sees a partial WAV.
        try:
            temporary = target.with_suffix(".wav.partial")
            temporary.write_bytes(audio)
            temporary.replace(target)
        except OSError as exc:
            print(f"[tts] cache write failed: {exc}", flush=True)


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
            # The robot asks for its own speaker rate so the audio never has to
            # be resampled again after it crosses the USB link.
            requested = body.get("sample_rate")
            sample_rate = self.server.sample_rate
            if requested is not None:
                sample_rate = int(requested)
                if sample_rate not in SUPPORTED_SAMPLE_RATES:
                    raise ValueError(
                        "sample_rate must be one of "
                        + ", ".join(str(rate) for rate in SUPPORTED_SAMPLE_RATES)
                    )
            audio = self.server.speak(text, sample_rate)
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
    def __init__(
        self,
        address,
        voice: str,
        rate: int,
        pitch: int = DEFAULT_PITCH,
        cache: SpeechCache | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        super().__init__(address, TtsHandler)
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.sample_rate = sample_rate
        self.cache = cache if cache is not None else SpeechCache(None)

    def speak(self, text: str, sample_rate: int | None = None) -> bytes:
        """Return spoken audio for one line, reusing the cache when possible."""
        sample_rate = sample_rate or self.sample_rate
        cached = self.cache.get(text, self.voice, self.rate, self.pitch, sample_rate)
        if cached is not None:
            print(f"[tts] cache hit ({len(cached)} bytes): {text[:60]!r}", flush=True)
            return cached
        audio = self.synthesize(text, sample_rate)
        self.cache.put(text, self.voice, self.rate, self.pitch, sample_rate, audio)
        return audio

    def synthesize(self, text: str, sample_rate: int | None = None) -> bytes:
        return synthesize_wav(
            text, self.voice, self.rate, self.pitch, sample_rate or self.sample_rate
        )


def prewarm(server: TtsServer, path: Path) -> int:
    """Render every non-empty line of a text file into the cache."""
    rendered = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if (
            server.cache.get(
                line, server.voice, server.rate, server.pitch, server.sample_rate
            )
            is not None
        ):
            continue
        try:
            audio = server.synthesize(line)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[tts] prewarm failed for {line[:60]!r}: {exc}", flush=True)
            continue
        server.cache.put(
            line,
            server.voice,
            server.rate,
            server.pitch,
            server.sample_rate,
            audio,
        )
        rendered += 1
    return rendered


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
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=SUPPORTED_SAMPLE_RATES,
        default=DEFAULT_SAMPLE_RATE,
        help=f"fallback render rate when a client asks for none "
        f"(default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"spoken audio cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="synthesize every request instead of reusing cached audio",
    )
    parser.add_argument(
        "--prewarm",
        metavar="FILE",
        help="render each line of this text file into the cache before serving",
    )
    args = parser.parse_args()

    cache = SpeechCache(None if args.no_cache else args.cache_dir)
    server = TtsServer(
        (args.host, args.port),
        args.voice,
        args.rate,
        args.pitch,
        cache,
        args.sample_rate,
    )
    if args.prewarm:
        count = prewarm(server, Path(args.prewarm).expanduser())
        print(f"Prewarmed {count} new line(s) into {cache.directory}", flush=True)
    print(
        f"Natural voice ready on http://{args.host}:{args.port}/tts "
        f"(voice={args.voice}, rate={args.rate}, pitch={args.pitch:+d}, "
        f"sample_rate={args.sample_rate}, "
        f"cache={'off' if not cache.enabled else cache.directory})",
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
