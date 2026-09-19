"""Local speech recognition and synthesis helpers for the greeter.

The robot hardware I/O remains in ``main.py``.  This module owns only the
provider-free audio transforms: wake-triggered utterance segmentation,
``whisper.cpp`` transcription, and ``espeak-ng`` synthesis.
"""

from __future__ import annotations

from collections import deque
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib import error, request
import wave

import numpy as np


class LocalVoiceError(RuntimeError):
    """A concise, user-safe local voice backend error."""


def _dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-9) / 32768.0)


class SpeechSegmenter:
    """Keep mic pre-roll and end a wake-triggered utterance after silence."""

    def __init__(
        self,
        sample_rate: int,
        *,
        threshold_db: float = -38.0,
        pre_roll_s: float = 1.5,
        min_utterance_s: float = 0.5,
        trailing_silence_s: float = 1.5,
        start_grace_s: float = 0.0,
        max_utterance_s: float = 8.0,
    ):
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.pre_roll_samples = max(0, int(pre_roll_s * sample_rate))
        self.min_samples = max(1, int(min_utterance_s * sample_rate))
        self.silence_samples_required = max(1, int(trailing_silence_s * sample_rate))
        self.start_grace_samples = max(0, int(start_grace_s * sample_rate))
        self.max_samples = max(self.min_samples, int(max_utterance_s * sample_rate))
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_size = 0
        self._recording: list[np.ndarray] | None = None
        self._recording_size = 0
        self._speech_seen = False
        self._silence_samples = 0
        self._after_trigger_samples = 0

    @property
    def recording(self) -> bool:
        return self._recording is not None

    @property
    def speech_seen(self) -> bool:
        return self._speech_seen

    def reset(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_size = 0
        self._recording = None
        self._recording_size = 0
        self._speech_seen = False
        self._silence_samples = 0
        self._after_trigger_samples = 0

    def _remember(self, audio: np.ndarray) -> None:
        self._pre_roll.append(audio)
        self._pre_roll_size += len(audio)
        while self._pre_roll and self._pre_roll_size > self.pre_roll_samples:
            removed = self._pre_roll.popleft()
            self._pre_roll_size -= len(removed)

    def _finish(self) -> np.ndarray:
        assert self._recording is not None
        utterance = np.concatenate(self._recording).astype(np.int16, copy=False)
        self.reset()
        return utterance

    def push(self, audio: np.ndarray, *, triggered: bool = False) -> np.ndarray | None:
        chunk = np.asarray(audio, dtype=np.int16).reshape(-1).copy()
        if not len(chunk):
            return None

        if self._recording is None:
            self._remember(chunk)
            if not triggered:
                return None
            self._recording = list(self._pre_roll)
            self._recording_size = sum(len(part) for part in self._recording)
            self._speech_seen = any(
                _dbfs(part) >= self.threshold_db for part in self._recording
            )
            return None

        self._recording.append(chunk)
        self._recording_size += len(chunk)
        self._after_trigger_samples += len(chunk)
        if _dbfs(chunk) >= self.threshold_db:
            self._speech_seen = True
            self._silence_samples = 0
        elif self._speech_seen:
            self._silence_samples += len(chunk)

        if self._recording_size >= self.max_samples:
            return self._finish()
        if (
            self._speech_seen
            and self._recording_size >= self.min_samples
            and self._after_trigger_samples >= self.start_grace_samples
            and self._silence_samples >= self.silence_samples_required
        ):
            return self._finish()
        return None


class WhisperCppTranscriber:
    """Transcribe mono PCM with the standalone whisper.cpp CLI."""

    DEFAULT_PROMPT = (
        "Hey BracketBot. Baymax. Weather in Waterloo. Wave. Handshake. "
        "Fist bump. Hug. Salute. Dance. Remind me in four minutes to take "
        "my meds. Set a timer for four minutes."
    )

    def __init__(
        self,
        binary: str,
        model: str,
        *,
        language: str = "en",
        threads: int = 4,
        timeout: float = 45.0,
        runner=subprocess.run,
    ):
        self.binary = binary
        self.model = model
        self.language = language
        self.threads = max(1, threads)
        self.timeout = timeout
        self._runner = runner

    def validate(self) -> None:
        if not (Path(self.binary).is_file() or shutil.which(self.binary)):
            raise LocalVoiceError(f"whisper.cpp CLI not found: {self.binary}")
        if not Path(self.model).is_file():
            raise LocalVoiceError(f"Whisper model not found: {self.model}")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.validate()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp_path = Path(temp.name)
                with wave.open(temp, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(np.asarray(audio, dtype=np.int16).tobytes())
            command = [
                self.binary,
                "--model",
                self.model,
                "--file",
                str(temp_path),
                "--language",
                self.language,
                "--threads",
                str(self.threads),
                "--prompt",
                self.DEFAULT_PROMPT,
                "--no-timestamps",
                "--no-prints",
                "--best-of",
                "1",
                "--beam-size",
                "1",
                "--no-fallback",
            ]
            result = self._runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-500:]
                raise LocalVoiceError(
                    f"whisper.cpp exited with {result.returncode}: {detail}"
                )
            return " ".join(result.stdout.split()).strip()
        except subprocess.TimeoutExpired as exc:
            raise LocalVoiceError("Local transcription timed out.") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or not len(audio):
        return np.asarray(audio, dtype=np.int16)
    target_len = max(1, round(len(audio) * target_rate / source_rate))
    source_x = np.arange(len(audio), dtype=np.float64)
    target_x = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(target_x, source_x, audio).clip(-32768, 32767).astype(np.int16)


class EspeakSynthesizer:
    """Render text to mono PCM using the local espeak-ng executable."""

    def __init__(
        self,
        binary: str = "espeak-ng",
        *,
        voice: str = "en-us",
        words_per_minute: int = 165,
        timeout: float = 20.0,
        runner=subprocess.run,
    ):
        self.binary = binary
        self.voice = voice
        self.words_per_minute = words_per_minute
        self.timeout = timeout
        self._runner = runner

    def validate(self) -> None:
        if not (Path(self.binary).is_file() or shutil.which(self.binary)):
            raise LocalVoiceError(f"espeak-ng not found: {self.binary}")

    def synthesize(self, text: str, target_rate: int) -> np.ndarray:
        self.validate()
        try:
            result = self._runner(
                [
                    self.binary,
                    "--stdout",
                    "-v",
                    self.voice,
                    "-s",
                    str(self.words_per_minute),
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LocalVoiceError("Local speech synthesis timed out.") from exc
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise LocalVoiceError(
                f"espeak-ng exited with {result.returncode}: {detail}"
            )
        pcm, source_rate = _decode_wav(result.stdout, "espeak-ng")
        return _resample(pcm, source_rate, target_rate)


def _decode_wav(payload: bytes, provider: str) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            if wav.getsampwidth() != 2:
                raise LocalVoiceError(f"{provider} returned non-16-bit audio.")
            source_rate = wav.getframerate()
            channels = wav.getnchannels()
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    except (EOFError, wave.Error) as exc:
        raise LocalVoiceError(f"{provider} returned an invalid WAV stream.") from exc
    return pcm, source_rate


class HttpTtsSynthesizer:
    """Use the private USB-link TTS service running on the connected Mac."""

    def __init__(self, endpoint: str, *, timeout: float = 20.0, opener=request.urlopen):
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener

    def validate(self) -> None:
        if not self.endpoint.startswith("http://"):
            raise LocalVoiceError("The local TTS endpoint must use HTTP on the USB link.")

    def synthesize(self, text: str, target_rate: int) -> np.ndarray:
        self.validate()
        # Ask the Mac to render at the speaker's own rate.  Rendering at 16 kHz
        # and stretching it here threw away everything above 8 kHz and added
        # interpolation aliasing, which is most of what made the robot sound
        # synthetic.  An older bridge ignores the field and we resample as before.
        payload = {"text": text, "sample_rate": int(target_rate)}
        http_request = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener(http_request, timeout=self.timeout) as response:
                payload = response.read()
        except (error.URLError, TimeoutError, OSError) as exc:
            raise LocalVoiceError(f"Local natural voice unavailable: {exc}") from exc
        pcm, source_rate = _decode_wav(payload, "Local natural voice")
        return _resample(pcm, source_rate, target_rate)


class FallbackSynthesizer:
    """Prefer natural speech but keep the robot able to talk if it is unavailable."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def validate(self) -> None:
        self.primary.validate()
        self.fallback.validate()

    def synthesize(self, text: str, target_rate: int) -> np.ndarray:
        try:
            return self.primary.synthesize(text, target_rate)
        except LocalVoiceError:
            return self.fallback.synthesize(text, target_rate)


def speaker_chunks(
    audio: np.ndarray,
    chunk_size: int,
    channels: int,
    *,
    lead_chunks: int = 4,
) -> list[np.ndarray]:
    """Pad mono PCM for the BBOS speaker jitter buffer."""
    mono = np.asarray(audio, dtype=np.int16).reshape(-1)
    lead = np.zeros(max(0, lead_chunks) * chunk_size, dtype=np.int16)
    padded = np.concatenate([lead, mono])
    padded = np.pad(padded, (0, -len(padded) % chunk_size))
    chunks = []
    for offset in range(0, len(padded), chunk_size):
        chunk = padded[offset : offset + chunk_size]
        if channels > 1:
            chunk = np.repeat(chunk[:, None], channels, axis=1).reshape(-1)
        chunks.append(chunk)
    return chunks


def play_dance_music(writer, path: Path, speaker_cfg, is_dancing, volume=0.55) -> None:
    """Stream the local upbeat WAV while the independently safe dance runs."""
    with wave.open(str(path), "rb") as source:
        if (
            source.getsampwidth() != 2
            or source.getcomptype() != "NONE"
            or source.getframerate() != speaker_cfg.sample_rate
        ):
            raise LocalVoiceError("The dance music format is not supported.")
        source_channels = source.getnchannels()
        period = speaker_cfg.chunk_size / speaker_cfg.sample_rate
        due = time.monotonic()
        loops = 0
        while is_dancing() and loops < 4:
            raw = source.readframes(speaker_cfg.chunk_size)
            if not raw:
                loops += 1
                source.rewind()
                continue
            samples = np.frombuffer(raw, dtype="<i2").reshape(-1, source_channels)
            if source_channels == 1 and speaker_cfg.channels > 1:
                samples = np.repeat(samples, speaker_cfg.channels, axis=1)
            elif source_channels > 1 and speaker_cfg.channels == 1:
                samples = samples.mean(axis=1, dtype=np.float32)[:, None]
            samples = (samples.astype(np.float32) * volume).clip(
                -32768, 32767
            ).astype(np.int16)
            if len(samples) < speaker_cfg.chunk_size:
                padding = np.zeros(
                    (speaker_cfg.chunk_size - len(samples), speaker_cfg.channels),
                    dtype=np.int16,
                )
                samples = np.concatenate((samples, padding))
            with writer.buf() as data:
                data["audio"] = samples
            due += period
            time.sleep(max(0.0, due - time.monotonic()))
