"""Local speech recognition and synthesis helpers for the greeter.

The robot hardware I/O remains in ``main.py``.  This module owns only the
provider-free audio transforms: wake-triggered utterance segmentation,
``whisper.cpp`` transcription, and ``espeak-ng`` synthesis.
"""

from __future__ import annotations

from collections import deque
import hashlib
import io
import json
from pathlib import Path
import re
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
        level_gain: float = 1.0,
        noise_margin_db: float = 8.0,
        peak_drop_db: float = 15.0,
        wake_tail_s: float = 0.3,
        no_speech_timeout_s: float = 5.0,
        tail_only_wait_s: float = 2.5,
    ):
        self.sample_rate = sample_rate
        # The end of the wake phrase lands in the first chunks of the turn.
        # Counted as speech, it let the pause people leave while waiting for
        # the listening light end the turn 1.0 s in, before the question, and
        # whisper then invented one from its prompt. Only sound after this
        # window opens the utterance; an empty turn gives up on its own.
        self.wake_tail_samples = max(0, int(wake_tail_s * sample_rate))
        self.no_speech_samples = max(1, int(no_speech_timeout_s * sample_rate))
        self.tail_only_samples = max(1, int(tail_only_wait_s * sample_rate))
        self.last_had_speech = False
        self._tail_speech = False
        self.threshold_db = threshold_db
        # A fixed threshold only works in a quiet room. With fans, motors and
        # other people talking the quietest chunk measured on the robot was
        # -34 dB against a -38 dB threshold, so no pause ever counted as
        # silence and every turn ran to the max-utterance cap. The threshold
        # therefore rises to sit above the room's idle noise floor and within
        # ``peak_drop_db`` of the loudest speech heard this turn.
        self.noise_margin_db = noise_margin_db
        self.peak_drop_db = peak_drop_db
        self._idle_levels: deque[float] = deque(maxlen=50)
        self._floor_db: float | None = None
        self._peak_db: float | None = None
        # Voice detection needs the boosted level, but the stored samples stay
        # unboosted: multiplying the audio itself clipped every close-up
        # utterance before whisper ever saw it, and clipping costs far more
        # accuracy than a quiet recording does.
        self.level_gain = max(1e-6, float(level_gain))
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
        self._loud_chunks = 0

    @property
    def recording(self) -> bool:
        return self._recording is not None

    @property
    def speech_seen(self) -> bool:
        return self._speech_seen

    @property
    def loud_chunks(self) -> int:
        """Speech chunks so far; unchanged means nothing new was said."""
        return self._loud_chunks

    def speculation_ready(self, after_s: float) -> bool:
        """True once the turn would end if the current pause simply continues.

        Transcribing at this point overlaps the recognizer with the rest of
        the trailing-silence wait instead of starting it afterwards.
        """
        return (
            self._recording is not None
            and self._speech_seen
            and self._recording_size >= self.min_samples
            and self._after_trigger_samples >= self.start_grace_samples
            and self._silence_samples >= int(after_s * self.sample_rate)
        )

    def snapshot(self) -> np.ndarray:
        assert self._recording is not None
        return np.concatenate(self._recording).astype(np.int16, copy=False)

    @property
    def active_threshold_db(self) -> float:
        threshold = self.threshold_db
        if self._floor_db is not None:
            threshold = max(threshold, self._floor_db + self.noise_margin_db)
        if self._peak_db is not None:
            threshold = max(threshold, self._peak_db - self.peak_drop_db)
        return threshold

    def reset(self) -> None:
        self._floor_db = None
        self._peak_db = None
        self._loud_chunks = 0
        self._pre_roll.clear()
        self._pre_roll_size = 0
        self._recording = None
        self._recording_size = 0
        self._speech_seen = False
        self._tail_speech = False
        self._silence_samples = 0
        self._after_trigger_samples = 0

    def _remember(self, audio: np.ndarray) -> None:
        self._pre_roll.append(audio)
        self._pre_roll_size += len(audio)
        while self._pre_roll and self._pre_roll_size > self.pre_roll_samples:
            removed = self._pre_roll.popleft()
            self._pre_roll_size -= len(removed)

    def level_dbfs(self, audio: np.ndarray) -> float:
        """The chunk's level as the detector sees it, with input gain applied."""
        return _dbfs(np.asarray(audio, dtype=np.int16).reshape(-1)) + 20.0 * np.log10(
            self.level_gain
        )

    def _finish(self) -> np.ndarray:
        assert self._recording is not None
        utterance = np.concatenate(self._recording).astype(np.int16, copy=False)
        self.last_had_speech = self._speech_seen or self._tail_speech
        self.reset()
        return utterance

    def push(self, audio: np.ndarray, *, triggered: bool = False) -> np.ndarray | None:
        chunk = np.asarray(audio, dtype=np.int16).reshape(-1).copy()
        if not len(chunk):
            return None

        level = self.level_dbfs(chunk)
        if self._recording is None:
            self._remember(chunk)
            if not triggered:
                self._idle_levels.append(level)
                return None
            self._recording = list(self._pre_roll)
            self._recording_size = sum(len(part) for part in self._recording)
            if len(self._idle_levels) >= 10:
                # A low percentile, so the wake phrase and passing chatter in
                # the window do not read as the room's floor.
                self._floor_db = float(np.percentile(self._idle_levels, 10))
            return None

        self._recording.append(chunk)
        self._recording_size += len(chunk)
        self._after_trigger_samples += len(chunk)
        if self._after_trigger_samples <= self.wake_tail_samples:
            # Probably the wake phrase, possibly a very quick command: keep
            # it, but let it close the turn only after the longer wait below.
            if level >= self.active_threshold_db:
                self._tail_speech = True
        elif level >= self.active_threshold_db:
            self._peak_db = level if self._peak_db is None else max(self._peak_db, level)
            self._speech_seen = True
            self._silence_samples = 0
            self._loud_chunks += 1
        elif self._speech_seen:
            self._silence_samples += len(chunk)

        if self._recording_size >= self.max_samples:
            return self._finish()
        if not self._speech_seen and self._after_trigger_samples >= (
            self.tail_only_samples if self._tail_speech else self.no_speech_samples
        ):
            return self._finish()
        if (
            self._speech_seen
            and self._recording_size >= self.min_samples
            and self._after_trigger_samples >= self.start_grace_samples
            and self._silence_samples >= self.silence_samples_required
        ):
            return self._finish()
        return None


# whisper.cpp labels non-speech audio instead of returning nothing, so a turn
# of silence or a cough used to arrive at the router as the literal words
# "blank audio" and was answered like a real question.
_NON_SPEECH_ANNOTATION = re.compile(
    r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|<[^>]*>"
)


def clean_transcript(text: str) -> str:
    """Drop whisper's non-speech annotations, leaving only spoken words."""
    stripped = _NON_SPEECH_ANNOTATION.sub(" ", text)
    stripped = " ".join(stripped.split()).strip()
    # An annotation-only result means nothing was said.
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return ""
    return _collapse_repeats(stripped)


def _collapse_repeats(text: str) -> str:
    """Keep one copy of a sentence whisper repeated back to back.

    Trailing room noise makes it loop on the last phrase, so one "Follow me"
    arrived as "Follow me. Follow me. Follow me."
    """
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if kept and key and key == re.sub(r"[^a-z0-9]+", " ", kept[-1].lower()).strip():
            continue
        kept.append(sentence)
    return " ".join(kept)


def condition_utterance(
    audio: np.ndarray,
    *,
    target_dbfs: float = -3.0,
    max_gain: float = 8.0,
) -> np.ndarray:
    """Remove DC offset and normalize level without clipping the speech.

    A fixed input gain has to be tuned for the quietest speaker, which then
    squares off every loud or close-up utterance.  Whisper loses far more
    accuracy to that clipping than to a low level, so the peak is scaled to a
    fixed headroom instead and the gain is only ever applied downward-safe.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size:
        return np.asarray(audio, dtype=np.int16).reshape(-1)
    samples = samples - float(samples.mean())
    peak = float(np.max(np.abs(samples)))
    if peak < 1.0:
        return np.zeros(samples.size, dtype=np.int16)
    target_peak = 32767.0 * (10.0 ** (target_dbfs / 20.0))
    gain = min(max(0.0, max_gain), target_peak / peak)
    return (samples * gain).clip(-32768, 32767).astype(np.int16)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -45.0,
    keep_s: float = 0.2,
    window_s: float = 0.02,
    noise_margin_db: float = 6.0,
) -> np.ndarray:
    """Cut leading and trailing silence, keeping a short pad on each side.

    The segmenter deliberately keeps pre-roll and trailing silence so no word
    is clipped, but handing that silence to whisper only invites it to invent
    text for it.
    """
    samples = np.asarray(audio, dtype=np.int16).reshape(-1)
    window = max(1, int(window_s * sample_rate))
    if samples.size <= window:
        return samples
    usable = samples.size - samples.size % window
    frames = samples[:usable].reshape(-1, window)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    levels = 20.0 * np.log10(np.maximum(rms, 1e-9) / 32768.0)
    # The room floor on the robot (-39 dB) sits above any fixed threshold that
    # is safe in a quiet room, so nothing was ever trimmed. Follow the floor,
    # but never rise to within 20 dB of the speech peak.
    floor_db = float(np.percentile(levels, 10))
    threshold_db = max(threshold_db, min(floor_db + noise_margin_db, float(levels.max()) - 20.0))
    loud = np.flatnonzero(levels >= threshold_db)
    if not loud.size:
        return samples
    pad = max(0, int(keep_s * sample_rate))
    start = max(0, loud[0] * window - pad)
    end = min(samples.size, (loud[-1] + 1) * window + pad)
    return samples[start:end]


def whisper_audio_context(sample_count: int, sample_rate: int) -> int:
    """Encoder frames needed for this utterance instead of whisper's fixed 30 s.

    Whisper pads every clip to 30 s (1500 frames at 50 per second) and the
    encoder is most of a turn's cost on this CPU, so a three-second command was
    paying for thirty. Generous padding keeps the transcript unchanged.
    """
    seconds = sample_count / max(1, sample_rate)
    frames = int(np.ceil(seconds * 50.0)) + 128
    frames = -(-frames // 64) * 64
    # Never below 768. large-v3-turbo on the GPU build falls apart on a tighter
    # window: at 384 it returned garbage and looped for 10-15 s on clips that
    # take 0.7 s at 768. The GPU makes the wider window nearly free.
    return int(min(1500, max(768, frames)))


class WhisperCppTranscriber:
    """Transcribe mono PCM with the standalone whisper.cpp CLI."""

    # Whisper conditions on this text, so it is the cheapest accuracy win
    # available: it should name the robot, every spoken command, and the
    # health vocabulary this robot is actually asked about.
    DEFAULT_PROMPT = (
        "Hey BracketBot. Baymax. Weather in Waterloo. Wave. Handshake. "
        "Fist bump. Hug. Salute. Namaste. Dance. Point at the person. Look at "
        "me. Stop. Follow me. Follow me around. Stop following me. Come here. "
        "Remind me in four minutes to take my meds. Set a timer for four "
        "minutes. Cancel my reminder. Check my heart rate. Check me out. "
        "Start my checkup. "
        "What should I do if I cut my finger? I burned my hand. A nosebleed, "
        "a sprained ankle, a bruise, a scrape, a splinter, a bee sting, a "
        "fever, a headache, dizziness, choking, an allergic reaction, CPR."
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
                "--audio-ctx",
                str(whisper_audio_context(len(audio), sample_rate)),
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
            return clean_transcript(result.stdout)
        except subprocess.TimeoutExpired as exc:
            raise LocalVoiceError("Local transcription timed out.") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Serialize mono PCM to an in-memory WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(audio, dtype=np.int16).tobytes())
    return buffer.getvalue()


def _multipart(fields: dict[str, str], filename: str, payload: bytes) -> tuple[bytes, str]:
    """Encode one file upload plus simple fields, without adding a dependency."""
    boundary = "----BracketBotWhisper" + hashlib.sha1(payload[:4096]).hexdigest()[:16]
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode("utf-8")
    )
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class WhisperServerTranscriber:
    """Transcribe against a resident whisper.cpp server.

    The CLI reloads the whole model from disk on every invocation, which is
    pure fixed cost on each spoken turn and by far the largest share of the
    gap between the person finishing a sentence and the robot answering.  A
    long-lived ``whisper-server`` holds the model in memory instead, so a turn
    pays only for inference.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        language: str = "en",
        timeout: float = 45.0,
        prompt: str | None = None,
        opener=request.urlopen,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.language = language
        self.timeout = timeout
        self.prompt = WhisperCppTranscriber.DEFAULT_PROMPT if prompt is None else prompt
        self._opener = opener

    def validate(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise LocalVoiceError(
                f"The whisper server endpoint must be an HTTP URL: {self.endpoint}"
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.validate()
        body, content_type = _multipart(
            {
                "temperature": "0.0",
                # No temperature retries: on a clip with no clear speech they
                # held a turn for 8 s and returned text from the prompt.
                "temperature_inc": "0.0",
                "response_format": "json",
                "language": self.language,
                "prompt": self.prompt,
                "audio_ctx": str(whisper_audio_context(len(audio), sample_rate)),
            },
            "utterance.wav",
            _wav_bytes(audio, sample_rate),
        )
        http_request = request.Request(
            f"{self.endpoint}/inference",
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        try:
            with self._opener(http_request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except (error.URLError, TimeoutError, OSError) as exc:
            raise LocalVoiceError(f"Whisper server unavailable: {exc}") from exc
        try:
            document = json.loads(raw)
            text = document.get("text", "") if isinstance(document, dict) else raw
        except json.JSONDecodeError:
            text = raw
        return clean_transcript(str(text))


class FallbackTranscriber:
    """Prefer the resident server, but never lose a turn if it is down."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def validate(self) -> None:
        # Only the offline path must be present; the server may start later.
        self.fallback.validate()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        try:
            return self.primary.transcribe(audio, sample_rate)
        except LocalVoiceError as exc:
            print(f"[local-voice] {exc} Falling back to the whisper CLI.", flush=True)
            return self.fallback.transcribe(audio, sample_rate)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or not len(audio):
        return np.asarray(audio, dtype=np.int16)
    target_len = max(1, round(len(audio) * target_rate / source_rate))
    source_x = np.arange(len(audio), dtype=np.float64)
    target_x = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(target_x, source_x, audio).clip(-32768, 32767).astype(np.int16)


def _bell(freqs: np.ndarray, centre: float, gain_db: float, octaves: float) -> np.ndarray:
    """A smooth EQ bump (in dB) centred on ``centre`` and ``octaves`` wide."""
    distance = np.log2(np.maximum(freqs, 1.0) / centre) / octaves
    return gain_db * np.exp(-0.5 * distance * distance)


def _fft_length(count: int) -> int:
    return 1 << max(1, int(count - 1).bit_length())


def _smooth(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``np.convolve(signal, kernel, mode="same")`` through the FFT.

    The direct form is signal x kernel multiplies, about half a billion for a
    five-second sentence against the 40 ms envelope window.
    """
    if signal.size < kernel.size:
        return np.convolve(signal, kernel, mode="same")
    length = _fft_length(signal.size + kernel.size - 1)
    full = np.fft.irfft(np.fft.rfft(signal, length) * np.fft.rfft(kernel, length), n=length)
    start = (kernel.size - 1) // 2
    return np.maximum(full[start : start + signal.size], 0.0)


def clarify_speech(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_peak: float = 0.89,
    threshold_db: float = -26.0,
    ratio: float = 3.0,
) -> np.ndarray:
    """Make synthesized speech intelligible on the robot's small speaker.

    Raw TTS puts most of its energy in the 120-500 Hz vowel body, which a small
    driver turns into boom, while the 2-5 kHz consonant band that carries
    intelligibility sits about 18 dB lower.  This removes what the speaker
    cannot reproduce, lifts the consonants, evens out word-to-word level so
    quiet syllables are not lost in the room, and normalizes the peak so every
    line plays at the same loudness.
    """
    samples = np.asarray(audio, dtype=np.float64).reshape(-1) / 32768.0
    if samples.size < 64 or not np.any(samples):
        return np.asarray(audio, dtype=np.int16).reshape(-1)

    # Zero-phase EQ in the frequency domain.
    # Padded to a power of two: a reply's arbitrary sample count can have a
    # large prime factor, which made this one transform cost most of a second
    # on the robot, all of it in front of the first spoken word.
    count = samples.size
    padded = _fft_length(count)
    spectrum = np.fft.rfft(samples, padded)
    freqs = np.fft.rfftfreq(padded, 1.0 / sample_rate)
    ratio_hp = (freqs / 150.0) ** 4
    gain_db = 10.0 * np.log10(np.maximum(ratio_hp / (1.0 + ratio_hp), 1e-12))
    gain_db += _bell(freqs, 300.0, -3.0, 0.9)  # mud
    gain_db += _bell(freqs, 3200.0, 7.0, 1.1)  # consonant presence
    gain_db += _bell(freqs, 7000.0, 3.0, 0.8)  # sibilant air
    samples = np.fft.irfft(spectrum * 10.0 ** (gain_db / 20.0), n=padded)[:count]

    # Downward compression driven by a smoothed RMS envelope.
    window = max(1, int(0.02 * sample_rate))
    kernel = np.hanning(window * 2 + 1)
    kernel /= kernel.sum()
    envelope = np.sqrt(_smooth(samples * samples, kernel) + 1e-12)
    level_db = 20.0 * np.log10(envelope)
    over = np.maximum(0.0, level_db - threshold_db)
    samples = samples * 10.0 ** (-over * (1.0 - 1.0 / ratio) / 20.0)

    # Short fades so the EQ never leaves a click at either edge.
    fade = min(samples.size // 2, int(0.005 * sample_rate))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        samples[:fade] *= ramp
        samples[-fade:] *= ramp[::-1]

    # Normalize on the body of the speech rather than its few tallest peaks,
    # then round those peaks off above a knee.  A handful of plosives would
    # otherwise hold the whole line several dB quieter than it needs to be.
    body = float(np.percentile(np.abs(samples), 99.7))
    if body > 0.0:
        samples = samples * (target_peak * 0.8 / body)
    knee = target_peak * 0.7
    span = target_peak - knee
    excess = np.abs(samples) - knee
    samples = np.where(
        excess > 0.0,
        np.sign(samples) * (knee + span * np.tanh(np.maximum(excess, 0.0) / span)),
        samples,
    )
    return (samples * 32767.0).clip(-32768, 32767).astype(np.int16)


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
        return clarify_speech(_resample(pcm, source_rate, target_rate), target_rate)


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
        return clarify_speech(_resample(pcm, source_rate, target_rate), target_rate)


class FallbackSynthesizer:
    """Prefer natural speech but keep the robot able to talk if it is unavailable."""

    def __init__(self, primary, fallback, *, retry_after_s: float = 60.0, clock=time.monotonic):
        self.primary = primary
        self.fallback = fallback
        self.retry_after_s = retry_after_s
        self._clock = clock
        self._primary_down_until = 0.0

    def validate(self) -> None:
        self.primary.validate()
        self.fallback.validate()

    def synthesize(self, text: str, target_rate: int) -> np.ndarray:
        # Replies are synthesized a sentence at a time. Retrying the primary
        # on every sentence made one reply switch voice (and accent) halfway
        # through, so a failure holds the fallback voice for a while.
        if self._clock() < self._primary_down_until:
            return self.fallback.synthesize(text, target_rate)
        try:
            return self.primary.synthesize(text, target_rate)
        except LocalVoiceError:
            self._primary_down_until = self._clock() + self.retry_after_s
            return self.fallback.synthesize(text, target_rate)


def split_sentences(text: str, *, min_chars: int = 24) -> list[str]:
    """Split a reply into speakable sentences for incremental synthesis.

    Very short fragments are merged forward so the robot never speaks a lone
    "Okay." with a synthesis gap behind it.
    """
    pieces = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if not pieces:
        return []
    merged = [pieces[0]]
    for piece in pieces[1:]:
        if len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


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
