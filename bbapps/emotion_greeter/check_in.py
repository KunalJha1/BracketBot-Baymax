"""Spoken check-in after a sustained sadness cue.

Baymax asks what is up, listens for an answer, and replies through the same
OpenRouter chat client the voice assistant uses. The speech helpers live in
``bbapps/greeter`` and are dependency-free apart from NumPy, so they are
imported from there rather than duplicated. BBOS readers/writers are passed in
so the conversation logic stays testable off the robot.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

GREETER_DIR = Path(__file__).resolve().parent.parent / "greeter"
if str(GREETER_DIR) not in sys.path:
    sys.path.append(str(GREETER_DIR))

from local_voice import (  # noqa: E402
    EspeakSynthesizer,
    FallbackSynthesizer,
    HttpTtsSynthesizer,
    LocalVoiceError,
    SpeechSegmenter,
    WhisperCppTranscriber,
    speaker_chunks,
)
from voice_router import OpenRouterClient  # noqa: E402


OPENING_LINE = "Hey, why are you sad? What's up?"
FALLBACK_REPLY = "I'm sorry you're feeling down. I'm right here if you want to talk."
NO_ANSWER_REPLY = "That's okay. I'm here whenever you want to talk."
CHECK_IN_SYSTEM_PROMPT = (
    "You are Baymax, a gentle, caring home robot. Your camera noticed that "
    "the person in front of you looked sad, so you asked them: "
    f"\"{OPENING_LINE}\" Reply to what they say in one or two short, warm, "
    "natural sentences, because your words are spoken aloud. Listen and "
    "validate their feelings, and when it fits ask one gentle follow-up "
    "question. Do not diagnose, lecture, or claim to know how they feel; the "
    "camera cue can be wrong, so if they say they are fine, accept it kindly. "
    "If they mention wanting to hurt themselves or being in danger, tell them "
    "you care and encourage them to contact someone they trust right now or "
    "call or text 988 (the Suicide and Crisis Lifeline in the US and Canada)."
)


def load_env(path: Path) -> None:
    """Load KEY=VALUE secrets (OPENROUTER_API_KEY) without overriding the shell."""
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class SadCheckIn:
    """Run one short spoken conversation at a time on the robot."""

    def __init__(
        self,
        *,
        synthesizer,
        transcriber,
        llm_factory: Callable[[], Any],
        speaker_config,
        mic_config,
        open_speaker: Callable[[], Any],
        open_mic: Callable[[], Any],
        max_turns: int = 3,
        answer_timeout: float = 8.0,
        mic_gain: float = 3.0,
        volume: float = 1.0,
        vad_threshold_db: float = -38.0,
        trailing_silence: float = 1.2,
        max_utterance: float = 12.0,
        log: Callable[[str], None] = lambda line: print(line, flush=True),
    ) -> None:
        self.synthesizer = synthesizer
        self.transcriber = transcriber
        self.llm_factory = llm_factory
        self.speaker_config = speaker_config
        self.mic_config = mic_config
        self.open_speaker = open_speaker
        self.open_mic = open_mic
        self.max_turns = max(1, max_turns)
        self.answer_timeout = answer_timeout
        self.mic_gain = mic_gain
        self.volume = volume
        self.vad_threshold_db = vad_threshold_db
        self.trailing_silence = trailing_silence
        self.max_utterance = max_utterance
        self.log = log
        self.lock = threading.Lock()
        self.status = "idle"

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def speak(self, text: str) -> None:
        self.status = "speaking"
        self.log(f"[check-in] Baymax: {text}")
        config = self.speaker_config
        pcm = self.synthesizer.synthesize(text, config.sample_rate)
        pcm = (pcm.astype(np.float32) * self.volume).clip(-32768, 32767)
        chunks = speaker_chunks(pcm.astype(np.int16), config.chunk_size, config.channels)
        period = config.chunk_size / config.sample_rate
        with self.open_speaker() as speaker:
            due = time.monotonic()
            for chunk in chunks:
                with speaker.buf() as frame:
                    frame["audio"] = chunk.reshape(-1, config.channels)
                due += period
                time.sleep(max(0.0, due - time.monotonic()))
            # Let the speaker buffer drain so the mic does not hear Baymax.
            time.sleep(0.4)

    def listen(self) -> str:
        """Record one answer; return "" if the person stays quiet."""
        self.status = "listening"
        segmenter = SpeechSegmenter(
            self.mic_config.sample_rate,
            threshold_db=self.vad_threshold_db,
            pre_roll_s=0.3,
            trailing_silence_s=self.trailing_silence,
            start_grace_s=0.0,
            max_utterance_s=self.max_utterance,
        )
        deadline = time.monotonic() + self.answer_timeout
        started = False
        utterance = None
        # Open a fresh reader after speaking so no stale or self-heard audio
        # from the prompt is replayed into the answer.
        with self.open_mic() as microphone:
            while utterance is None:
                # Once the person starts talking, let them finish past the
                # answer timeout; the segmenter's max_utterance bounds it.
                if not segmenter.speech_seen and time.monotonic() > deadline:
                    return ""
                if not microphone.ready():
                    time.sleep(0.01)
                    continue
                audio = microphone.data["audio"].reshape(-1).astype(np.float32)
                boosted = (audio * self.mic_gain).clip(-32768, 32767).astype(np.int16)
                utterance = segmenter.push(boosted, triggered=not started)
                started = True
        self.status = "thinking"
        return self.transcriber.transcribe(utterance, self.mic_config.sample_rate)

    def converse(self) -> None:
        if not self.lock.acquire(blocking=False):
            return
        try:
            llm = self.llm_factory()
            self.speak(OPENING_LINE)
            for turn in range(self.max_turns):
                answer = self.listen()
                if not answer:
                    if turn == 0:
                        self.speak(NO_ANSWER_REPLY)
                    self.log("[check-in] No answer; ending conversation")
                    return
                self.log(f"[check-in] Heard: {answer}")
                try:
                    reply = llm.complete(answer).text
                except Exception as exc:
                    self.log(f"[check-in] LLM unavailable: {type(exc).__name__}: {exc}")
                    self.speak(FALLBACK_REPLY)
                    return
                self.speak(reply)
        except Exception as exc:
            # A failed provider must end only this conversation, never vision.
            self.log(f"[check-in] Conversation failed: {type(exc).__name__}: {exc}")
        finally:
            self.status = "idle"
            self.lock.release()

    def start_async(self) -> bool:
        if self.busy:
            return False
        threading.Thread(target=self.converse, name="sad-check-in", daemon=True).start()
        return True


def build_check_in(args, Config, Reader, Type, Writer) -> SadCheckIn:
    """Create the robot check-in; raises LocalVoiceError if speech is unavailable."""
    load_env(args.env)
    transcriber = WhisperCppTranscriber(
        args.whisper_bin, args.whisper_model, threads=args.whisper_threads
    )
    offline = EspeakSynthesizer(args.espeak_bin)
    tts_url = args.tts_url or os.environ.get("LOCAL_TTS_URL", "")
    synthesizer = (
        FallbackSynthesizer(HttpTtsSynthesizer(tts_url), offline) if tts_url else offline
    )
    transcriber.validate()
    synthesizer.validate()
    return SadCheckIn(
        synthesizer=synthesizer,
        transcriber=transcriber,
        # A fresh client per conversation keeps one person's history out of
        # the next check-in. Tools stay off: this is a listening conversation.
        llm_factory=lambda: OpenRouterClient(system_prompt=CHECK_IN_SYSTEM_PROMPT),
        speaker_config=Config("speaker"),
        mic_config=Config("mic"),
        open_speaker=lambda: Writer(
            "speaker.audio", Type("speaker_audio"), keeptime=False, buf_ms=400
        ),
        open_mic=lambda: Reader("mic.audio", keeptime=False),
        max_turns=args.check_in_turns,
        answer_timeout=args.check_in_answer_timeout,
        mic_gain=args.mic_gain,
    )


__all__ = [
    "CHECK_IN_SYSTEM_PROMPT",
    "LocalVoiceError",
    "OPENING_LINE",
    "SadCheckIn",
    "build_check_in",
]
