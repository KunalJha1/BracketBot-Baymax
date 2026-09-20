"""Spoken check-in after a sustained sadness cue.

Baymax asks what is up, listens for an answer, and replies through the same
OpenRouter chat client the voice assistant uses. The speech helpers live in
``bbapps/greeter`` and are dependency-free apart from NumPy, so they are
imported from there rather than duplicated. BBOS readers/writers are passed in
so the conversation logic stays testable off the robot.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any, Callable, Iterator, Sequence

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
from voice_router import BrowserbaseSearchClient, OpenRouterClient  # noqa: E402
import speech_relay  # noqa: E402


# Short openers keep the first spoken moment quick and stop the robot from
# saying the exact same sentence every time it reads a frown. The first line
# matches the recorded ``sad_prompt.wav`` fallback.
OPENING_LINES = (
    "Hey, why are you sad? What's up?",
    "Hey, you okay? What's going on?",
    "You look a little down. What happened?",
    "Hey. Rough moment? I'm listening.",
    "That looked like a heavy sigh. What's up?",
)
OPENING_LINE = OPENING_LINES[0]
FALLBACK_REPLY = "I'm sorry you're feeling down. I'm right here if you want to talk."
NO_ANSWER_REPLY = "That's okay. I'm here whenever you want to talk."
DISMISSED_REPLY = "Okay. I'm here if you need me."
# The check-in has no wake word, so without a way out the person is held in it
# for every remaining turn: "stop" reaches only the voice assistant, which
# answers "No voice action is running" while Baymax keeps asking follow-ups.
_DISMISSAL = re.compile(
    r"^(?:no |okay |ok |please |baymax )*"
    r"(?:stop(?: it| talking| that)?|be quiet|quiet|shut up|go away|"
    r"leave me alone|never ?mind|that s all|that is all|"
    r"i m (?:fine|okay|ok|good|all right|alright)|"
    r"i am (?:fine|okay|ok|good|all right|alright)|"
    r"no thanks|no thank you|bye|goodbye)"
    r"(?: please| now| baymax| thanks| thank you)*$"
)
# Said mid check-in, the wake phrase means the person is talking to the voice
# assistant, which hears it too. Answering as well makes two replies to one
# sentence, and the check-in can only pretend to do the gesture they asked for.
_WAKE_PHRASE = re.compile(r"\b(?:hey|hi|hay|okay|ok) bracket ?bot\b")
LAST_TURN_NOTE = (
    "\n\n(This is your last reply in this conversation, and nobody will hear "
    "an answer to it: close warmly and do not ask a question.)"
)
CHECK_IN_SYSTEM_PROMPT = (
    "You are Baymax, a gentle, caring home robot. Your camera noticed that "
    "the person in front of you looked sad, so you opened with a short "
    f"question like \"{OPENING_LINE}\" Reply to what they say in one or two "
    "short, warm, natural sentences, because your words are spoken aloud and "
    "a long answer feels slow. Lead with the reply itself: no preamble, no "
    "restating what they told you, no stage directions. You cannot move in "
    "this conversation, so never act out or describe a hug or any other "
    "gesture; if they ask for one, tell them to say \"Hey BracketBot, do a "
    "hug.\" Listen and validate "
    "their feelings, and when it fits ask one gentle follow-up question. Do "
    "not diagnose, lecture, or claim to know how they feel; the camera cue "
    "can be wrong, so if they say they are fine, accept it kindly. If they "
    "mention wanting to hurt themselves or being in danger, tell them you "
    "care and encourage them to contact someone they trust right now or call "
    "or text 988 (the Suicide and Crisis Lifeline in the US and Canada)."
)
# Check-in states as the assistant's LedStatus names them.
LED_STATUS = {
    "speaking": "speaking",
    "listening": "listening",
    "thinking": "processing",
}
# Splitting a reply on sentence ends lets the first sentence start playing
# while the rest is still being synthesized.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def speech_segments(text: str, min_chars: int = 24) -> list[str]:
    """Split spoken text into sentences, merging fragments too short to stream."""
    segments: list[str] = []
    for part in _SENTENCE_END.split(text.strip()):
        part = part.strip()
        if not part:
            continue
        if segments and len(segments[-1]) < min_chars:
            segments[-1] = f"{segments[-1]} {part}"
        else:
            segments.append(part)
    return segments or [text.strip()]


def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def is_dismissal(text: str) -> bool:
    """True when the whole answer asks Baymax to stop, not "I can't stop crying"."""
    return bool(_DISMISSAL.match(_normalized(text)))


def addresses_assistant(text: str) -> bool:
    return bool(_WAKE_PHRASE.search(_normalized(text)))


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
        trailing_silence: float = 0.7,
        max_utterance: float = 12.0,
        speaker_drain: float = 0.4,
        dismiss_snooze: float = 300.0,
        openings: Sequence[str] = OPENING_LINES,
        show_led: Callable[[str | None], None] = lambda status: None,
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
        self.speaker_drain = speaker_drain
        self.dismiss_snooze = dismiss_snooze
        # No new check-in starts before this; set when the person sends Baymax
        # away so the same resting face does not reopen the conversation.
        self.quiet_until = 0.0
        self.openings = tuple(openings) or (OPENING_LINE,)
        self.show_led = show_led
        self.log = log
        self.lock = threading.Lock()
        self._status = "idle"
        # Pre-rendered openers so a frown is answered without waiting on TTS.
        self._voice_cache: dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._last_opening: str | None = None
        self._synth_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="check-in-tts"
        )

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        if value == self._status:
            return
        self._status = value
        # The voice assistant owns led.ctrl; it mirrors this on the neck with
        # its own colors, so waiting for an answer glows blue like a wake turn.
        self.show_led(LED_STATUS.get(value))

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def next_opening(self) -> str:
        """Pick an opener, avoiding the line used for the previous check-in."""
        choices = [line for line in self.openings if line != self._last_opening]
        opening = random.choice(choices or list(self.openings))
        self._last_opening = opening
        return opening

    def prewarm(self) -> int:
        """Synthesize every opener up front; returns how many are cached.

        Rendering the openers at startup is what makes the trigger feel
        instant: at cue time the robot only has to push PCM at the speaker.
        """
        started = time.monotonic()
        for line in self.openings:
            try:
                self._pcm_for(line)
            except Exception as exc:
                self.log(
                    f"[check-in] Could not pre-render an opener "
                    f"({type(exc).__name__}: {exc})"
                )
                break
        cached = len(self._voice_cache)
        if cached:
            self.log(
                f"[check-in] Pre-rendered {cached} opener(s) in "
                f"{(time.monotonic() - started) * 1000:.0f} ms"
            )
        return cached

    def prewarm_async(self) -> None:
        threading.Thread(
            target=self.prewarm, name="check-in-prewarm", daemon=True
        ).start()

    def _pcm_for(self, text: str) -> np.ndarray:
        """Synthesize ``text``, reusing a cached opener when there is one."""
        with self._cache_lock:
            cached = self._voice_cache.get(text)
        if cached is not None:
            return cached
        pcm = self.synthesizer.synthesize(text, self.speaker_config.sample_rate)
        if text in self.openings:
            with self._cache_lock:
                self._voice_cache[text] = pcm
        return pcm

    def _pcm_stream(self, segments: Sequence[str]) -> Iterator[np.ndarray]:
        """Yield each sentence's PCM while the next one is already rendering."""
        ahead = None
        for index, segment in enumerate(segments):
            pcm = ahead.result() if ahead is not None else self._pcm_for(segment)
            ahead = (
                self._synth_pool.submit(self._pcm_for, segments[index + 1])
                if index + 1 < len(segments)
                else None
            )
            yield pcm

    def speak(self, text: str) -> None:
        self.status = "speaking"
        self.log(f"[check-in] Baymax: {text}")
        stack = ExitStack()
        try:
            speaker = stack.enter_context(self.open_speaker())
        except RuntimeError as error:
            # speaker.audio takes one writer process, and the always-on voice
            # assistant holds it for its whole lifetime. Ask whoever owns it to
            # say this instead of failing the conversation. Delegating before
            # synthesising also saves a TTS call we could not have played.
            stack.close()
            self.log(f"[check-in] Speaker owned elsewhere ({error}); relaying")
            if not speech_relay.request(text=text):
                raise LocalVoiceError(
                    "speaker.audio is owned by another process and nothing is "
                    "serving the speech relay"
                ) from error
            return
        with stack:
            config = self.speaker_config
            segments = speech_segments(text)
            period = config.chunk_size / config.sample_rate
            due = time.monotonic()
            # Only the first sentence needs the jitter-buffer lead; padding the
            # later ones would open a silent gap mid-reply.
            lead_chunks = 4
            for pcm in self._pcm_stream(segments):
                pcm = (pcm.astype(np.float32) * self.volume).clip(-32768, 32767)
                chunks = speaker_chunks(
                    pcm.astype(np.int16),
                    config.chunk_size,
                    config.channels,
                    lead_chunks=lead_chunks,
                )
                lead_chunks = 0
                # Never burst after a slow synthesis: resume pacing from now.
                due = max(due, time.monotonic())
                for chunk in chunks:
                    with speaker.buf() as frame:
                        frame["audio"] = chunk.reshape(-1, config.channels)
                    due += period
                    time.sleep(max(0.0, due - time.monotonic()))
            # Let the speaker buffer drain so the mic does not hear Baymax.
            # This one stays generous on purpose: shaving it would risk the
            # robot transcribing its own voice, which costs a whole turn.
            time.sleep(self.speaker_drain)

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
            # The opener is already rendered, so nothing but the speaker sits
            # between the sadness cue and Baymax's voice.
            self.speak(self.next_opening())
            for turn in range(self.max_turns):
                answer = self.listen()
                if not answer:
                    if turn == 0:
                        self.speak(NO_ANSWER_REPLY)
                    self.log("[check-in] No answer; ending conversation")
                    return
                self.log(f"[check-in] Heard: {answer}")
                if addresses_assistant(answer):
                    self.quiet_until = time.monotonic() + self.dismiss_snooze
                    self.log("[check-in] Wake phrase heard; leaving it to the assistant")
                    return
                if is_dismissal(answer):
                    self.quiet_until = time.monotonic() + self.dismiss_snooze
                    self.speak(DISMISSED_REPLY)
                    self.log("[check-in] Dismissed; ending conversation")
                    return
                if turn == self.max_turns - 1:
                    # A closing question would go unheard, which invites an
                    # answer nobody is listening for.
                    answer += LAST_TURN_NOTE
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
        if self.busy or time.monotonic() < self.quiet_until:
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
        # the next check-in. Tools stay off with an unconfigured search client:
        # this is a listening conversation, and a tool round would add a whole
        # extra round trip before Baymax can answer. The short token budget
        # keeps replies to the couple of sentences the prompt asks for, which
        # is both faster to generate and faster to speak.
        llm_factory=lambda: OpenRouterClient(
            system_prompt=CHECK_IN_SYSTEM_PROMPT,
            web_search=BrowserbaseSearchClient(api_key=""),
            max_tokens=110,
        ),
        speaker_config=Config("speaker"),
        mic_config=Config("mic"),
        open_speaker=lambda: Writer(
            "speaker.audio", Type("speaker_audio"), keeptime=False, buf_ms=400
        ),
        open_mic=lambda: Reader("mic.audio", keeptime=False),
        max_turns=args.check_in_turns,
        answer_timeout=args.check_in_answer_timeout,
        mic_gain=args.mic_gain,
        trailing_silence=args.check_in_trailing_silence,
        show_led=speech_relay.post_led_status,
    )


__all__ = [
    "CHECK_IN_SYSTEM_PROMPT",
    "DISMISSED_REPLY",
    "LAST_TURN_NOTE",
    "LocalVoiceError",
    "OPENING_LINE",
    "OPENING_LINES",
    "SadCheckIn",
    "build_check_in",
    "speech_segments",
]
