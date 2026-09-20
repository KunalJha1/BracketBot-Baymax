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
import json
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


# Short, consent-based openers keep the first spoken moment quick, make it
# clear that silence/space is a valid answer, and stop the robot from saying
# the exact same sentence every time it reads a frown.
OPENING_LINES = (
    "Hey, quick check-in. Want to talk, or would you rather have some space?",
    "Hey, you okay? We can talk, or I can give you some space.",
    "You seem a little down. Want to talk about it, or should I leave you be?",
    "Hey. Rough moment? I'm here if you want to talk, and it's okay if you don't.",
    "Just checking in. Want to talk, or would you rather have some quiet?",
)
OPENING_LINE = OPENING_LINES[0]
FALLBACK_REPLY = "I'm sorry you're feeling down. I'm right here if you want to talk."
NO_ANSWER_REPLY = "That's okay. I'm here whenever you want to talk."
DISMISSED_REPLY = (
    "Okay. I'll give you some space. Just say Hey BracketBot if you need me."
)
# The check-in has no wake word, so without a way out the person is held in it
# for every remaining turn: "stop" reaches only the voice assistant, which
# answers "No voice action is running" while Baymax keeps asking follow-ups.
_DISMISSAL = re.compile(
    r"^(?:no |okay |ok |please |baymax )*"
    r"(?:no|nah|nope|not (?:now|right now|today)|maybe later|"
    r"stop(?: it| talking| that)?|be quiet|quiet|shut up|go away|"
    r"leave me alone|never ?mind|that s all|that is all|"
    r"(?:give|leave) me (?:some )?space|let me be|"
    r"i (?:want|need) (?:some )?(?:space|quiet)|"
    r"i (?:would|d) rather not|"
    r"i (?:do not|don t|dont) (?:really )?(?:want to |wanna |feel like )?"
    r"(?:talk|talking|chat|chatting)|"
    r"i m (?:fine|okay|ok|good|all right|alright)|"
    r"i am (?:fine|okay|ok|good|all right|alright)|"
    r"no thanks|no thank you|bye|goodbye)"
    r"(?: please| now| baymax| thanks| thank you)*$"
)
# --- Ground check-in: the robot has driven up to somebody lying on the floor. ---
# The voice assistant speaks the arrival line (it owns the speaker and the drive
# action), then drops this file; the vision app listens for the answer, because
# the microphone, transcriber and LLM for unprompted conversations live here.
GROUND_ARRIVAL_FILE = Path("/tmp/bracketbot_ground_arrived.json")
GROUND_ARRIVAL_MAX_AGE_S = 8.0
GROUND_OKAY_REPLY = (
    "Okay, glad you're alright. I'll leave you be. Just say Hey BracketBot if you need me."
)
# Same polarity as the arrival line ("... are you in trouble"), so a bare "no"
# always means "I'm fine" and a bare "yes" always means trouble.
GROUND_ASK_AGAIN = "I didn't hear you. Are you in trouble? Say I'm okay, or say help."
GROUND_NO_ANSWER_REPLY = (
    "I can't hear an answer. If anyone is nearby, someone here may need help. I'm staying right here."
)
GROUND_HELP_REPLY = (
    "Okay. I'm staying right here with you. If anyone is nearby, someone here needs help."
)
GROUND_CONTEXT_NOTE = (
    " (Context: you are a robot that just found this person lying on the floor and asked "
    "if they are okay. Be calm and practical in one or two short sentences. Do not give "
    "medical instructions; suggest calling for help or emergency services if they are hurt.)"
)
_GROUND_NOT_OKAY = re.compile(
    r"\b(?:not (?:so |too |very |really |that |feeling |doing )*(?:okay|ok|fine|good|alright|all right|well)|"
    r"help|hurt|hurts|pain|fell|fallen|"
    r"can t (?:get up|move|breathe)|cannot (?:get up|move|breathe)|stuck|dizzy|bleeding|"
    r"ambulance|emergency|doctor|9 ?1 ?1)\b"
)
_GROUND_OKAY = re.compile(
    r"\b(?:i m|i am|im|we re|we are|all|it s|its|everything s|everything is|that s|doing)? ?"
    r"(?:okay|ok|fine|good|alright|all right|all good)\b|"
    r"\b(?:just|only) (?:resting|relaxing|lying|laying|sleeping|napping|chilling|stretching|"
    r"sitting|joking|kidding|testing|playing)\b|"
    r"\b(?:don t|do not|dont) need (?:any )?help\b|\bno help\b|\bgo away\b|\bleave me\b"
)


_GROUND_NO_HELP = re.compile(r"\b(?:(?:don t|do not|dont) (?:need|want)|no need for|need no|no) (?:any )?help\b")
_GROUND_BARE_YES = re.compile(r"^(?:yes|yeah|yep|yup|uh huh|i am|i think so|kind of|a little|a bit)(?: please)?$")


def ground_answer_kind(text: str) -> str:
    """"help", "okay" or "unclear". Any sign of trouble wins over a reassuring word."""
    words = _normalized(text)
    # "I don't need help" must not trip on the word "help".
    declined = _GROUND_NO_HELP.sub(" ", words)
    if _GROUND_NOT_OKAY.search(declined) or _GROUND_BARE_YES.match(words):
        return "help"
    if _GROUND_OKAY.search(words) or is_dismissal(text):
        return "okay"
    return "unclear"


def read_ground_arrival(path: Path = GROUND_ARRIVAL_FILE, now=time.time) -> float | None:
    """When the assistant last said it reached someone, if that was moments ago."""
    try:
        arrived = float(json.loads(path.read_text())["arrived_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return arrived if 0 <= now() - arrived <= GROUND_ARRIVAL_MAX_AGE_S else None


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
    "their feelings. Do not ask a follow-up unless their answer clearly says "
    "they want to keep talking; if they sound brief, unsure, or reluctant, "
    "close warmly instead. Ask at most one follow-up question in the entire "
    "conversation. Do "
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
        max_turns: int = 2,
        answer_timeout: float = 8.0,
        mic_gain: float = 3.0,
        volume: float = 1.0,
        vad_threshold_db: float = -38.0,
        trailing_silence: float = 0.7,
        max_utterance: float = 12.0,
        speaker_drain: float = 0.4,
        dismiss_snooze: float = 1800.0,
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
        # No new check-in starts before this. Every completed check-in gets a
        # quiet period, whether the person talks, declines, or stays silent,
        # so one expression cannot turn into a loop of interruptions.
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
                    self.log("[check-in] Wake phrase heard; leaving it to the assistant")
                    return
                if is_dismissal(answer):
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
            # A check-in should never immediately chase the person with a new
            # one. The regular wake-word assistant remains available during
            # this frown-triggered quiet period.
            self.quiet_until = max(
                self.quiet_until, time.monotonic() + self.dismiss_snooze
            )
            self.status = "idle"
            self.lock.release()

    def converse_ground(self, on_result: Callable[[str], None]) -> None:
        """Listen to the person the robot just drove up to; report how it ended.

        ``on_result`` gets "okay" (stand down), "help" or "silent" (keep the
        emergency showing). The arrival question has already been spoken.
        """
        if not self.lock.acquire(blocking=False):
            return
        result = "silent"
        try:
            for attempt in range(2):
                answer = self.listen()
                if answer:
                    break
                if attempt == 0:
                    self.speak(GROUND_ASK_AGAIN)
            if not answer:
                self.log("[ground-check-in] No answer")
                self.speak(GROUND_NO_ANSWER_REPLY)
                return
            self.log(f"[ground-check-in] Heard: {answer}")
            kind = ground_answer_kind(answer)
            if kind == "okay":
                result = "okay"
                self.speak(GROUND_OKAY_REPLY)
                return
            result = "help"
            if kind == "help":
                self.speak(GROUND_HELP_REPLY)
                return
            try:
                self.speak(self.llm_factory().complete(answer + GROUND_CONTEXT_NOTE).text)
            except Exception as exc:
                self.log(f"[ground-check-in] LLM unavailable: {type(exc).__name__}: {exc}")
                self.speak(GROUND_HELP_REPLY)
        except Exception as exc:
            self.log(f"[ground-check-in] Conversation failed: {type(exc).__name__}: {exc}")
        finally:
            self.status = "idle"
            self.lock.release()
            self.log(f"[ground-check-in] Result: {result}")
            on_result(result)

    def start_ground_async(self, on_result: Callable[[str], None]) -> bool:
        # No quiet period here: a sad check-in's snooze must not mute an emergency.
        if self.busy:
            return False
        threading.Thread(
            target=self.converse_ground, args=(on_result,), name="ground-check-in", daemon=True
        ).start()
        return True

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
        dismiss_snooze=args.check_in_quiet_seconds,
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
