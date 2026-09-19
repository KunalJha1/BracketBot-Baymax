# /// script
# requires-python = "==3.10.*"
# dependencies = ["bbos", "numpy"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Gemini-free BracketBot voice assistant.

Uses the installed ``hey_bracket_bot`` wake-word daemon, local whisper.cpp,
local eSpeak synthesis, GPT-OSS through OpenRouter, and Browserbase Search.
It intentionally does not start the camera/YOLO greeter.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from bbos import Config, Reader, Type, Writer

try:
    from .gesture_runtime import RecordedGestureController
    from .local_voice import (
        EspeakSynthesizer,
        FallbackSynthesizer,
        HttpTtsSynthesizer,
        LocalVoiceError,
        SpeechSegmenter,
        WhisperCppTranscriber,
        play_dance_music,
        speaker_chunks,
    )
    from .voice_router import (
        OpenRouterClient,
        RouteKind,
        VoiceRouter,
        default_question_response_cache,
        default_seed_pairs,
    )
    from .voice_actions import SILENT_ACTIONS, RppgScanner, VoiceActionController
    from .reminders import default_reminder_db_path, default_timezone_name
    from .person_finder import PersonTrackerClient
except ImportError:
    from gesture_runtime import RecordedGestureController
    from local_voice import (
        EspeakSynthesizer,
        FallbackSynthesizer,
        HttpTtsSynthesizer,
        LocalVoiceError,
        SpeechSegmenter,
        WhisperCppTranscriber,
        play_dance_music,
        speaker_chunks,
    )
    from voice_router import (
        OpenRouterClient,
        RouteKind,
        VoiceRouter,
        default_question_response_cache,
        default_seed_pairs,
    )
    from voice_actions import SILENT_ACTIONS, RppgScanner, VoiceActionController
    from reminders import default_reminder_db_path, default_timezone_name
    from person_finder import PersonTrackerClient


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE secrets without adding a dotenv dependency."""
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def play_speech(writer, synthesizer, text, speaker_cfg, volume: float) -> None:
    pcm = synthesizer.synthesize(text, speaker_cfg.sample_rate)
    pcm = (pcm.astype(np.float32) * volume).clip(-32768, 32767).astype(np.int16)
    chunks = speaker_chunks(pcm, speaker_cfg.chunk_size, speaker_cfg.channels)
    # Match the BBOS speaker consumer exactly. Sending 10% fast eventually
    # overwrote ring-buffer chunks and made longer replies sound choppy.
    period = speaker_cfg.chunk_size / speaker_cfg.sample_rate
    due = time.monotonic()
    for chunk in chunks:
        with writer.buf() as data:
            data["audio"] = chunk.reshape(-1, speaker_cfg.channels)
        due += period
        time.sleep(max(0.0, due - time.monotonic()))


class LedStatus:
    """Continuously publish the assistant state to BracketBot's LEDs.

    The LED daemon returns to its default animation a few seconds after the
    last control frame, while Whisper and web requests can take longer than
    that.  A small dedicated publisher keeps the current state visible without
    making the audio loop responsible for refresh timing.
    """

    COLORS = {
        "idle": (0, 0, 0),
        "listening": (0, 190, 255),
        "processing": (255, 120, 0),
        "speaking": (170, 0, 255),
        "error": (255, 0, 0),
    }

    def __init__(self, refresh_s: float = 0.35):
        self.refresh_s = refresh_s
        self._status = "idle"
        self._effect = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set(self, status: str) -> None:
        if status not in self.COLORS:
            raise ValueError(f"Unknown LED status: {status}")
        with self._lock:
            self._status = status

    def start_effect(self, rgb, pattern: str, duration: float) -> None:
        if pattern not in {"solid", "pulse", "blink"}:
            raise ValueError(f"Unknown LED pattern: {pattern}")
        with self._lock:
            self._effect = (tuple(rgb), pattern, time.monotonic(), float(duration))

    def clear_effect(self) -> None:
        with self._lock:
            self._effect = None

    @staticmethod
    def _effect_scale(pattern: str, elapsed: float) -> float:
        if pattern == "solid":
            return 1.0
        if pattern == "blink":
            return 1.0 if int(elapsed * 3) % 2 == 0 else 0.08
        return 0.2 + 0.8 * (0.5 - 0.5 * math.cos(2 * math.pi * elapsed / 1.6))

    def _run(self) -> None:
        try:
            with Writer("led.ctrl", Type("led_ctrl"), keeptime=False) as led:
                while not self._stop.is_set():
                    with self._lock:
                        status = self._status
                        effect = self._effect
                    now = time.monotonic()
                    if effect is not None and now - effect[2] >= effect[3]:
                        with self._lock:
                            if self._effect == effect:
                                self._effect = None
                        effect = None
                    # Do not publish an idle frame. The LED daemon treats a
                    # quiet controller as released and restores the robot's
                    # configured idle color after its short stale timeout.
                    if status != "idle":
                        color = self.COLORS[status]
                    elif effect is not None:
                        rgb, pattern, started, _duration = effect
                        scale = self._effect_scale(pattern, now - started)
                        color = tuple(round(value * scale) for value in rgb)
                    else:
                        color = None
                    if color is not None:
                        with led.buf() as frame:
                            frame["rgb"] = np.asarray(
                                color, dtype=np.uint8
                            )
                            frame["brightness"] = np.int16(-1)
                            frame["period_ms"] = np.uint16(0)
                    self._stop.wait(self.refresh_s)
        except Exception as exc:
            # Voice should remain usable even if the optional LED daemon is
            # unavailable or another application currently owns the writer.
            print(f"[local-assistant] LED status unavailable: {exc}")

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._run,
            name="assistant-led-status",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def answer_text(router: VoiceRouter, utterance: str) -> str:
    decision = router.route(utterance)
    return decision.reply or "I did not understand that."


def run_voice(args, router, transcriber, synthesizer, action_controller) -> None:
    mic_cfg = Config("mic")
    speaker_cfg = Config("speaker")
    segmenter = SpeechSegmenter(
        mic_cfg.sample_rate,
        threshold_db=args.vad_threshold_db,
        pre_roll_s=args.pre_roll,
        trailing_silence_s=args.trailing_silence,
        start_grace_s=args.wake_grace,
        max_utterance_s=args.max_utterance,
    )
    last_wake_active = False
    pending_wake = False
    mic_last_update = time.monotonic()
    mic_reconnect_after_s = 1.0
    mic_shape_logged = False

    with (
        LedStatus() as leds,
        Reader("wakeword.state", keeptime=False) as wakeword,
        Writer(
            "speaker.audio",
            Type("speaker_audio"),
            keeptime=False,
            buf_ms=400,
        ) as speaker,
    ):
        def announce(text: str) -> None:
            # Results of background actions, such as a heart-rate scan.
            print(f"[local-assistant] Reply: {text}")
            action_controller.speak(
                play_speech, speaker, synthesizer, text, speaker_cfg, args.volume
            )

        action_controller.bind(speaker, speaker_cfg, leds, announce=announce)
        microphone = Reader("mic.audio", keeptime=False).__enter__()
        try:
            print("[local-assistant] Ready. Say: Hey BracketBot, then your question.")
            while True:
                triggered = False
                if wakeword.ready():
                    active = bool(wakeword.data["active"])
                    triggered = active and not last_wake_active
                    last_wake_active = active
                    if triggered:
                        # Reopen at the turn boundary. A long-lived reader can
                        # remain mapped to a stale mic slot even while the mic
                        # daemon and other readers are healthy. The user waits
                        # for cyan before asking, so no query audio is lost.
                        microphone.__exit__(None, None, None)
                        microphone = Reader(
                            "mic.audio", keeptime=False
                        ).__enter__()
                        mic_last_update = time.monotonic()
                        segmenter.reset()
                        pending_wake = True
                        leds.set("listening")
                        print("[local-assistant] Wake phrase detected; recording...")

                if not microphone.ready():
                    if time.monotonic() - mic_last_update >= mic_reconnect_after_s:
                        microphone.__exit__(None, None, None)
                        microphone = Reader(
                            "mic.audio", keeptime=False
                        ).__enter__()
                        mic_last_update = time.monotonic()
                        print(
                            "[local-assistant] Mic reader stale; reconnected",
                            flush=True,
                        )
                    time.sleep(0.01)
                    continue
                mic_last_update = time.monotonic()
                raw_audio = microphone.data["audio"]
                if not mic_shape_logged:
                    # More than one channel means the direction a voice came
                    # from can be estimated to pick the first turn.
                    print(f"[local-assistant] mic frame shape {raw_audio.shape}", flush=True)
                    mic_shape_logged = True
                audio = raw_audio.reshape(-1).copy()
                boosted = (
                    audio.astype(np.float32) * args.mic_gain
                ).clip(-32768, 32767).astype(np.int16)
                if pending_wake:
                    # A wake event and mic frame are produced by independent
                    # daemons. Keep the wake latched until a real audio frame
                    # arrives instead of dropping it after one loop iteration.
                    triggered = True
                    pending_wake = False
                if args.always_listen and not segmenter.recording:
                    rms = max(
                        1.0,
                        float(np.sqrt(np.mean(boosted.astype(np.float64) ** 2))),
                    )
                    dbfs = 20.0 * np.log10(rms / 32768.0)
                    triggered = dbfs >= args.vad_threshold_db

                utterance_audio = segmenter.push(boosted, triggered=triggered)
                if utterance_audio is None:
                    continue
                leds.set("processing")
                try:
                    transcript = transcriber.transcribe(
                        utterance_audio, mic_cfg.sample_rate
                    )
                    if not transcript:
                        print("[local-assistant] No speech recognized")
                        continue
                    print(f"[local-assistant] Heard: {transcript}")
                    decision = router.route(transcript)
                    reply = decision.reply or "I did not understand that."
                    print(f"[local-assistant] Reply: {reply}")
                    leds.set("speaking")
                    if decision.action in SILENT_ACTIONS and decision.action_started:
                        pass
                    else:
                        action_controller.speak(
                            play_speech,
                            speaker,
                            synthesizer,
                            reply,
                            speaker_cfg,
                            args.volume,
                        )
                except LocalVoiceError as exc:
                    print(f"[local-assistant] {exc}")
                    leds.set("error")
                    time.sleep(0.8)
                except Exception as exc:
                    # A failed network provider or malformed response must end
                    # only this turn, never the always-on assistant process.
                    print(
                        f"[local-assistant] Turn failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    leds.set("error")
                    time.sleep(0.8)
                finally:
                    leds.set("idle")
                    segmenter.reset()
        finally:
            action_controller.unbind()
            microphone.__exit__(None, None, None)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Gemini-free BracketBot voice assistant")
    parser.add_argument("--env", type=Path, default=Path(__file__).parent.parent / ".env")
    parser.add_argument("--text", help="route one typed query without robot audio")
    parser.add_argument("--speak-text", help="synthesize one phrase on the robot speaker")
    parser.add_argument(
        "--preflight-gesture",
        choices=("wave", "salute", "handshake", "fist bump", "hug", "namaste", "dance"),
        help="run fresh IMU, arm-entry, and depth checks without moving",
    )
    parser.add_argument("--mic-gain", type=float, default=3.0)
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--vad-threshold-db", type=float, default=-38.0)
    parser.add_argument("--pre-roll", type=float, default=1.5)
    parser.add_argument(
        "--trailing-silence",
        type=float,
        default=1.5,
        help="seconds of silence after speech before submitting the turn",
    )
    parser.add_argument(
        "--wake-grace",
        type=float,
        default=1.5,
        help="minimum listening time after the wake-word trigger",
    )
    parser.add_argument("--max-utterance", type=float, default=8.0)
    parser.add_argument("--always-listen", action="store_true")
    parser.add_argument(
        "--reminder-db",
        type=Path,
        help="persistent reminder SQLite path (default: BAYMAX_REMINDER_DB_PATH or user state)",
    )
    parser.add_argument(
        "--timezone",
        help="IANA timezone for reminders (default: BAYMAX_TIMEZONE or system timezone)",
    )
    parser.add_argument(
        "--whisper-bin",
        default="/home/bracketbot/.local/share/whisper.cpp/build/bin/whisper-cli",
    )
    parser.add_argument(
        "--whisper-model",
        default="/home/bracketbot/.local/share/whisper.cpp/models/ggml-base.en.bin",
    )
    parser.add_argument("--whisper-threads", type=int, default=6)
    parser.add_argument("--espeak-bin", default="espeak-ng")
    parser.add_argument("--espeak-voice", default="en-us")
    parser.add_argument("--espeak-speed", type=int, default=165)
    parser.add_argument(
        "--tts-url",
        default=os.environ.get("LOCAL_TTS_URL", ""),
        help="optional natural-voice service on the private USB link",
    )
    parser.add_argument(
        "--rppg-script",
        type=Path,
        default=Path(__file__).parent.parent / "rppg" / "robot_rppg.py",
        help="read-only head-camera heart-rate scan used for heart-rate and checkup requests",
    )
    parser.add_argument(
        "--person-tracker-script",
        type=Path,
        default=Path(__file__).parent.parent / "person" / "person_tracker.py",
        help="turns in place to face the person before camera actions",
    )
    parser.add_argument(
        "--no-person-finder",
        action="store_true",
        help="never turn to look for the person; use whatever is in view",
    )
    args = parser.parse_args()

    load_env(args.env)
    person_finder = None
    if not args.no_person_finder:
        person_finder = PersonTrackerClient(args.person_tracker_script)
        # Start now so the face model is warm and it is already remembering
        # where people are before the first request.
        person_finder.start()
    gesture_controller = RecordedGestureController(
        Path(__file__).parent / "movements"
    )
    action_controller = VoiceActionController(
        gesture_controller,
        Path(__file__).parent.parent / "play_sound" / "wavs",
        heart_rate_scanner=RppgScanner(args.rppg_script),
        person_finder=person_finder,
        reminder_db_path=args.reminder_db or default_reminder_db_path(),
        reminder_timezone=args.timezone or default_timezone_name(),
    )
    router = VoiceRouter(
        OpenRouterClient(
            response_cache=default_question_response_cache(),
            seed_pairs=default_seed_pairs(),
        ),
        action_executor=action_controller.start,
        stop_executor=action_controller.stop,
        reminder_executor=action_controller.schedule_reminder,
        reminder_cancel_executor=action_controller.cancel_reminders,
        reminder_list_executor=action_controller.list_reminders,
    )
    transcriber = WhisperCppTranscriber(
        args.whisper_bin,
        args.whisper_model,
        threads=args.whisper_threads,
    )
    offline_synthesizer = EspeakSynthesizer(
        args.espeak_bin,
        voice=args.espeak_voice,
        words_per_minute=args.espeak_speed,
    )
    synthesizer = (
        FallbackSynthesizer(HttpTtsSynthesizer(args.tts_url), offline_synthesizer)
        if args.tts_url
        else offline_synthesizer
    )
    transcriber.validate()
    synthesizer.validate()

    print(
        "[local-assistant] "
        f"OpenRouter={'yes' if router.llm.configured else 'no'} "
        f"Browserbase={'yes' if router.llm.web_search.configured else 'no'}"
    )
    try:
        if args.preflight_gesture:
            passed, status = gesture_controller.preflight(args.preflight_gesture)
            print(status)
            if not passed:
                raise SystemExit(2)
            return
        if args.text:
            print(answer_text(router, args.text))
            # Let a background scan finish and print its result before exit.
            action_controller.wait()
            return
        if args.speak_text:
            speaker_cfg = Config("speaker")
            with Writer(
                "speaker.audio", Type("speaker_audio"), keeptime=False, buf_ms=400
            ) as speaker:
                play_speech(speaker, synthesizer, args.speak_text, speaker_cfg, args.volume)
            return
        run_voice(args, router, transcriber, synthesizer, action_controller)
    finally:
        action_controller.close()
        gesture_controller.close()
        if person_finder is not None:
            person_finder.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[local-assistant] Stopped")
