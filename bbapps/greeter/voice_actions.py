"""Robot-local execution for voice actions.

The lightweight assistant already owns the speaker and LED writers.  This
controller deliberately reuses those owners instead of starting competing
BBOS writers for sounds, lights, and routines.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import wave

import numpy as np

try:
    from .reminders import (
        PersistentReminderScheduler,
        default_reminder_db_path,
    )
except ImportError:  # ``uv run voice_actions.py``-style standalone imports.
    from reminders import PersistentReminderScheduler, default_reminder_db_path


GESTURES = frozenset(
    {"wave", "salute", "handshake", "fist bump", "hug", "namaste", "dance", "goodbye"}
)

LED_EFFECTS = {
    "light-calm": ((72, 205, 220), "pulse", 4.0),
    "light-ready": ((70, 220, 120), "solid", 3.0),
    "light-thinking": ((70, 125, 255), "pulse", 4.0),
    "light-celebrate": ((255, 70, 210), "blink", 4.0),
    "lights-off": ((0, 0, 0), "solid", 0.25),
}

SOUND_FILES = {
    "sound-processing": "robot_processing.wav",
    "sound-birthday": "happy_birthday.wav",
    "sound-low-battery": "low_battery_1.wav",
    "music-calm": "baymax_calm.wav",
    "music-celebration": "baymax_celebration.wav",
}

ROUTINES = {
    "welcome": ("light-ready", "wave"),
    "thinking": ("light-thinking", "sound-processing"),
    "celebrate": ("light-celebrate", "sound-birthday"),
    "double-wave": ("wave", "wave"),
    "calm-moment": ("light-calm", "music-calm"),
    "dance-party": ("light-celebrate", "music-celebration", "dance"),
}

AUDIO_ACTIONS = frozenset(SOUND_FILES)
AUDIO_ROUTINES = frozenset(
    routine
    for routine, steps in ROUTINES.items()
    if any(step in SOUND_FILES for step in steps)
)
SILENT_ACTIONS = AUDIO_ACTIONS | AUDIO_ROUTINES | {"dance"}

# Read-only camera scans. They never open a motor or arm writer.
HEALTH_ACTIONS = frozenset({"heart-rate", "checkup"})
TYPICAL_RESTING_BPM = (60, 100)
# Actions aimed at a person: face them before starting. A missing person tracker
# leaves these exactly as they were (no turning, no search).
PERSON_GESTURES = frozenset({"handshake", "fist bump", "hug"})
NOT_FOUND_MESSAGE = (
    "I can't see you. Could you step in front of my camera and ask me again?"
)
SCAN_LED = ((72, 205, 220), "pulse")
REMINDER_LED = ((255, 185, 40), "blink", 8.0)
# A reminder can land while the audience is watching the arms rather than
# the lights. A short chime ahead of the spoken text makes the delivery
# legible across a noisy room without owning the speaker for long.
REMINDER_SOUND = "sound-processing"
MAX_REMINDER_SECONDS = 365 * 24 * 60 * 60


class HeartRateScanError(RuntimeError):
    """The scan could not run; the message is safe to log."""


class RppgScanner:
    """Run ``robot_rppg.py`` as a separate process and return its result.

    The scan needs MediaPipe, OpenCV, and SciPy. Keeping it in its own uv
    environment keeps the always-on assistant light, and killing the process
    is a clean way to cancel it.
    """

    def __init__(
        self,
        script: Path,
        *,
        uv_bin: str | None = None,
        duration_s: float = 20.0,
        startup_timeout_s: float = 90.0,
    ):
        self.script = Path(script)
        self.uv_bin = uv_bin or shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
        self.duration_s = duration_s
        self.timeout_s = duration_s + startup_timeout_s

    @property
    def installed(self) -> bool:
        return self.script.is_file()

    def scan(self, cancel: threading.Event) -> dict | None:
        if not self.installed:
            raise HeartRateScanError(f"scan script is missing: {self.script}")
        command = [
            self.uv_bin, "run", "--quiet", self.script.name,
            "--duration", str(self.duration_s),
        ]
        process = subprocess.Popen(
            command,
            cwd=self.script.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.timeout_s
        stdout = stderr = ""
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancel.is_set() or time.monotonic() > deadline:
                    # uv runs the scan in a child Python; stop both.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate()
                    if cancel.is_set():
                        return None
                    raise HeartRateScanError("scan timed out")
        try:
            report = json.loads(stdout)
        except json.JSONDecodeError:
            tail = (stderr or "").strip().splitlines()[-1:] or ["no output"]
            raise HeartRateScanError(
                f"scan exited with code {process.returncode}: {tail[0]}"
            ) from None
        result = report.get("result") if isinstance(report, dict) else None
        return result if isinstance(result, dict) else None


def heart_rate_message(result: dict | None, checkup: bool = False) -> str:
    """Spoken result. Always framed as an estimate, never as a diagnosis."""
    prefix = "Your checkup is done. " if checkup else ""
    if not result or result.get("bpm") is None:
        return prefix + (
            "I couldn't get a clear heart-rate reading. Please face my camera "
            "in even light, hold still, and ask me again."
        )
    bpm = round(float(result["bpm"]))
    if not result.get("confident"):
        return prefix + (
            f"My best guess is about {bpm} beats per minute, but the signal was "
            "weak, so please don't rely on it."
        )
    message = f"Your heart rate looks like about {bpm} beats per minute."
    if checkup:
        low, high = TYPICAL_RESTING_BPM
        if low <= bpm <= high:
            message += " That's within the typical adult resting range."
        else:
            message += (
                " That's outside the typical adult resting range, though camera "
                "readings can be off. If you feel unwell, please check with a "
                "doctor or a proper medical device."
            )
    return prefix + message + " This is a camera estimate, not a medical measurement."


class VoiceActionController:
    """Serialize allowlisted voice actions and propagate cancellation."""

    def __init__(
        self,
        gesture_controller,
        assets_dir: Path,
        heart_rate_scanner=None,
        person_finder=None,
        reminder_db_path: str | Path | None = None,
        reminder_timezone: str | None = None,
        reminder_scheduler=None,
    ):
        self.gesture_controller = gesture_controller
        self.assets_dir = Path(assets_dir)
        self.heart_rate_scanner = heart_rate_scanner
        self.person_finder = person_finder
        self.speaker = None
        self.speaker_cfg = None
        self.leds = None
        self.announce = print
        self.speaker_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._cancel = threading.Event()
        self._shutdown = threading.Event()
        self._thread = None
        self._reminder_db_path = reminder_db_path
        self._reminder_timezone = reminder_timezone
        self._reminder_scheduler = reminder_scheduler
        self._owns_reminder_scheduler = False
        if reminder_db_path is not None and reminder_scheduler is None:
            # Production passes an explicit path so recovery begins as soon as
            # the assistant starts, before the first new reminder request.
            self._ensure_reminder_scheduler()

    def bind(self, speaker, speaker_cfg, leds, announce=None) -> None:
        self.speaker = speaker
        self.speaker_cfg = speaker_cfg
        self.leds = leds
        self.announce = announce or print

    def unbind(self) -> None:
        self.speaker = None
        self.speaker_cfg = None
        self.leds = None
        self.announce = print

    @staticmethod
    def _duration_text(seconds: float) -> str:
        for unit_seconds, singular, plural in (
            (3600, "hour", "hours"),
            (60, "minute", "minutes"),
            (1, "second", "seconds"),
        ):
            value = seconds / unit_seconds
            if value >= 1 and abs(value - round(value)) < 1e-9:
                count = int(round(value))
                return f"{count} {singular if count == 1 else plural}"
        return f"{seconds:g} seconds"

    def _deliver_reminder(self, reminder) -> None:
        if self._shutdown.is_set():
            raise RuntimeError("assistant is shutting down")
        if self.speaker is None:
            # Startup recovery can find an overdue reminder before BBOS audio
            # is bound. Leave it pending and retry rather than marking a print
            # to stdout as successful delivery.
            raise RuntimeError("assistant audio is not ready")
        if self.leds is not None:
            self.leds.start_effect(*REMINDER_LED)
        try:
            self._play_sound(self.assets_dir / SOUND_FILES[REMINDER_SOUND])
        except Exception:
            # The chime is an attention aid. A missing or unplayable asset must
            # not fail delivery, because that would retry the whole reminder
            # and suppress the spoken text the reminder exists to give.
            pass
        message = (
            f"Reminder: {reminder.message}."
            if reminder.message
            else "Your timer is finished."
        )
        self.announce(message)

    def _ensure_reminder_scheduler(self):
        if self._reminder_scheduler is None:
            path = self._reminder_db_path or default_reminder_db_path()
            self._reminder_scheduler = PersistentReminderScheduler(
                path,
                self._deliver_reminder,
                timezone_name=self._reminder_timezone,
            )
            self._owns_reminder_scheduler = True
        return self._reminder_scheduler

    def schedule_reminder(self, reminder):
        """Persist a non-blocking reminder that can survive assistant restarts."""
        delay = float(reminder.delay_seconds)
        if delay <= 0:
            return False, "Please choose a reminder time greater than zero."
        if delay > MAX_REMINDER_SECONDS:
            return False, "Please choose a reminder within the next year."

        self._ensure_reminder_scheduler().schedule_after(
            delay,
            reminder.message,
            source="voice",
        )

        duration = self._duration_text(delay)
        if reminder.message:
            return True, f"Okay. I'll remind you in {duration} to {reminder.message}."
        return True, f"Okay. Your {duration} timer is set."

    def set_reminder(self, due_at, message, *, timezone_name=None, source="internal"):
        """Typed internal ``set-reminder`` action for absolute timestamps."""
        reminder = self._ensure_reminder_scheduler().schedule_at(
            due_at,
            message,
            source=source,
            timezone_name=timezone_name,
        )
        return reminder.public()

    def reminder_records(self):
        """Typed internal ``list-reminders`` result."""
        return [
            reminder.public()
            for reminder in self._ensure_reminder_scheduler().list_pending()
        ]

    def list_reminders(self):
        reminders = self._ensure_reminder_scheduler().list_pending()
        if not reminders:
            return True, "You don't have any active reminders or timers."
        descriptions = []
        for reminder in reminders[:3]:
            label = reminder.message or "timer"
            due = reminder.public()["due_at_local"]
            descriptions.append(f"number {reminder.id}, {label}, due {due}")
        extra = len(reminders) - len(descriptions)
        suffix = f", plus {extra} more" if extra else ""
        return True, "Your active reminders are " + "; ".join(descriptions) + suffix + "."

    def cancel_reminder(self, reminder_id: int):
        """Typed internal ``cancel-reminder`` action for one reminder."""
        cancelled = self._ensure_reminder_scheduler().cancel(reminder_id)
        if not cancelled:
            return False, f"Reminder {int(reminder_id)} is not active."
        return True, f"Okay. I cancelled reminder {int(reminder_id)}."

    def cancel_reminders(self):
        cancelled = self._ensure_reminder_scheduler().cancel()
        if not cancelled:
            return False, "You don't have an active reminder or timer."
        noun = "reminder" if len(cancelled) == 1 else "reminders"
        return True, f"Okay. I cancelled {len(cancelled)} {noun}."

    def reminder_audit(self, reminder_id=None, limit=100):
        return self._ensure_reminder_scheduler().audit(reminder_id, limit)

    def _release_when_gesture_finishes(self, action: str) -> None:
        if action == "dance" and self.speaker is not None:
            self._play_sound(
                self.assets_dir / SOUND_FILES["music-celebration"],
                while_running=self.gesture_controller.running,
            )
        while self.gesture_controller.running() and not self._shutdown.wait(0.02):
            if self._cancel.is_set():
                self.gesture_controller.stop()
        self._operation_lock.release()

    def start(self, action: str):
        if action not in (
            GESTURES | LED_EFFECTS.keys() | SOUND_FILES.keys() | ROUTINES.keys() | HEALTH_ACTIONS
        ):
            return False, f"Voice action '{action}' is not installed."
        if not self._operation_lock.acquire(blocking=False):
            return False, "Another voice action is already running."
        self._cancel.clear()

        if action in HEALTH_ACTIONS:
            return self._start_heart_rate_scan(checkup=action == "checkup")

        if action in PERSON_GESTURES and self.person_finder is not None:
            return self._start_person_gesture(action)

        if action in GESTURES:
            started, message = self.gesture_controller.start(action)
            if not started:
                self._operation_lock.release()
                return False, message
            self._thread = threading.Thread(
                target=self._release_when_gesture_finishes,
                args=(action,),
                name=f"voice-action-{action}",
                daemon=True,
            )
            self._thread.start()
            return True, message

        if self.speaker is None or self.speaker_cfg is None or self.leds is None:
            self._operation_lock.release()
            return False, "That voice action is unavailable before audio starts."

        def run():
            try:
                steps = ROUTINES.get(action, (action,))
                for step in steps:
                    if self._cancel.is_set() or self._shutdown.is_set():
                        break
                    self._run_step(step)
            except Exception as exc:
                print(f"[voice-action] {action} failed safely: {exc}", flush=True)
            finally:
                self._operation_lock.release()

        self._thread = threading.Thread(
            target=run,
            name=f"voice-action-{action}",
            daemon=True,
        )
        self._thread.start()
        return True, f"Started {action}"

    def _face_person(self, purpose: str) -> bool:
        """Turn in place to find and face the person. False means do not continue."""
        if self.person_finder is None:
            return True
        result = self.person_finder.acquire(purpose, self._cancel)
        print(f"[voice-action] person finder: {result}", flush=True)
        if self._cancel.is_set() or self._shutdown.is_set():
            return False
        if result.get("unavailable"):
            return True               # behave as before the tracker existed
        if not result.get("found"):
            if result.get("refused"):
                self.announce(
                    f"I can't turn to look for you right now because {result.get('reason')}. "
                    "Please step in front of my camera."
                )
            else:
                self.announce(NOT_FOUND_MESSAGE)
            return False
        band = result.get("distance")
        if band in ("far", "close"):
            self.announce(
                "Please come a little closer."
                if band == "far"
                else "Could you step back a little?"
            )
            if self._cancel.wait(3.0):
                return False
        elif abs(result.get("turned_deg") or 0) >= 20:
            self.announce("There you are.")
        return True

    def _start_person_gesture(self, action: str):
        def run():
            try:
                if self._face_person("gesture"):
                    self._run_step(action)
            except Exception as exc:
                print(f"[voice-action] {action} failed safely: {exc}", flush=True)
                self.announce(str(exc))
            finally:
                self._operation_lock.release()

        self._thread = threading.Thread(target=run, name=f"voice-action-{action}", daemon=True)
        self._thread.start()
        return True, f"Finding you, then starting {action}"

    def _start_heart_rate_scan(self, checkup: bool):
        scanner = self.heart_rate_scanner
        if scanner is None or not getattr(scanner, "installed", True):
            self._operation_lock.release()
            return False, "Heart-rate scanning is not installed on this robot yet."

        def run():
            leds = self.leds
            try:
                if not self._face_person("scan"):
                    return
                if leds is not None:
                    rgb, pattern = SCAN_LED
                    leds.start_effect(rgb, pattern, 600.0)
                try:
                    result = scanner.scan(self._cancel)
                    message = heart_rate_message(result, checkup=checkup)
                except Exception as exc:
                    print(f"[voice-action] heart-rate scan failed safely: {exc}", flush=True)
                    message = "Sorry, I couldn't run the heart-rate scan right now."
                if leds is not None:
                    leds.clear_effect()
                if not self._cancel.is_set() and not self._shutdown.is_set():
                    self.announce(message)
            finally:
                self._operation_lock.release()

        self._thread = threading.Thread(target=run, name="voice-action-heart-rate", daemon=True)
        self._thread.start()
        return True, "Started heart-rate scan"

    def _run_step(self, action: str) -> None:
        if action in GESTURES:
            started, message = self.gesture_controller.start(action)
            if not started:
                raise RuntimeError(message)
            while self.gesture_controller.running() and not self._shutdown.wait(0.02):
                if self._cancel.is_set():
                    self.gesture_controller.stop()
            return

        if action in LED_EFFECTS:
            rgb, pattern, duration = LED_EFFECTS[action]
            self.leds.start_effect(rgb, pattern, duration)
            self._cancel.wait(duration)
            if self._cancel.is_set():
                self.leds.clear_effect()
            return

        self._play_sound(self.assets_dir / SOUND_FILES[action])

    def _play_sound(self, path: Path, volume: float = 0.65, while_running=None) -> None:
        if not path.is_file():
            raise RuntimeError(f"sound asset is missing: {path.name}")
        cfg = self.speaker_cfg
        with wave.open(str(path), "rb") as source:
            if (
                source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
                or source.getframerate() != cfg.sample_rate
            ):
                raise RuntimeError(f"unsupported sound format: {path.name}")
            source_channels = source.getnchannels()
            period = cfg.chunk_size / cfg.sample_rate
            due = time.monotonic()
            with self.speaker_lock:
                while not self._cancel.is_set() and not self._shutdown.is_set():
                    if while_running is not None and not while_running():
                        break
                    raw = source.readframes(cfg.chunk_size)
                    if not raw:
                        break
                    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, source_channels)
                    if source_channels == 1 and cfg.channels > 1:
                        samples = np.repeat(samples, cfg.channels, axis=1)
                    elif source_channels > 1 and cfg.channels == 1:
                        samples = samples.mean(axis=1, dtype=np.float32)[:, None]
                    samples = (samples.astype(np.float32) * volume).clip(
                        -32768, 32767
                    ).astype(np.int16)
                    if len(samples) < cfg.chunk_size:
                        padding = np.zeros(
                            (cfg.chunk_size - len(samples), cfg.channels), dtype=np.int16
                        )
                        samples = np.concatenate((samples, padding))
                    with self.speaker.buf() as frame:
                        frame["audio"] = samples
                    due += period
                    self._cancel.wait(max(0.0, due - time.monotonic()))

    def speak(self, speak, *args, **kwargs) -> None:
        with self.speaker_lock:
            speak(*args, **kwargs)

    def stop(self):
        if not self._operation_lock.locked() and not self.gesture_controller.running():
            return False, "No voice action is running."
        self._cancel.set()
        self.gesture_controller.stop()
        if self.leds is not None:
            self.leds.clear_effect()
        return True, "Okay. Stopping safely."

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current background action finishes."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def close(self, timeout: float = 8.0) -> None:
        self._shutdown.set()
        self._cancel.set()
        if self._owns_reminder_scheduler and self._reminder_scheduler is not None:
            # Closing the worker leaves scheduled rows intact for recovery on
            # the next launch. It must never translate shutdown into cancel.
            self._reminder_scheduler.close(timeout=min(timeout, 3.0))
        self.gesture_controller.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
