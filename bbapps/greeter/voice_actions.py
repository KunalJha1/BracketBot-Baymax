"""Robot-local execution for voice actions.

The lightweight assistant already owns the speaker and LED writers.  This
controller deliberately reuses those owners instead of starting competing
BBOS writers for sounds, lights, and routines.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import wave
from zoneinfo import ZoneInfo

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
# The scan estimates once a second, but speaking every estimate would queue
# speech faster than the speaker can drain it. Speak a few spaced readings and
# let the rest go to the log only.
SPOKEN_TICKS = 4
TICK_SPACING_S = 4.0
# Actions aimed at a person: face them before starting. A missing person tracker
# leaves these exactly as they were (no turning, no search).
PERSON_GESTURES = frozenset({"handshake", "fist bump", "hug"})
# A sound cue played partway through a gesture: (file, seconds after the
# gesture starts). The fist bump's cue lands as the fist pulls back from the
# bump, which is 2.5 s into the recording at 0.6x speed plus the ease-in.
GESTURE_SOUNDS = {"fist bump": ("fist_bump_balalala.wav", 4.8)}
# Turn to face the speaker and stop there: the person finder with no gesture.
LOOK_ACTIONS = frozenset({"look-at-me"})
TRACKER_MISSING_MESSAGE = "I can't look for people right now."
NOT_FOUND_MESSAGE = (
    "I can't see you. Could you step in front of my camera and ask me again?"
)
# Drive behind a person. The follow runner owns drive.ctrl and led.ctrl while it runs.
FOLLOW_ACTIONS = frozenset({"follow-me"})
FOLLOW_MISSING_MESSAGE = "Following is not installed on this robot yet."
FOLLOW_HEARTBEAT_S = 0.25         # the runner stops itself after 1 s of silence
# Same colours the dashboard-run follower shows (follow_core.STATE_LED).
FOLLOW_STATE_LED = {
    "SEARCHING": ((70, 125, 255), "pulse"),
    "FOLLOWING": ((70, 220, 120), "solid"),
    "BLOCKED": ((255, 160, 0), "solid"),
    "LOST": ((255, 160, 0), "blink"),
}
FOLLOW_STATE_MESSAGES = {
    "FOLLOWING": "Okay. I'm following you.",
    "LOST": "I lost you. Please stand in front of me.",
    "BLOCKED": "Something is in my way.",
}
# Creep up to a person the vision app has confirmed is lying on the ground, then
# ask if they are okay. Started by GroundAlertWatcher, never by a spoken request.
GROUND_ACTIONS = frozenset({"check-on-person"})
GROUND_APPROACH_V_MAX = 0.05
GROUND_START_MESSAGE = (
    "I think someone is on the ground. I'm coming over to check on you. "
    "Say hey BracketBot, stop, to cancel."
)
GROUND_ALERT_FILE = Path("/tmp/bracketbot_ground_alert.json")
GROUND_ARRIVAL_FILE = Path("/tmp/bracketbot_ground_arrived.json")  # check_in.GROUND_ARRIVAL_FILE
GROUND_ALERT_MAX_AGE_S = 1.0
GROUND_REARM_CLEAR_S = 10.0
# A short chirp while driving behind someone, so they can tell it is still there
# without looking back. Silent while searching or lost: those states speak.
FOLLOW_CHIRP_SOUND = "follow_chirp.wav"
FOLLOW_CHIRP_STATES = frozenset({"FOLLOWING", "BLOCKED"})
FOLLOW_CHIRP_PERIOD_S = 7.5
FOLLOW_CHIRP_VOLUME = 0.5
# A green pulse makes the capture state unmistakable for the person being
# scanned, and stays distinct from the assistant's blue thinking light.
SCAN_LED = ((70, 220, 120), "pulse")
CLEAR_FOREHEAD_MESSAGE = "Please move any hair off your forehead, then hold still."
_FIRST_TO_SECOND_PERSON = {
    "my": "your",
    "mine": "yours",
    "myself": "yourself",
    "me": "you",
    "i": "you",
    "i'm": "you're",
    "i've": "you've",
    "i'll": "you'll",
}
_FIRST_PERSON = re.compile(r"\b(?:i'm|i've|i'll|myself|mine|my|me|i)\b(?!')")
# Rising two-note ding played the moment a reminder or timer is stored, so the
# person knows it took before the spoken confirmation starts.
REMINDER_SET_CHIME = ((880.0, 0.11), (1318.5, 0.2))
REMINDER_SET_LED = ((255, 185, 40), "blink", 1.2)
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

    def scan(self, cancel: threading.Event, on_tick=None, on_guidance=None) -> dict | None:
        """Run the scan, optionally reporting per-second estimates as they arrive.

        ``on_tick(bpm, progress)`` is called for every intermediate estimate the
        scan publishes, with ``bpm`` None until the signal locks. It runs on this
        thread, so a slow tick handler (speaking one, for instance) only delays
        the next tick; the scan process keeps measuring either way.
        """
        if not self.installed:
            raise HeartRateScanError(f"scan script is missing: {self.script}")
        command = [
            self.uv_bin, "run", "--quiet", self.script.name,
            "--duration", str(self.duration_s),
            "--progress-json",
        ]
        process = subprocess.Popen(
            command,
            cwd=self.script.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.timeout_s
        lines: "queue.Queue[str | None]" = queue.Queue()

        def pump():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=pump, name="rppg-stdout", daemon=True)
        reader.start()

        def stop(reason: str | None):
            # uv runs the scan in a child Python; stop both.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if reason is not None:
                raise HeartRateScanError(reason)

        report = None
        while True:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                if cancel.is_set():
                    stop(None)
                    return None
                if time.monotonic() > deadline:
                    stop("scan timed out")
                continue
            if line is None:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # uv or a library wrote a non-JSON line; ignore it.
            if not isinstance(message, dict):
                continue
            if "result" in message:
                report = message
                continue
            if on_guidance is not None and message.get("guidance"):
                on_guidance(str(message["guidance"]))
                continue
            if on_tick is not None and "progress" in message:
                bpm = message.get("bpm")
                on_tick(
                    None if bpm is None else float(bpm),
                    float(message.get("progress") or 0.0),
                )
            if cancel.is_set():
                stop(None)
                return None

        process.wait()
        if cancel.is_set():
            return None
        if report is None:
            stderr = process.stderr.read() if process.stderr else ""
            tail = (stderr or "").strip().splitlines()[-1:] or ["no output"]
            raise HeartRateScanError(
                f"scan exited with code {process.returncode}: {tail[0]}"
            )
        result = report.get("result")
        return result if isinstance(result, dict) else None


class FollowRunner:
    """Run ``robot_follow.py`` (depth person tracking + PID) as a separate process.

    The runner stops the base by itself within a second if this process stops
    sending heartbeats, so a crashed or killed assistant cannot leave it driving.
    """

    def __init__(self, script: Path, *, python_bin: str | Path | None = None, v_max: float = 0.30):
        self.script = Path(script)
        self.python_bin = str(python_bin or Path.home() / "bbos" / ".venv" / "bin" / "python")
        self.v_max = v_max

    @property
    def installed(self) -> bool:
        return self.script.is_file()

    def follow(self, cancel: threading.Event, on_state=None) -> str:
        """Follow until cancelled or the runner exits; returns the runner's last log line."""
        lines = self._run(
            [
                "--v-max", str(self.v_max),
                "--relock",                # keep following until told to stop
                "--no-odom-check",         # false-alarms whenever this balancing base turns
                "--human-gate",            # lock onto people the vision app's YOLO sees, not shapes
            ],
            cancel, on_state,
        )
        return lines[-1] if lines else ""

    def check_on_person(self, cancel: threading.Event, on_state=None) -> list[str]:
        """Approach the one confirmed ground pose; returns the runner's log lines.

        The runner keeps every ground-approach limit (0.05 m/s, 1 m outside the
        body envelope, fresh single target, 120 s). It prints the check-in line
        instead of playing it, because this process owns the speaker.
        """
        return self._run(
            ["--ground-approach", "--no-speech", "--v-max", str(GROUND_APPROACH_V_MAX)],
            cancel, on_state,
        )

    def _run(self, mode_args: list[str], cancel: threading.Event, on_state=None) -> list[str]:
        process = subprocess.Popen(
            [
                self.python_bin, self.script.name, *mode_args,
                # Ours, and idle: the action lock keeps it from turning while we drive.
                "--ignore-writer", "person_tracker.py",
                "--no-led",                # the assistant holds BBOS's only led.ctrl writer
            ],
            cwd=self.script.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        lines: list[str] = []

        def pump():
            for line in process.stdout:
                line = line.strip()
                if not line.startswith("[follow] "):
                    continue  # FOLLOW_STATUS telemetry is for the dashboard
                print(line, flush=True)
                lines.append(line.removeprefix("[follow] "))
                if on_state is not None and lines[-1].startswith("state "):
                    on_state(lines[-1].removeprefix("state "))

        reader = threading.Thread(target=pump, name="follow-stdout", daemon=True)
        reader.start()

        def send(message: str) -> bool:
            try:
                process.stdin.write(json.dumps({"type": message}) + "\n")
                process.stdin.flush()
                return True
            except (OSError, ValueError):
                return False

        try:
            while process.poll() is None and not cancel.wait(FOLLOW_HEARTBEAT_S):
                send("heartbeat")
            if process.poll() is None:
                send("stop")               # the runner ramps to zero and exits
                process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            pass
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=2.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)   # drive.ctrl times out in 0.1 s
                except ProcessLookupError:
                    pass
                process.wait()
            reader.join(timeout=1.0)
        return lines


def note_ground_arrival(path: Path = GROUND_ARRIVAL_FILE) -> None:
    """Tell the vision app the arrival question was just asked (see check_in.py)."""
    try:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"arrived_at": time.time()}))
        os.replace(temporary, path)
    except OSError as exc:
        print(f"[voice-action] could not signal ground arrival: {exc}", flush=True)


class GroundAlertWatcher:
    """Start one check-in when the vision app confirms a person lying on the ground.

    One attempt per alert episode: after an attempt starts, whether it arrives,
    is refused, or is cancelled with "stop", nothing restarts until the alert
    has stayed clear for a while. A busy assistant keeps the attempt pending.
    """

    def __init__(
        self,
        controller,
        path: Path = GROUND_ALERT_FILE,
        *,
        poll_s: float = 0.5,
        rearm_clear_s: float = GROUND_REARM_CLEAR_S,
        clock=time.monotonic,
        wall=time.time,
    ):
        self.controller = controller
        self.path = Path(path)
        self.poll_s = poll_s
        self.rearm_clear_s = rearm_clear_s
        self.clock, self.wall = clock, wall
        self.armed = True
        self._clear_since: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _alerting(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            age = self.wall() - float(payload["published_at"])
            return (
                payload["status"] == "alert"
                and 0.0 <= age <= GROUND_ALERT_MAX_AGE_S
                # Two people down is ambiguous; the runner would refuse to move.
                and len(payload["alerts"]) == 1
            )
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def poll(self) -> bool:
        """One watch step; returns whether a check-in was started."""
        if not self._alerting():
            now = self.clock()
            if self._clear_since is None:
                self._clear_since = now
            if now - self._clear_since >= self.rearm_clear_s:
                self.armed = True
            return False
        self._clear_since = None
        if not self.armed or self.controller.listening.is_set():
            return False
        started, message = self.controller.start("check-on-person")
        if started:
            self.armed = False
            print("[ground-watch] confirmed alert: starting check-in", flush=True)
        return started

    def start(self) -> None:
        def run():
            while not self._stop.wait(self.poll_s):
                try:
                    self.poll()
                except Exception as exc:
                    print(f"[ground-watch] poll failed: {exc}", flush=True)

        self._thread = threading.Thread(target=run, name="ground-alert-watch", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def tick_message(bpm: float, index: int) -> str:
    """Short spoken reading for an in-progress scan. Kept terse so the next
    estimate is not still waiting on the speaker."""
    if index == 0:
        return f"I'm reading about {round(bpm)} beats per minute."
    return f"About {round(bpm)}."


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
        follow_runner=None,
        reminder_db_path: str | Path | None = None,
        reminder_timezone: str | None = None,
        reminder_scheduler=None,
    ):
        self.gesture_controller = gesture_controller
        self.assets_dir = Path(assets_dir)
        self.heart_rate_scanner = heart_rate_scanner
        self.person_finder = person_finder
        self.follow_runner = follow_runner
        self.speaker = None
        self.speaker_cfg = None
        self.leds = None
        self.announce = print
        self.speaker_lock = threading.Lock()
        # Set by the assistant while it records a wake-word turn, so background
        # sounds stay out of the audio whisper has to understand ("stop").
        self.listening = threading.Event()
        self._operation_lock = threading.Lock()
        self._cancel = threading.Event()
        self._last_find: dict = {}
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
        if seconds < 60 or seconds != int(seconds):
            return f"{seconds:g} {'second' if seconds == 1 else 'seconds'}"
        remaining = int(seconds)
        parts = []
        for unit_seconds, singular, plural in (
            (3600, "hour", "hours"),
            (60, "minute", "minutes"),
            (1, "second", "seconds"),
        ):
            count, remaining = divmod(remaining, unit_seconds)
            if count:
                parts.append(f"{count} {singular if count == 1 else plural}")
        return " ".join(parts)

    @staticmethod
    def _second_person(message: str) -> str:
        """Say "take your meds" back to the person who said "take my meds"."""
        return _FIRST_PERSON.sub(
            lambda match: _FIRST_TO_SECOND_PERSON[match.group(0)], message
        )

    @staticmethod
    def _due_text(reminder) -> str:
        """A speakable local due time: "5:30 PM", "9 AM tomorrow"."""
        due = datetime.fromtimestamp(reminder.due_at, ZoneInfo(reminder.timezone))
        now = datetime.now(due.tzinfo)
        hour = due.hour % 12 or 12
        text = f"{hour}:{due.minute:02d}" if due.minute else f"{hour}"
        text += " AM" if due.hour < 12 else " PM"
        days = (due.date() - now.date()).days
        if days == 1:
            text += " tomorrow"
        elif days != 0:
            text += f" on {due.strftime('%A, %B')} {due.day}"
        return text

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
            f"Reminder: {self._second_person(reminder.message)}."
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
        print(
            f"[voice-action] {reminder.kind if hasattr(reminder, 'kind') else 'reminder'} "
            f"set: {delay:g}s, message={reminder.message!r}",
            flush=True,
        )
        self._acknowledge_reminder_set()

        due_text = getattr(reminder, "due_text", None)
        when = f"at {due_text}" if due_text else f"in {self._duration_text(delay)}"
        if reminder.message:
            connector = getattr(reminder, "connector", "to")
            message = self._second_person(reminder.message)
            return True, f"Okay. I'll remind you {when} {connector} {message}."
        if due_text:
            return True, f"Okay. Your alarm is set for {due_text}."
        return True, f"Okay. Your timer is set for {self._duration_text(delay)}."

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
            label = (
                self._second_person(reminder.message) if reminder.message else "a timer"
            )
            remaining = reminder.due_at - time.time()
            if 0 < remaining < 3600:
                due = f"in {self._duration_text(max(1, round(remaining / 60)) * 60)}"
                if remaining < 60:
                    due = f"in {self._duration_text(round(remaining))}"
            else:
                due = f"at {self._due_text(reminder)}"
            descriptions.append(f"{label}, {due}")
        extra = len(reminders) - len(descriptions)
        suffix = f", plus {extra} more" if extra else ""
        return True, "You have " + "; ".join(descriptions) + suffix + "."

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

    def _start_gesture_sound(self, action: str) -> None:
        """Play the gesture's sound cue at its moment, unless the gesture ends first."""
        cue = GESTURE_SOUNDS.get(action)
        if cue is None or self.speaker is None or self.speaker_cfg is None:
            return
        filename, delay = cue
        # A high fist bump raises the lift first, so the recording starts later.
        delay += getattr(self.gesture_controller, "lead_delay_seconds", 0.0)

        def run():
            if self._cancel.wait(delay) or not self.gesture_controller.running():
                return
            print(f"[voice-action] {action} sound cue: {filename}", flush=True)
            try:
                self._play_sound(self.assets_dir / filename, volume=0.9)
            except Exception as exc:
                print(f"[voice-action] {action} sound skipped: {exc}", flush=True)

        threading.Thread(target=run, name=f"voice-sound-{action}", daemon=True).start()

    def _release_when_gesture_finishes(self, action: str) -> None:
        self._start_gesture_sound(action)
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
            | LOOK_ACTIONS | FOLLOW_ACTIONS | GROUND_ACTIONS
        ):
            return False, f"Voice action '{action}' is not installed."
        if not self._operation_lock.acquire(blocking=False):
            return False, "Another voice action is already running."
        self._cancel.clear()

        if action in HEALTH_ACTIONS:
            return self._start_heart_rate_scan(checkup=action == "checkup")

        if action in LOOK_ACTIONS:
            return self._start_look_at_me()

        if action in FOLLOW_ACTIONS:
            return self._start_follow()

        if action in GROUND_ACTIONS:
            return self._start_ground_check()

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

    def _face_person(
        self,
        purpose: str,
        *,
        check_distance: bool = True,
        require_tracker: bool = False,
        defer_close: bool = False,
    ) -> bool:
        """Turn in place to find and face the person. False means do not continue."""
        if self.person_finder is None:
            if require_tracker:
                self.announce(TRACKER_MISSING_MESSAGE)
            return not require_tracker
        result = self.person_finder.acquire(purpose, self._cancel)
        self._last_find = result
        print(f"[voice-action] person finder: {result}", flush=True)
        if self._cancel.is_set() or self._shutdown.is_set():
            return False
        if result.get("unavailable"):
            if require_tracker:
                self.announce(TRACKER_MISSING_MESSAGE)
                return False
            return True               # behave as before the tracker existed
        if not result.get("found"):
            if defer_close and (result.get("refused") or result.get("error")):
                # The base is busy (teleop open, lean mode), so nobody could be
                # looked for. A gesture does not need the turn: the arm's own
                # depth check decides whether it is safe, and the fist bump
                # aims at the fist it sees, so carry on facing forward.
                print(f"[voice-action] not turning: {result.get('reason')}", flush=True)
                return True
            if result.get("refused"):
                self.announce(
                    f"I can't turn to look for you right now because {result.get('reason')}. "
                    "Please step in front of my camera."
                )
            else:
                self.announce(NOT_FOUND_MESSAGE)
            return False
        band = result.get("distance") if check_distance else None
        if band == "close" and defer_close:
            # The caller tries the move first and only asks for room if the
            # surroundings check actually refuses it.
            return True
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

    def _start_look_at_me(self):
        def run():
            try:
                # Facing someone has no reach to get right, so any distance is fine.
                found = self._face_person(
                    "look", check_distance=False, require_tracker=True
                )
                # A real turn already earned "There you are."
                if found and abs(self._last_find.get("turned_deg") or 0) < 20:
                    self.announce("I see you.")
            except Exception as exc:
                print(f"[voice-action] look-at-me failed safely: {exc}", flush=True)
            finally:
                self._operation_lock.release()

        self._thread = threading.Thread(target=run, name="voice-action-look-at-me", daemon=True)
        self._thread.start()
        return True, "Okay. Looking for you."

    def _start_follow(self):
        runner = self.follow_runner
        if runner is None or not getattr(runner, "installed", True):
            self._operation_lock.release()
            return False, FOLLOW_MISSING_MESSAGE

        def run():
            leds = self.leds
            try:
                # Lock-on needs the person within 30 degrees of straight ahead, so
                # turn to them first. Without a tracker, follow whoever stands in front.
                if not self._face_person("look", check_distance=False):
                    return
                spoken = set()
                current = [""]
                ended = threading.Event()

                def chirp():
                    path = self.assets_dir / FOLLOW_CHIRP_SOUND
                    while not ended.wait(FOLLOW_CHIRP_PERIOD_S) and not self._cancel.is_set():
                        if current[0] not in FOLLOW_CHIRP_STATES or self.listening.is_set():
                            continue
                        try:
                            self._play_sound(path, volume=FOLLOW_CHIRP_VOLUME)
                        except Exception as exc:
                            print(f"[voice-action] follow chirp off: {exc}", flush=True)
                            return

                threading.Thread(target=chirp, name="follow-chirp", daemon=True).start()

                def on_state(state):
                    current[0] = state
                    if leds is not None and state in FOLLOW_STATE_LED:
                        leds.start_effect(*FOLLOW_STATE_LED[state], 3600.0)
                    if state == "SEARCHING":
                        spoken.clear()     # a fresh lock-on after a long loss: speak again
                    message = FOLLOW_STATE_MESSAGES.get(state)
                    # Say each state once: a person weaving through a doorway
                    # would otherwise be told "I lost you" every few seconds.
                    if message is None or state in spoken or self._cancel.is_set():
                        return
                    spoken.add(state)
                    self.announce(message)

                try:
                    last = runner.follow(self._cancel, on_state=on_state)
                finally:
                    ended.set()
                if leds is not None:
                    leds.clear_effect()
                print(f"[voice-action] follow ended: {last}", flush=True)
                if not self._cancel.is_set() and not self._shutdown.is_set():
                    if last.startswith("refusing to start: "):
                        reason = last.removeprefix("refusing to start: ")
                        self.announce(f"I can't follow you right now: {reason}.")
                    else:
                        self.announce("I've stopped following.")
            except Exception as exc:
                print(f"[voice-action] follow-me failed safely: {exc}", flush=True)
            finally:
                self._operation_lock.release()

        self._thread = threading.Thread(target=run, name="voice-action-follow-me", daemon=True)
        self._thread.start()
        return True, "Okay. Stand in front of me and I'll follow you."

    def _start_ground_check(self):
        runner = self.follow_runner
        if runner is None or not getattr(runner, "installed", True) or self.speaker is None:
            self._operation_lock.release()
            return False, FOLLOW_MISSING_MESSAGE

        def run():
            try:
                self.announce(GROUND_START_MESSAGE)
                if self._cancel.is_set() or self._shutdown.is_set():
                    return
                lines = runner.check_on_person(self._cancel)
                last = lines[-1] if lines else ""
                print(f"[voice-action] ground check-in ended: {last}", flush=True)
                if self._cancel.is_set() or self._shutdown.is_set():
                    return
                said = [line.removeprefix("say ") for line in lines if line.startswith("say ")]
                if said:
                    self.announce(said[-1])
                    # The vision app holds the microphone side of unprompted
                    # conversations: it now listens for "I'm okay" or "help".
                    note_ground_arrival()
                elif last.startswith("refusing to start: "):
                    reason = last.removeprefix("refusing to start: ")
                    self.announce(f"I can't come over right now: {reason}.")
                else:
                    self.announce("I've stopped. I could not reach you safely.")
            except Exception as exc:
                print(f"[voice-action] ground check-in failed safely: {exc}", flush=True)
            finally:
                self._operation_lock.release()

        self._thread = threading.Thread(target=run, name="voice-action-ground-check", daemon=True)
        self._thread.start()
        return True, GROUND_START_MESSAGE

    def _start_person_gesture(self, action: str):
        def run():
            try:
                if self._face_person("gesture", defer_close=True):
                    try:
                        self._run_step(action)
                    except RuntimeError:
                        # Standing close is normal for a hug or a handshake,
                        # so asking everyone close to step back cost three
                        # seconds before every one of them. The depth check is
                        # what decides whether the arms have room: ask only
                        # when it says no, then check again.
                        if (self._last_find or {}).get("distance") != "close":
                            raise
                        self.announce("Could you step back a little?")
                        if self._cancel.wait(3.0):
                            return
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
                spoken = []
                guidance_spoken = set()
                last_tick = [0.0]

                def on_guidance(kind):
                    if kind != "clear-forehead" or kind in guidance_spoken:
                        return
                    if self._cancel.is_set() or self._shutdown.is_set():
                        return
                    guidance_spoken.add(kind)
                    self.announce(CLEAR_FOREHEAD_MESSAGE)

                def on_tick(bpm, progress):
                    # Every estimate is logged; only a spaced few are spoken.
                    print(
                        f"[voice-action] heart-rate tick {progress:.0%} "
                        f"bpm={'--' if bpm is None else round(bpm, 1)}",
                        flush=True,
                    )
                    if bpm is None or len(spoken) >= SPOKEN_TICKS:
                        return
                    if self._cancel.is_set() or self._shutdown.is_set():
                        return
                    now = time.monotonic()
                    if spoken and now - last_tick[0] < TICK_SPACING_S:
                        return
                    last_tick[0] = now
                    self.announce(tick_message(bpm, len(spoken)))
                    spoken.append(bpm)

                try:
                    result = scanner.scan(
                        self._cancel, on_tick=on_tick, on_guidance=on_guidance
                    )
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

    def _body_turner(self):
        """Callable that turns the base for an aimed gesture, or None when it may not."""
        turn = getattr(self.person_finder, "turn", None)
        find = self._last_find or {}
        # No turn when the tracker just said the base is not ours to move.
        if turn is None or not find.get("found") or find.get("centered") is False:
            return None

        def turn_body(delta_deg):
            result = turn(delta_deg, self._cancel)
            print(f"[voice-action] body turn {delta_deg:+.0f} deg: {result}", flush=True)
            return bool(result.get("ok"))

        return turn_body

    def _run_step(self, action: str) -> None:
        if action in GESTURES:
            turn_body = self._body_turner() if action in PERSON_GESTURES else None
            if turn_body is None:
                started, message = self.gesture_controller.start(action)
            else:
                started, message = self.gesture_controller.start(action, turn_body=turn_body)
            if not started:
                raise RuntimeError(message)
            self._start_gesture_sound(action)
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

    def _play_chime(self, notes, volume: float = 0.5) -> None:
        """Play short synthesized notes; needs no sound asset on the robot."""
        cfg = self.speaker_cfg
        if self.speaker is None or cfg is None:
            return
        tones = []
        for frequency, seconds in notes:
            t = np.arange(int(cfg.sample_rate * seconds)) / cfg.sample_rate
            fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.012)
            tones.append(np.sin(2 * np.pi * frequency * t) * np.exp(-t * 6.0) * fade)
        signal_ = (np.concatenate(tones) * volume * 32767).astype(np.int16)
        signal_ = np.pad(signal_, (0, -len(signal_) % cfg.chunk_size))
        period = cfg.chunk_size / cfg.sample_rate
        due = time.monotonic()
        with self.speaker_lock:
            for start in range(0, len(signal_), cfg.chunk_size):
                chunk = signal_[start:start + cfg.chunk_size]
                with self.speaker.buf() as frame:
                    frame["audio"] = np.repeat(chunk[:, None], cfg.channels, axis=1)
                due += period
                time.sleep(max(0.0, due - time.monotonic()))

    def _acknowledge_reminder_set(self) -> None:
        try:
            if self.leds is not None:
                self.leds.start_effect(*REMINDER_SET_LED)
            self._play_chime(REMINDER_SET_CHIME)
        except Exception as exc:
            # Feedback only: the reminder is already stored.
            print(f"[voice-action] reminder chime skipped: {exc}", flush=True)

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
