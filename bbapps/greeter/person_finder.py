"""Assistant-side client for the robot's person tracker.

``scripts/person_tracker.py`` runs as a long-lived child process so its face
model stays loaded and it can keep remembering where it last saw someone.
This client starts it, sends JSON-line requests, and turns cancellation into a
``cancel`` request so the tracker zeroes the base before replying.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time


UNAVAILABLE = {"found": False, "unavailable": True, "reason": "person tracker is not running"}


class PersonTrackerClient:
    def __init__(self, script: Path, *, uv_bin: str | None = None, timeout_s: float = 60.0):
        self.script = Path(script)
        self.uv_bin = uv_bin or shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
        self.timeout_s = timeout_s
        self._process: subprocess.Popen | None = None
        self._ready = threading.Event()
        self._replies: dict[int, dict] = {}
        self._reply_ready = threading.Condition()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    @property
    def installed(self) -> bool:
        return self.script.is_file()

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        with self._lock:
            if self.running():
                return True
            if not self.installed:
                print(f"[person-finder] tracker script is missing: {self.script}", flush=True)
                return False
            self._ready.clear()
            self._process = subprocess.Popen(
                [self.uv_bin, "run", "--quiet", self.script.name],
                cwd=self.script.parent,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            threading.Thread(
                target=self._read_replies, args=(self._process,), name="person-finder", daemon=True
            ).start()
            return True

    def _read_replies(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("ready"):
                self._ready.set()
                continue
            with self._reply_ready:
                self._replies[message.get("id")] = message
                self._reply_ready.notify_all()
        self._ready.clear()
        with self._reply_ready:
            self._reply_ready.notify_all()

    def _send(self, message: dict) -> bool:
        process = self._process
        if process is None or process.poll() is not None:
            return False
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def _wait_reply(self, request_id: int, deadline: float, cancel=None) -> dict | None:
        with self._reply_ready:
            while request_id not in self._replies:
                if not self.running() or time.monotonic() > deadline:
                    return None
                if cancel is not None and cancel.is_set():
                    return None
                self._reply_ready.wait(0.1)
            return self._replies.pop(request_id)

    def acquire(self, purpose: str, cancel: threading.Event, hint_deg: float | None = None) -> dict:
        """Find and face a person. Blocks until done, cancelled, or timed out."""
        if not self.start() or not self._ready.wait(timeout=45.0):
            return dict(UNAVAILABLE)
        request_id = next(self._ids)
        if not self._send({"id": request_id, "cmd": "acquire", "purpose": purpose, "hint_deg": hint_deg}):
            return dict(UNAVAILABLE)
        reply = self._wait_reply(request_id, time.monotonic() + self.timeout_s, cancel)
        if reply is not None:
            return reply
        # Cancelled or too slow: have the tracker stop turning and confirm.
        self._send({"cmd": "cancel"})
        reply = self._wait_reply(request_id, time.monotonic() + 3.0)
        if cancel.is_set():
            return {"found": False, "reason": "cancelled"}
        return reply or {"found": False, "reason": "timed out looking for a person"}

    def turn(self, delta_deg: float, cancel: threading.Event) -> dict:
        """Turn in place by ``delta_deg`` (positive left). ``ok`` says whether it happened."""
        if not self.running() or not self._ready.is_set():
            return {"ok": False, "reason": "person tracker is not running"}
        request_id = next(self._ids)
        if not self._send({"id": request_id, "cmd": "turn", "delta_deg": float(delta_deg)}):
            return {"ok": False, "reason": "person tracker is not running"}
        reply = self._wait_reply(request_id, time.monotonic() + 15.0, cancel)
        if reply is not None:
            return reply
        self._send({"cmd": "cancel"})
        self._wait_reply(request_id, time.monotonic() + 3.0)
        return {"ok": False, "reason": "cancelled" if cancel.is_set() else "timed out turning"}

    def close(self, timeout: float = 3.0) -> None:
        process = self._process
        if process is None:
            return
        try:
            process.stdin.close()          # the tracker exits when stdin closes
            process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._process = None

