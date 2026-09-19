# /// script
# requires-python = "==3.10.*"
# ///
"""Autostart shim that runs the Gemini-free voice assistant as a BBOS app.

``app_manager`` only discovers ``<app>/main.py``, but the assistant itself has
to stay in ``greeter/`` beside the modules it imports (``voice_router``,
``gesture_runtime``, ``movements/``) and beside the ``.env`` it reads from
``bbapps/``. Exec'ing it from there keeps one copy of the source instead of
duplicating the greeter into a second app folder.

Listed in ``bbapps/.autostart`` so the wake-word listener comes back on boot.
"""
import os
from pathlib import Path

ASSISTANT = Path.home() / "bbapps" / "greeter" / "local_assistant.py"

if not ASSISTANT.is_file():
    raise SystemExit(f"[voice] assistant not found: {ASSISTANT}")

# app_manager chdir's into this shim's folder; the assistant resolves its
# siblings and --env relative to its own path, so run it from greeter/.
os.chdir(ASSISTANT.parent)
os.execvp("uv", ["uv", "run", str(ASSISTANT)])
