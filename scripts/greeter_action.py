"""Bridge a dashboard action to the robot-local greeter HTTP service.

Runs on the robot. The process remains alive for the duration of the gesture,
so the dashboard's existing PID/signal cancellation path stays effective.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from urllib import error, request


DEFAULT_URL = "http://127.0.0.1:8018"
POLL_SECONDS = 0.10
STOP_TIMEOUT = 8.0
stop_requested = False


def api_call(base_url, path, payload=None, timeout=3.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    api_request = request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(api_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(detail).get("message", detail)
        except json.JSONDecodeError:
            message = detail
        raise RuntimeError(str(message).strip() or f"greeter returned HTTP {exc.code}") from exc
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Could not reach robot vision on port 8018; ensure bbapps/emotion_greeter is running"
        ) from exc


def write_pid_file(path):
    if path is not None:
        path.write_text(f"{os.getpid()}\n")


def remove_pid_file(path):
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def main():
    global stop_requested
    parser = argparse.ArgumentParser(description="Run a greeter camera action")
    parser.add_argument("action", choices=("point", "point-left", "point-right"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()

    def on_signal(*_):
        global stop_requested
        if not stop_requested:
            print("[camera-action] stop requested; returning arm safely", flush=True)
            stop_requested = True
            try:
                api_call(args.url, "/api/action/stop", {}, timeout=2.0)
            except RuntimeError as exc:
                print(f"[camera-action] stop request failed: {exc}", flush=True)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    write_pid_file(args.pid_file)
    try:
        result = api_call(args.url, "/api/action", {"action": args.action})
        print(f"[camera-action] {result['message']}", flush=True)
        deadline = None
        seen_logs = 0
        while True:
            status = api_call(args.url, "/api/status", timeout=2.0)
            logs = status.get("movement_log") or []
            for line in logs[seen_logs:]:
                print(f"[robot] {line}", flush=True)
            seen_logs = len(logs)
            if not status.get("movement_running", False):
                if status.get("movement_error"):
                    raise RuntimeError(str(status["movement_error"]))
                break
            if stop_requested:
                deadline = deadline or time.monotonic() + STOP_TIMEOUT
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for the arm to return safely")
            time.sleep(POLL_SECONDS)
        print("[camera-action] complete", flush=True)
    except RuntimeError as exc:
        sys.exit(f"[camera-action] {exc}")
    finally:
        remove_pid_file(args.pid_file)


if __name__ == "__main__":
    main()
