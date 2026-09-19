#!/usr/bin/env bash
# Restart the voice assistant if it dies.
#
# app_manager starts apps listed in bbapps/.autostart at boot, but it does NOT
# bring one back after a crash: its reconcile loop drops the stale lock so the
# app "isn't relaunched". This re-creates the lock, which app_manager then
# picks up ("Detected external creation of <app>_lock. Starting.").
#
# A crash and a deliberate `stop voice` both leave no lock and no process, so
# they cannot be told apart from here. This watchdog being active IS the
# intent: to stop the assistant for real, stop this service first.
#     sudo systemctl stop voice-watchdog && stop voice
set -uo pipefail

APP=voice
ENTRY=/home/bracketbot/bbapps/greeter/local_assistant.py
LOCK=/dev/shm/app-${APP}_lock
POLL_S=15

# Back off after repeated fast failures so a reproducible startup crash is not
# retried forever at full speed.
FAST_FAIL_WINDOW_S=90
FAST_FAIL_LIMIT=5
BACKOFF_S=600

fails=0
last_start=0

while true; do
  sleep "$POLL_S"

  # Healthy: the process is up. Reset the failure counter once it has been
  # alive longer than the fast-fail window.
  if pgrep -f "$ENTRY" >/dev/null 2>&1; then
    now=$(date +%s)
    if (( last_start > 0 && now - last_start > FAST_FAIL_WINDOW_S )); then
      fails=0
    fi
    continue
  fi

  # A lock with no process means app_manager is mid-start; give it a cycle.
  if [[ -e "$LOCK" ]]; then
    continue
  fi

  now=$(date +%s)
  if (( now - last_start < FAST_FAIL_WINDOW_S )); then
    (( fails++ ))
  else
    fails=1
  fi

  if (( fails >= FAST_FAIL_LIMIT )); then
    echo "[voice-watchdog] $fails fast failures; backing off ${BACKOFF_S}s"
    sleep "$BACKOFF_S"
    fails=0
    continue
  fi

  echo "[voice-watchdog] $APP is down; recreating lock (attempt $fails)"
  : > "$LOCK"
  last_start=$now
done
