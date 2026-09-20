#!/usr/bin/env bash
# Restart the emotion greeter if it dies.
#
# Mirrors bbapps/voice/watchdog.sh. app_manager starts apps listed in
# bbapps/.autostart at boot, but it does NOT bring one back after a crash: its
# reconcile loop drops the stale lock so the app "isn't relaunched". This
# re-creates the lock, which app_manager then picks up ("Detected external
# creation of <app>_lock. Starting.").
#
# The greeter is a long-running vision loop fed by a camera and two ONNX
# models, so a single bad frame used to end it silently -- a NaN face box from
# YuNet killed it mid-run, and it then stayed dead for over an hour with nobody
# noticing, because a stopped app looks exactly like one that was never
# started.
#
# A crash and a deliberate `stop emotion_greeter` both leave no lock and no
# process, so they cannot be told apart from here. This watchdog being active
# IS the intent: to stop the greeter for real, stop this service first.
#     sudo systemctl stop emotion-greeter-watchdog && stop emotion_greeter
set -uo pipefail

APP=emotion_greeter
ENTRY=/home/bracketbot/bbapps/emotion_greeter/main.py
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
    echo "[greeter-watchdog] $fails fast failures; backing off ${BACKOFF_S}s"
    sleep "$BACKOFF_S"
    fails=0
    continue
  fi

  echo "[greeter-watchdog] $APP is down; recreating lock (attempt $fails)"
  : > "$LOCK"
  last_start=$now
done
