#!/usr/bin/env bash
set -euo pipefail

robot_host="${BAYMAX_SSH_HOST:-bot}"
proxy_host="${BAYMAX_USB_HOST_IP:-192.168.55.100}"
proxy_port="${BAYMAX_PROXY_PORT:-8899}"
proxy_url="http://${proxy_host}:${proxy_port}"
tts_port="${BAYMAX_TTS_PORT:-8900}"
tts_url="http://${proxy_host}:${tts_port}/tts"
proxy_pid=""
tts_pid=""

cleanup() {
  if [[ -n "$proxy_pid" ]]; then
    kill "$proxy_pid" 2>/dev/null || true
  fi
  if [[ -n "$tts_pid" ]]; then
    kill "$tts_pid" 2>/dev/null || true
  fi
  ssh -o BatchMode=yes "$robot_host" \
    "pkill -INT -f '[l]ocal_assistant.py' || true" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if ! ifconfig | grep -q "inet ${proxy_host} "; then
  echo "USB host address ${proxy_host} is not active. Is BracketBot connected?" >&2
  exit 1
fi

uvx --from proxy-py proxy \
  --hostname "$proxy_host" \
  --port "$proxy_port" \
  --num-workers 1 \
  --log-level warning &
proxy_pid=$!

python3 scripts/local_tts_server.py \
  --host "$proxy_host" \
  --port "$tts_port" &
tts_pid=$!

for _ in {1..50}; do
  if nc -z "$proxy_host" "$proxy_port" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if ! nc -z "$proxy_host" "$proxy_port" 2>/dev/null; then
  echo "Local HTTPS proxy did not start on ${proxy_host}:${proxy_port}." >&2
  exit 1
fi
for _ in {1..50}; do
  if nc -z "$proxy_host" "$tts_port" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if ! nc -z "$proxy_host" "$tts_port" 2>/dev/null; then
  echo "Local natural-voice service did not start on ${proxy_host}:${tts_port}." >&2
  exit 1
fi

echo "Proxy ready on the private BracketBot USB link."
echo "Natural voice ready on the private BracketBot USB link."
echo "Starting Gemini-free voice assistant on ${robot_host}..."
ssh -tt "$robot_host" \
  "cd /home/bracketbot/bbapps/greeter && env HTTPS_PROXY='$proxy_url' https_proxy='$proxy_url' LOCAL_TTS_URL='$tts_url' /home/bracketbot/.local/bin/uv run --offline local_assistant.py"
