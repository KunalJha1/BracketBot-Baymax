#!/usr/bin/env bash
set -euo pipefail

robot_host="${BAYMAX_SSH_HOST:-bot}"
proxy_host="${BAYMAX_USB_HOST_IP:-192.168.55.100}"
proxy_port="${BAYMAX_PROXY_PORT:-8899}"
proxy_url="http://${proxy_host}:${proxy_port}"
tts_port="${BAYMAX_TTS_PORT:-8900}"
tts_url="http://${proxy_host}:${tts_port}/tts"
tts_voice="${BAYMAX_TTS_VOICE:-Samantha}"
tts_rate="${BAYMAX_TTS_RATE:-160}"
tts_pitch="${BAYMAX_TTS_PITCH:-0}"
whisper_port="${BAYMAX_WHISPER_PORT:-8910}"
whisper_home="${WHISPER_CPP_HOME:-/home/bracketbot/.local/share/whisper.cpp}"
whisper_model_name="${WHISPER_CPP_MODEL_NAME:-base.en}"
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
    "pkill -INT -f '[l]ocal_assistant.py' || true; \
     pkill -INT -f '[w]hisper-server' || true" >/dev/null 2>&1 || true
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
  --port "$tts_port" \
  --voice "$tts_voice" \
  --rate "$tts_rate" \
  --pitch "$tts_pitch" &
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
echo "Syncing the voice router and safety-gated gesture runtime..."
scp -q \
  bbapps/greeter/local_assistant.py \
  bbapps/greeter/local_voice.py \
  bbapps/greeter/voice_router.py \
  bbapps/greeter/voice_actions.py \
  bbapps/greeter/reminders.py \
  bbapps/greeter/person_finder.py \
  bbapps/greeter/gesture_safety.py \
  bbapps/greeter/gesture_runtime.py \
  bbapps/greeter/speech_relay.py \
  "$robot_host:/home/bracketbot/bbapps/greeter/"
# The prepared answers must land beside the greeter modules: the robot has no
# repository around them, so this is where the router looks for the seed.
scp -q assets/response-cache-seed.json \
  "$robot_host:/home/bracketbot/bbapps/greeter/"
scp -q bbapps/greeter/movements/*.json \
  "$robot_host:/home/bracketbot/bbapps/greeter/movements/"
scp -q bbapps/mimic/recordings/dance.json \
  "$robot_host:/home/bracketbot/bbapps/mimic/recordings/dance.json"
scp -q \
  bbapps/play_sound/wavs/robot_processing.wav \
  bbapps/play_sound/wavs/happy_birthday.wav \
  bbapps/play_sound/wavs/low_battery_1.wav \
  bbapps/play_sound/wavs/baymax_calm.wav \
  bbapps/play_sound/wavs/baymax_celebration.wav \
  bbapps/play_sound/wavs/fist_bump_balalala.wav \
  "$robot_host:/home/bracketbot/bbapps/play_sound/wavs/"
echo "Syncing the read-only heart-rate scan..."
ssh -o BatchMode=yes "$robot_host" "mkdir -p /home/bracketbot/bbapps/rppg"
scp -q \
  scripts/robot_rppg.py \
  rppg.py \
  assets/models/face_detection_yunet_2026may.onnx \
  "$robot_host:/home/bracketbot/bbapps/rppg/"
# Install the scan's OpenCV/SciPy environment now, through the
# proxy, so the first "what's my heart rate" does not wait on downloads.
if ! ssh -o BatchMode=yes "$robot_host" \
  "cd /home/bracketbot/bbapps/rppg && env HTTPS_PROXY='$proxy_url' https_proxy='$proxy_url' /home/bracketbot/.local/bin/uv run --quiet robot_rppg.py --self-test"; then
  echo "Heart-rate scan environment did not install; heart-rate requests will fail until it does." >&2
fi
echo "Syncing the person tracker (turns in place to face you)..."
ssh -o BatchMode=yes "$robot_host" "mkdir -p /home/bracketbot/bbapps/person"
scp -q \
  scripts/person_tracker.py \
  scripts/camera_geometry.py \
  assets/models/face_detection_yunet_2026may.onnx \
  "$robot_host:/home/bracketbot/bbapps/person/"
if ! ssh -o BatchMode=yes "$robot_host" \
  "cd /home/bracketbot/bbapps/person && env HTTPS_PROXY='$proxy_url' https_proxy='$proxy_url' /home/bracketbot/.local/bin/uv run --quiet person_tracker.py --check-deps"; then
  echo "Person tracker environment did not install; camera actions will use whatever is in view." >&2
fi
echo "Syncing the person follower (\"follow me\")..."
ssh -o BatchMode=yes "$robot_host" "mkdir -p /home/bracketbot/bbapps/follow"
scp -q \
  scripts/robot_follow.py \
  scripts/follow_core.py \
  scripts/follow_perception.py \
  scripts/follow_calibration.py \
  "$robot_host:/home/bracketbot/bbapps/follow/"
# Keep the Whisper model resident. The CLI reloads it from disk on every
# spoken turn, which is fixed latency in front of every single answer. If the
# server does not come up the assistant simply falls back to the CLI.
echo "Starting the resident Whisper server on ${robot_host}..."
whisper_url=""
if ssh -o BatchMode=yes "$robot_host" \
  "pkill -INT -f '[w]hisper-server' || true; \
   [[ -x '$whisper_home/build/bin/whisper-server' ]] && \
   nohup '$whisper_home/build/bin/whisper-server' \
     --model '$whisper_home/models/ggml-$whisper_model_name.bin' \
     --host 127.0.0.1 --port '$whisper_port' --threads 6 \
     >/tmp/whisper-server.log 2>&1 & \
   sleep 0.2" >/dev/null 2>&1; then
  for _ in {1..60}; do
    if ssh -o BatchMode=yes "$robot_host" \
      "curl -sf -o /dev/null http://127.0.0.1:${whisper_port}/" >/dev/null 2>&1; then
      whisper_url="http://127.0.0.1:${whisper_port}"
      break
    fi
    sleep 0.25
  done
fi
if [[ -n "$whisper_url" ]]; then
  echo "Whisper model is resident on ${whisper_url}; turns skip the model load."
else
  echo "Resident Whisper server unavailable; using the whisper CLI per turn." >&2
fi

echo "Starting Gemini-free voice assistant on ${robot_host}..."
ssh -tt "$robot_host" \
  "cd /home/bracketbot/bbapps/greeter && env HTTPS_PROXY='$proxy_url' https_proxy='$proxy_url' LOCAL_TTS_URL='$tts_url' WHISPER_SERVER_URL='$whisper_url' /home/bracketbot/.local/bin/uv run --offline local_assistant.py"
