#!/usr/bin/env bash
set -euo pipefail

install_root="${WHISPER_CPP_HOME:-/home/bracketbot/.local/share/whisper.cpp}"
model_name="${WHISPER_CPP_MODEL_NAME:-base.en}"

if command -v espeak-ng >/dev/null \
  && [[ -x "$install_root/build/bin/whisper-cli" ]] \
  && [[ -f "$install_root/models/ggml-$model_name.bin" ]]; then
  echo "Local voice dependencies are already ready."
  exit 0
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y espeak-ng git cmake g++

if [[ ! -d "$install_root/.git" ]]; then
  if [[ -e "$install_root" ]]; then
    echo "$install_root exists but is not a whisper.cpp Git checkout." >&2
    echo "Move it aside or set WHISPER_CPP_HOME to a new directory." >&2
    exit 1
  fi
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$install_root"
else
  git -C "$install_root" pull --ff-only
fi

cmake -S "$install_root" -B "$install_root/build" \
  -DGGML_CUDA=1 \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$install_root/build" -j4 --config Release

"$install_root/models/download-ggml-model.sh" "$model_name"

echo "Local voice dependencies are ready."
echo "Whisper CLI: $install_root/build/bin/whisper-cli"
echo "Whisper model: $install_root/models/ggml-$model_name.bin"
