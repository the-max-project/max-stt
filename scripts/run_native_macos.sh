#!/usr/bin/env bash
# Runs the STT FastAPI app natively on macOS (outside Docker) so it can use
# Apple's MLX/Metal backend for GPU-accelerated Whisper transcription.
# See doc/dev-mac/LargeSttDelayPlan.md (Option 2) for background.
set -euo pipefail
cd "$(dirname "$0")/.."

# Prefer python3.11 (matches the container's PYTHON_VERSION) if available.
PYBIN="python3"
if command -v python3.11 >/dev/null 2>&1; then
    PYBIN="python3.11"
fi

if [ ! -d .venv-native ]; then
    "$PYBIN" -m venv .venv-native
fi
# shellcheck disable=SC1091
source .venv-native/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt mlx-whisper

# export MODEL_SIZE="${STT_MODEL_SIZE:-large-v3}"
export MODEL_SIZE="${STT_MODEL_SIZE:-small}"
export DEVICE=mps          # unused by the mlx branch, kept for log clarity
export COMPUTE_TYPE=native # unused by the mlx branch, kept for log clarity
export UVICORN_HOST=0.0.0.0
export UVICORN_PORT="${STT_NATIVE_PORT:-8090}"

echo "Starting native STT service on http://${UVICORN_HOST}:${UVICORN_PORT} (model=${MODEL_SIZE})"
python -m uvicorn src.app:app --host "$UVICORN_HOST" --port "$UVICORN_PORT"
