#!/usr/bin/env bash
# start.sh — boot the E2E TTS API.
# Pulls the LoRA checkpoints + reference voice from HF, then launches uvicorn.
#
# Required env:
#   HF_REPO     e.g. Shxbhxm21/voxcpm2-hinglish-lora   (holds lora_only/ + reference.wav)
#   API_TOKEN   shared bearer token
#   HF_TOKEN    read token (if HF_REPO is private)
# Optional:
#   LORA_STEP   checkpoint step (default 100)
#   PORT        listen port (default 8000)
#   HF_HOME     persistent cache dir for the ~3GB base model (e.g. /workspace/hf)
#   ASSET_DIR   where to download the LoRA (default /workspace/assets)
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load config from a .env file next to this script, if present. Keeps secrets out of
# shell history / `ps` / tmux env. Real values go in .env (gitignored); see .env.example.
if [ -f "$APP_DIR/.env" ]; then
  echo "[start] loading config from $APP_DIR/.env"
  set -a; . "$APP_DIR/.env"; set +a
fi

: "${HF_REPO:?set HF_REPO (in .env or env) to the HF repo holding lora_only/ + reference.wav}"
PORT="${PORT:-8000}"
ASSET_DIR="${ASSET_DIR:-/workspace/assets}"

echo "[start] downloading $HF_REPO -> $ASSET_DIR"
hf download "$HF_REPO" --repo-type model --local-dir "$ASSET_DIR"

export LORA_DIR="${LORA_DIR:-$ASSET_DIR/lora_only}"
export REFERENCE_WAV="${REFERENCE_WAV:-$ASSET_DIR/reference.wav}"

echo "[start] LORA_DIR=$LORA_DIR step=${LORA_STEP:-100}  port=$PORT"
cd "$APP_DIR"
exec uvicorn server:app --host 0.0.0.0 --port "$PORT"
