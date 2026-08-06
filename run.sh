#!/usr/bin/env bash
#
# AVEP — Launch the FastAPI server (web UI + API).
# Uses the project virtualenv (.venv). Falls back to creating it if missing.
#
# Usage:
#   ./run.sh                 # start server on http://localhost:8000
#   ./run.sh --port 9000     # custom port
#   HOST=127.0.0.1 ./run.sh  # bind to localhost only
#
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
PYTHON="$VENV/bin/python"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# Parse --port / --host flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Ensure venv exists
if [[ ! -x "$PYTHON" ]]; then
  echo "[run] No virtualenv found — creating one with uv..."
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV" --python 3.11
    VIRTUAL_ENV="$PWD/$VENV" uv pip install -r requirements.txt
  else
    python3 -m venv "$VENV"
    "$PYTHON" -m pip install -r requirements.txt
  fi
fi

# Check ffmpeg is on PATH (needed for audio extract + render)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[run] WARNING: ffmpeg not found on PATH. Install via 'brew install ffmpeg'."
fi

echo "[run] Starting AVEP server → http://localhost:$PORT"
echo "[run] Web UI: http://localhost:$PORT   |   API docs: http://localhost:$PORT/docs"
exec "$PYTHON" -m uvicorn api.app:app --reload --host "$HOST" --port "$PORT"
