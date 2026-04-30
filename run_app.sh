#!/usr/bin/env bash
# Launch the LingBot-Map Gradio UI.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/Scripts/python.exe" ] && [ ! -x ".venv/bin/python" ]; then
    echo "[error] .venv not found. Run setup first." >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -x ".venv/Scripts/python.exe" ]; then
    ".venv/Scripts/python.exe" app.py "$@"
else
    ".venv/bin/python" app.py "$@"
fi
