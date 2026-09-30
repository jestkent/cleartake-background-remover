#!/usr/bin/env bash
# ClearTake launcher for macOS and Linux.
set -euo pipefail
cd "$(dirname "$0")"

printf '\n  ClearTake\n  ---------\n\n'

if ! command -v python3 >/dev/null 2>&1; then
    echo "  Python 3 was not found. Install Python 3.10 or newer."
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "  WARNING: ffmpeg was not found on your PATH."
    echo "  ClearTake cannot process anything without it."
    echo "    macOS:  brew install ffmpeg"
    echo "    Debian: sudo apt install ffmpeg"
    echo
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "  First run. Setting up a virtual environment..."
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "  Installing dependencies. This takes a minute."
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt --quiet
    echo "  Done."
    echo
else
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "  Starting on http://127.0.0.1:7788"
echo "  Press Ctrl+C to stop."
echo

exec python -m backend.cli serve
