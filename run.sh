#!/usr/bin/env bash
# run.sh — launch the demo (macOS/Linux). Unsets PYTHONPATH to avoid the
# global Hermes venv clobbering our project venv on Windows-style shells.
set -e
cd "$(dirname "$0")"
unset PYTHONPATH
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
python server.py
