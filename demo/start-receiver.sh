#!/bin/sh
# Start the dashboard receiver on all interfaces so the board can reach it (http://<laptop-ip>:8000, e.g. http://169.254.125.210:8000 over USB).
cd "$(dirname "$0")/.." || exit 1
set -a; . demo/receiver.env; set +a
mkdir -p dashboard/var
exec dashboard/.venv/bin/python -m dashboard --host 0.0.0.0 --port 8000
