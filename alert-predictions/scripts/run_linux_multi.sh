#!/usr/bin/env bash
set -euo pipefail

# Activate venv
source "/home/hackgodx/Projects/RP/venv/bin/activate" || {
  echo "❌ Failed to activate virtualenv. Check path." >&2
  exit 1
}

# Delegate to the Python orchestrator (accepts the same CLI flags as before)
python scripts/00_run_linux_pipeline.py "$@"

