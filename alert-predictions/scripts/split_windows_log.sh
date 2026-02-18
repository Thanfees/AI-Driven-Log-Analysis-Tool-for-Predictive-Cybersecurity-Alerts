#!/usr/bin/env bash
set -euo pipefail
# Split a large Windows log into N equal chunks (line count based), default 20 parts.
# Usage: ./scripts/split_windows_log.sh raw_logs/windows/Windows.log [chunks]
# Output: raw_logs/windows/chunks/win_00.log ...
LOG_PATH=${1:-}
PARTS=${2:-20}
if [[ -z "$LOG_PATH" ]]; then
  echo "Usage: $0 <path/to/Windows.log> [parts]" >&2
  exit 1
fi
if [[ ! -f "$LOG_PATH" ]]; then
  echo "❌ File not found: $LOG_PATH" >&2
  exit 1
fi
out_dir=$(dirname "$LOG_PATH")/chunks
mkdir -p "$out_dir"
base="$out_dir/win_"
split -n l/$PARTS -d --additional-suffix=.log "$LOG_PATH" "$base"
echo "✅ Split complete: $(ls -1 $out_dir | wc -l) files in $out_dir"
