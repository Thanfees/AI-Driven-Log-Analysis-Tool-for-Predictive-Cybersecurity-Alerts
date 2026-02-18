#!/usr/bin/env bash
set -euo pipefail

source "/home/hackgodx/Projects/RP/venv/bin/activate" || {
  echo "❌ Failed to activate virtualenv. Check path." >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: run_linux_realtime.sh -f <LOG_FILE> -m <MODEL_DIR> [-o OUT] [-w WINDOW_SEC] [-y YEAR]

  -f  Log file to tail (e.g., /var/log/syslog)       [required]
  -m  Trained baseline model dir (with threshold)    [required]
  -o  Output CSV for predictions                     [default: outputs/linux/realtime_predictions.csv]
  -w  Window seconds                                 [default: 60]
  -y  Year for syslog timestamps                     [default: 2026]
USAGE
}

LOG_FILE=""
MODEL_DIR=""
OUT="outputs/linux/realtime_predictions.csv"
WIN_SEC="60"
YEAR="2026"

while getopts ":f:m:o:w:y:h" opt; do
  case "$opt" in
    f) LOG_FILE="$OPTARG" ;;
    m) MODEL_DIR="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    w) WIN_SEC="$OPTARG" ;;
    y) YEAR="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [[ -z "$LOG_FILE" || -z "$MODEL_DIR" ]]; then
  echo "❌ Missing required args" >&2
  usage
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

python src/linux/realtime/08_realtime_infer_baseline.py \
  --log-file "$LOG_FILE" \
  --model-dir "$MODEL_DIR" \
  --out "$OUT" \
  --window-sec "$WIN_SEC" \
  --year "$YEAR"

