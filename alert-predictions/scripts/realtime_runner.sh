#!/usr/bin/env bash
set -euo pipefail

# One-command realtime demo: replays a log file at a given rate into a
# temporary syslog file and runs realtime inference with K-confirm.

usage() {
  cat <<'USAGE'
Usage: scripts/realtime_runner.sh -l <LOG_FILE> -m <MODEL_DIR> [options]

Required:
  -l  Input log to replay (syslog format text)
  -m  Trained model dir (contains final_model.joblib + threshold.txt)

Options:
  -o  Output CSV for predictions            [default: outputs/linux/realtime_predictions.csv]
  -w  Window seconds                        [default: 60]
  -k  K consecutive positives to confirm    [default: 3]
  -y  Year for syslog timestamps            [default: 2026]
  -r  Replay rate (lines per second)        [default: 50]
  -P  Print every window's score (verbose)
  -h  Show this help

Example:
  source "/home/hackgodx/Projects/RP/venv/bin/activate"
  bash scripts/realtime_runner.sh \
    -l raw_logs/linux/synth_80k_loanom.log \
    -m models/linux/baseline_combined_w60s_h15m \
    -o outputs/linux/realtime_demo.csv \
    -w 60 -k 3 -r 50
USAGE
}

LOG_IN=""
MODEL_DIR=""
OUT="outputs/linux/realtime_predictions.csv"
WIN_SEC="60"
KCONFIRM="3"
YEAR="2026"
RATE="50"

PRINT_RAW=0
while getopts ":l:m:o:w:k:y:r:Ph" opt; do
  case "$opt" in
    l) LOG_IN="$OPTARG" ;;
    m) MODEL_DIR="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    w) WIN_SEC="$OPTARG" ;;
    k) KCONFIRM="$OPTARG" ;;
    y) YEAR="$OPTARG" ;;
    r) RATE="$OPTARG" ;;
    P) PRINT_RAW=1 ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [[ -z "$LOG_IN" || -z "$MODEL_DIR" ]]; then
  echo "❌ Missing required args: -l <LOG_FILE> and -m <MODEL_DIR>" >&2
  usage
  exit 1
fi

if [[ ! -f "$LOG_IN" ]]; then
  echo "❌ Log file not found: $LOG_IN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/final_model.joblib" || ! -f "$MODEL_DIR/threshold.txt" ]]; then
  echo "❌ Model artifacts not found in: $MODEL_DIR (need final_model.joblib + threshold.txt)" >&2
  exit 1
fi

# Activate venv if not already
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if ! source "/home/hackgodx/Projects/RP/venv/bin/activate" 2>/dev/null; then
    echo "⚠️ Could not auto-activate venv. Ensure 'python' points to your project venv." >&2
  fi
fi

mkdir -p "$(dirname "$OUT")"

PIPE=$(mktemp -u /tmp/realtime_pipe_XXXXXX)
mkfifo "$PIPE"
LOG_DST=$(mktemp /tmp/realtime_syslog_XXXXXX.log)

cleanup() {
  set +e
  [[ -n "${INFER_PID:-}" ]] && kill "${INFER_PID}" 2>/dev/null || true
  [[ -n "${REPLAYER_PID:-}" ]] && kill "${REPLAYER_PID}" 2>/dev/null || true
  [[ -n "${WRITER_PID:-}" ]] && kill "${WRITER_PID}" 2>/dev/null || true
  rm -f "$PIPE" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Writer: forward pipe to a file the realtime scorer tails
cat "$PIPE" >> "$LOG_DST" &
WRITER_PID=$!

# Compute delay per line from RATE (lines/sec)
DELAY=$(python - "$RATE" << 'PY'
import sys
try:
    r = float(sys.argv[1])
    assert r > 0
except Exception:
    r = 50.0
print(f"{1.0/r:.6f}")
PY
)

echo "🟦 Replaying $LOG_IN at ${RATE} lines/sec (delay=${DELAY}s)"
awk -v delay="$DELAY" '{print; fflush(); system("sleep " delay);}' "$LOG_IN" > "$PIPE" &
REPLAYER_PID=$!

echo "🟦 Realtime infer -> $OUT (window=${WIN_SEC}s, k-confirm=${KCONFIRM})"
python src/linux/realtime/10_realtime_infer.py \
  --log-file "$LOG_DST" \
  --model-dir "$MODEL_DIR" \
  --out "$OUT" \
  --window-sec "$WIN_SEC" \
  --k-confirm "$KCONFIRM" \
  --year "$YEAR" \
  $( [[ "$PRINT_RAW" -eq 1 ]] && echo "--print-raw" ) &
INFER_PID=$!

# Wait for replay to finish, then allow a couple of window ticks, and exit
wait "$REPLAYER_PID" || true
sleep 3

echo "✅ Replay completed. Stopping realtime inference..."
exit 0
