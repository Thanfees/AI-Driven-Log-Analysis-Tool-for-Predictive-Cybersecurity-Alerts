#!/usr/bin/env bash
set -euo pipefail

# Activate venv (path contains spaces; keep quotes)
source "/home/hackgodx/Projects/RP/venv/bin/activate" || {
  echo "❌ Failed to activate virtualenv. Check path." >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: run_linux_single.sh -i <INPUT_(.log|.csv)> [options]

Options:
  -i  Input file (.log or .csv)                            [required]
  -w  Window size (e.g., 60s, 5min)                        [default: 60s]
  -H  Horizon minutes for y_future                         [default: 10]
  -T  Use trend features (02b)                             [flag]
  -p  Target precision for threshold                       [default: 0.60]
  -l  Min lines filter (train+infer)                       [default: 3]
  -k  K consecutive positives to confirm                   [default: 2]
  -m  Model dir for baseline                               [default: models/linux/baseline_single]
  -o  Outputs dir for predictions                          [default: outputs/linux]
  -S  Also train GRU sequence model                        [flag]
  -L  Sequence length (if -S)                              [default: 10]

Examples:
  ./scripts/run_linux_single.sh -i raw_logs/linux/linux.log -w 60s -H 30 -T -p 0.6 -l 3 -k 2
  ./scripts/run_linux_single.sh -i converted_csv/linux.log.csv -w 5min -H 30
USAGE
}

INPUT=""
WINDOW="60s"
HORIZON="10"
USE_TRENDS=0
TARGET_PREC="0.60"
MIN_LINES="3"
KCONFIRM="2"
MODEL_DIR="models/linux/baseline_single"
OUT_DIR="outputs/linux"
DO_SEQ=0
SEQ_LEN="10"

while getopts ":i:w:H:Tp:l:k:m:o:SL:h" opt; do
  case "$opt" in
    i) INPUT="$OPTARG" ;;
    w) WINDOW="$OPTARG" ;;
    H) HORIZON="$OPTARG" ;;
    T) USE_TRENDS=1 ;;
    p) TARGET_PREC="$OPTARG" ;;
    l) MIN_LINES="$OPTARG" ;;
    k) KCONFIRM="$OPTARG" ;;
    m) MODEL_DIR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    S) DO_SEQ=1 ;;
    L) SEQ_LEN="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "❌ Missing -i <input> (.log or .csv)" >&2
  usage
  exit 1
fi

mkdir -p data/linux/processed data/linux/labeled "$OUT_DIR" "$(dirname "$MODEL_DIR")"

STEM_BASE=$(basename "$INPUT")
STEM=${STEM_BASE%.*}

# Step 01: convert .log -> .csv (optional)
CSV_IN=""
if [[ "$INPUT" == *.log ]]; then
  CSV_IN="converted_csv/${STEM}.csv"
  mkdir -p "$(dirname "$CSV_IN")"
  echo "\n🟦 Convert log to CSV -> $CSV_IN"
  python src/linux/pipeline/01_convert_log_to_csv.py \
    --log-path "$INPUT" \
    --output "$CSV_IN"
else
  CSV_IN="$INPUT"
fi

# 01: windowize
WINDOWZ="data/linux/processed/${STEM}_windowz.csv"
echo "\n🟦 Windowize -> $WINDOWZ"
python src/linux/pipeline/02_windowize.py \
  --input "$CSV_IN" \
  --output "$WINDOWZ" \
  --window "$WINDOW"

# 02: rule labels
LABELED="data/linux/labeled/${STEM}_windowz_labeled.csv"
echo "\n🟦 Label (rules) -> $LABELED"
python src/linux/pipeline/03_label_windows.py \
  --input "$WINDOWZ" \
  --output "$LABELED" \
  --text-col text_with_proc

FILE_FOR_FUTURE="$LABELED"
if [[ "$USE_TRENDS" -eq 1 ]]; then
  TRENDS="data/linux/labeled/${STEM}_windowz_labeled_trends.csv"
  echo "\n🟦 Add trend features -> $TRENDS"
  python src/linux/pipeline/04_add_trends.py \
    --input "$LABELED" \
    --output "$TRENDS"
  FILE_FOR_FUTURE="$TRENDS"
fi

# 03: future labels
FUTURE="data/linux/labeled/${STEM}_future.csv"
echo "\n🟦 Make future labels (H=$HORIZON) -> $FUTURE"
python src/linux/pipeline/05_make_future_labels.py \
  --input "$FILE_FOR_FUTURE" \
  --output "$FUTURE" \
  --horizon-min "$HORIZON"

# 04: train baseline
echo "\n🟦 Train baseline -> $MODEL_DIR"
python src/linux/pipeline/06_train_baseline.py \
  --input "$FUTURE" \
  --model-dir "$MODEL_DIR" \
  --target-precision "$TARGET_PREC" \
  --min-lines "$MIN_LINES"

# 05: infer baseline (use trends table if enabled, else labeled windowz)
PRED="$OUT_DIR/${STEM}_predictions.csv"
INFER_INPUT="$LABELED"
if [[ "$USE_TRENDS" -eq 1 ]]; then
  INFER_INPUT="$TRENDS"
fi
echo "\n🟦 Infer baseline -> $PRED (from $INFER_INPUT)"
python src/linux/pipeline/07_infer_baseline.py \
  --input "$INFER_INPUT" \
  --model-dir "$MODEL_DIR" \
  --output "$PRED" \
  --min-lines "$MIN_LINES" \
  --k-confirm "$KCONFIRM"

# 06: optional GRU
if [[ "$DO_SEQ" -eq 1 ]]; then
  SEQ_DIR="models/linux/seq_single"
  echo "\n🟦 Train GRU sequence -> $SEQ_DIR"
  python src/linux/pipeline/08_train_seq_gru.py \
    --input "$FUTURE" \
    --model-dir "$SEQ_DIR" \
    --seq-len "$SEQ_LEN"
fi

echo "\n✅ DONE"
echo "  • Windows:   $WINDOWZ"
echo "  • Labeled:   $LABELED"
echo "  • Future:    $FUTURE"
echo "  • Model dir: $MODEL_DIR"
echo "  • Predict:   $PRED"
