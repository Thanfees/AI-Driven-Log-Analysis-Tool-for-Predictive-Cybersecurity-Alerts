#!/usr/bin/env python3
"""
10_realtime_infer.py - Real-time anomaly prediction from live log stream.

Tails a log file and generates early warning predictions using the trained
baseline model with K-consecutive confirmation.

Usage:
    python src/linux/realtime/10_realtime_infer.py \\
        --log-file /var/log/syslog \\
        --model-dir models/linux/baseline_combined \\
        --out outputs/linux/realtime_predictions.csv \\
        --window-sec 60 \\
        --k-confirm 3
"""

import argparse
import json
import time
import re
import logging
import sys
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, Generator, List, Optional

import joblib
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.linux.common.constants import (
        SYSLOG_PATTERN,
        KW_PATTERNS,
        DEFAULT_YEAR,
        DEFAULT_K_CONFIRM,
        setup_logging,
        get_logger,
    )
except ImportError:
    SYSLOG_PATTERN = re.compile(
        r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<host>\S+)\s+"
        r"(?P<proc>[A-Za-z0-9_.-]+)"
        r"(?:\[(?P<pid>\d+)\])?"
        r":\s*(?P<msg>.*)$"
    )
    KW_PATTERNS = {
        "kw_error": r"\berror\b|\bfail(ed|ure)?\b|\bdenied\b|\bcritical\b|\bpanic\b",
        "kw_timeout": r"\btimeout\b|\btimed? out\b|\bNo route to host\b|\bConnection refused\b",
        "kw_auth": r"\bFailed password\b|\binvalid user\b|\bauthentication failure\b",
        "kw_kernel": r"\boom-killer\b|\bOut of memory\b|\bKernel panic\b",
        "kw_segfault": r"\bsegfault\b|\bcore dumped\b",
        "kw_firewall": r"\bUFW\b|\biptables\b",
    }
    DEFAULT_YEAR = 2026
    DEFAULT_K_CONFIRM = 3
    
    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def parse_syslog(line: str, year: int) -> Optional[Dict]:
    """
    Parse a syslog line into structured fields.
    
    Args:
        line: Raw log line
        year: Year for timestamps
        
    Returns:
        Dict with parsed fields or None
    """
    m = SYSLOG_PATTERN.match(line)
    if not m:
        return None
    
    ts = m.group("ts")
    try:
        dt = datetime.strptime(f"{ts} {year}", "%b %d %H:%M:%S %Y")
    except ValueError:
        dt = datetime.now()
    
    return {
        "datetime": dt,
        "proc": m.group("proc") or "",
        "msg": m.group("msg") or "",
        "raw": line.strip()
    }


def floor_time(dt: datetime, window_sec: int) -> datetime:
    """Floor datetime to window boundary."""
    seconds = int(dt.timestamp())
    floored = seconds - (seconds % window_sec)
    return datetime.fromtimestamp(floored)


def tail_f(path: Path) -> Generator[str, None, None]:
    """Tail a file like 'tail -F'."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)  # Go to end
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            yield line


def load_threshold(model_dir: Path) -> float:
    """Load threshold from JSON or text file."""
    json_path = model_dir / "threshold.json"
    txt_path = model_dir / "threshold.txt"
    
    if json_path.exists():
        data = json.loads(json_path.read_text())
        return float(data["threshold"])
    elif txt_path.exists():
        return float(txt_path.read_text().strip())
    else:
        raise FileNotFoundError(f"No threshold file found in {model_dir}")


def get_expected_columns(model) -> tuple:
    """Extract expected numeric columns from trained model."""
    expected_procs = set()
    expected_kws = set()
    expected_num_cols = []
    
    try:
        features = getattr(model, "named_steps", {}).get("features")
        if features is not None and hasattr(features, "transformers"):
            for name, trans, cols in features.transformers:
                if name == "num" and isinstance(cols, list):
                    expected_num_cols = list(cols)
                    break
        
        for c in expected_num_cols:
            base = None
            if c.endswith("_roll3_mean"):
                base = c[:-11]
            elif c.endswith("_roll6_sum"):
                base = c[:-10]
            elif c.endswith("_diff1"):
                base = c[:-6]
            elif c.startswith("cnt_proc_") or c.startswith("kw_"):
                base = c
            
            if base:
                if base.startswith("cnt_proc_"):
                    expected_procs.add(base[len("cnt_proc_"):])
                if base.startswith("kw_"):
                    expected_kws.add(base)
    except Exception as e:
        logger.warning(f"Could not extract expected columns: {e}")
    
    return expected_num_cols, expected_procs, expected_kws


def build_window_features(
    buffer_lines: List[Dict],
    current_bucket: datetime,
    expected_procs: set,
    expected_kws: set,
    expected_num_cols: List[str],
    history: Dict,
) -> Dict:
    """
    Build feature dictionary for a window.
    
    Args:
        buffer_lines: List of parsed log entries in the window
        current_bucket: Window timestamp
        expected_procs: Set of process names to track
        expected_kws: Set of keyword columns to track
        expected_num_cols: All expected numeric columns
        history: Rolling history for trend features
        
    Returns:
        Feature dictionary for the window
    """
    text_with_proc = " || ".join([f"{x['proc']}: {x['msg']}" for x in buffer_lines])
    lines_count = len(buffer_lines)
    
    row = {
        "bucket": current_bucket.isoformat(),
        "lines": lines_count,
        "text_with_proc": text_with_proc,
    }
    
    # Process counts
    if expected_procs:
        proc_counts = defaultdict(int)
        for x in buffer_lines:
            p = x["proc"]
            if p in expected_procs:
                proc_counts[p] += 1
        
        for p in expected_procs:
            base = f"cnt_proc_{p}"
            val = int(proc_counts.get(p, 0))
            row[base] = val
            
            # Rolling features
            hist = history[base]
            prev = hist[-1] if len(hist) > 0 else 0
            row[f"{base}_diff1"] = val - prev
            row[f"{base}_roll3_mean"] = (sum(list(hist)[-2:]) + val) / float(min(len(hist) + 1, 3))
            row[f"{base}_roll6_sum"] = sum(list(hist)) + val
            hist.append(val)
    
    # Keyword counts
    if expected_kws:
        compiled = {k: re.compile(KW_PATTERNS.get(k, k), re.IGNORECASE) for k in expected_kws}
        kw_counts = {k: 0 for k in expected_kws}
        
        for x in buffer_lines:
            msg = x["msg"] or ""
            for kname, rx in compiled.items():
                if rx.search(msg):
                    kw_counts[kname] += 1
        
        for kname, val in kw_counts.items():
            row[kname] = int(val)
            hist = history[kname]
            prev = hist[-1] if len(hist) > 0 else 0
            row[f"{kname}_diff1"] = val - prev
            row[f"{kname}_roll3_mean"] = (sum(list(hist)[-2:]) + val) / float(min(len(hist) + 1, 3))
            row[f"{kname}_roll6_sum"] = sum(list(hist)) + val
            hist.append(val)
    
    # Ensure all expected columns exist
    for c in expected_num_cols:
        if c not in row:
            row[c] = 0
    
    return row


def main() -> None:
    """Main entry point for realtime inference script."""
    setup_logging()
    
    ap = argparse.ArgumentParser(
        description="Real-time anomaly prediction from live log stream."
    )
    ap.add_argument("--log-file", required=True, help="Path to live log file")
    ap.add_argument("--model-dir", required=True, help="Trained model directory")
    ap.add_argument("--out", required=True, help="Output predictions CSV")
    ap.add_argument("--window-sec", type=int, default=60)
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument(
        "--k-confirm", 
        type=int, 
        default=DEFAULT_K_CONFIRM,
        help=f"Consecutive positives to confirm (default: {DEFAULT_K_CONFIRM})"
    )
    ap.add_argument("--print-raw", action="store_true", help="Print every window's score")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "final_model.joblib"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from: {model_dir}")
    model = joblib.load(model_path)
    threshold = load_threshold(model_dir)
    logger.info(f"Using threshold: {threshold:.4f}")

    # Get expected columns from model
    expected_num_cols, expected_procs, expected_kws = get_expected_columns(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize output file
    if not out_path.exists():
        out_path.write_text(
            "bucket,early_warning_score,predict_raw,predict_confirmed,lines,example_text\n"
        )

    buffer_lines: List[Dict] = []
    current_bucket: Optional[datetime] = None
    history = defaultdict(lambda: deque(maxlen=6))
    
    k = max(1, int(args.k_confirm))
    pred_buffer = deque(maxlen=k)

    logger.info(f"Starting realtime inference on: {args.log_file}")
    logger.info(f"Window: {args.window_sec}s, K-confirm: {k}")

    for line in tail_f(Path(args.log_file)):
        parsed = parse_syslog(line, args.year)
        if not parsed:
            continue

        dt = parsed["datetime"]
        bucket = floor_time(dt, args.window_sec)

        if current_bucket is None:
            current_bucket = bucket

        # Window changed - process previous window
        if bucket != current_bucket:
            row = build_window_features(
                buffer_lines,
                current_bucket,
                expected_procs,
                expected_kws,
                expected_num_cols,
                history,
            )

            row_df = pd.DataFrame([row])
            score = float(model.predict_proba(row_df)[:, 1][0])
            pred = 1 if score >= threshold else 0
            pred_buffer.append(pred)
            pred_confirm = 1 if (len(pred_buffer) == k and sum(pred_buffer) == k) else 0

            lines_count = len(buffer_lines)
            example = row["text_with_proc"][:250].replace("\n", " ").replace(",", " ")

            # Write prediction
            with out_path.open("a", encoding="utf-8") as f:
                f.write(f"{current_bucket.isoformat()},{score:.6f},{pred},{pred_confirm},{lines_count},{example}\n")

            if args.print_raw:
                print(f"{current_bucket}  score={score:.4f}  raw={pred}  lines={lines_count}  {example[:80]}")
            
            if pred_confirm == 1:
                logger.warning(
                    f"🚨 EARLY WARNING (confirmed={k}) @ {current_bucket} | "
                    f"score={score:.3f} | lines={lines_count}"
                )

            # Reset for new bucket
            buffer_lines = []
            current_bucket = bucket

        # Buffer current line
        buffer_lines.append(parsed)


if __name__ == "__main__":
    main()
