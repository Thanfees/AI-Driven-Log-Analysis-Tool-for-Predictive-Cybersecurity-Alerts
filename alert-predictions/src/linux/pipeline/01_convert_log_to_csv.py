#!/usr/bin/env python3
"""
01_convert_log_to_csv.py - Parse syslog-formatted .log files and export CSV.

Supports a single file or a directory of .log files. When a directory is
given, all matching .log files are converted and saved under an output base
folder.

Usage:
    # Single file
    python src/linux/pipeline/01_convert_log_to_csv.py \\
        --log-path /var/log/syslog \\
        --output /tmp/syslog.csv
    
    # Directory
    python src/linux/pipeline/01_convert_log_to_csv.py \\
        --log-path raw_logs/linux \\
        --output-dir data/linux/raw_csv \\
        --recursive
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.linux.common.constants import (
        SYSLOG_PATTERN,
        setup_logging,
        get_logger,
    )
except ImportError:
    import re
    SYSLOG_PATTERN = re.compile(
        r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<host>\S+)\s+"
        r"(?P<proc>[A-Za-z0-9_.-]+)"
        r"(?:\[(?P<pid>\d+)\])?"
        r":\s*(?P<msg>.*)$"
    )
    
    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def parse_log(
    path: Path, 
    encoding: str = "utf-8", 
    errors: str = "replace"
) -> pd.DataFrame:
    """
    Parse a syslog file and extract structured fields.
    
    Args:
        path: Path to the log file
        encoding: File encoding (default: utf-8)
        errors: Error handling for encoding issues (default: replace)
        
    Returns:
        DataFrame with columns: line_no, timestamp, host, process, pid, message, raw
    """
    rows = []
    
    try:
        content = path.read_text(encoding=encoding, errors=errors)
    except Exception as e:
        logger.error(f"Failed to read file {path}: {e}")
        raise
    
    for line_no, line in enumerate(content.splitlines(), start=1):
        match = SYSLOG_PATTERN.match(line)
        if match:
            rows.append({
                "line_no": line_no,
                "timestamp": match.group("ts"),
                "host": match.group("host"),
                "process": match.group("proc"),
                "pid": match.group("pid") or "",
                "message": match.group("msg"),
                "raw": line,
            })
        else:
            # Store unparseable lines with empty fields
            rows.append({
                "line_no": line_no,
                "timestamp": "",
                "host": "",
                "process": "",
                "pid": "",
                "message": "",
                "raw": line,
            })
    
    return pd.DataFrame(rows)


def compute_default_output(
    log_path: Path, 
    base_dir: Optional[str], 
    root: Optional[Path] = None
) -> Path:
    """
    Build default CSV output path, optionally rooted under base_dir.
    
    Args:
        log_path: Path to the input log file
        base_dir: Base directory for output files
        root: Root directory for computing relative paths
        
    Returns:
        Path for the output CSV file
    """
    suffix = log_path.suffix or ""
    new_suffix = suffix + ".csv" if suffix else ".csv"

    if base_dir:
        base = Path(base_dir)
        relative = None
        if root is not None:
            try:
                relative = log_path.relative_to(root)
            except ValueError:
                relative = None
        if relative is None:
            # Fallback to common roots or filename
            for r in (Path("logs"), Path("raw_logs")):
                try:
                    relative = log_path.relative_to(r)
                    break
                except ValueError:
                    continue
        if relative is None:
            relative = Path(log_path.name)
        out_path = base / relative
    else:
        out_path = log_path
    
    return out_path.with_suffix(new_suffix)


def glob_logs(root: Path, recursive: bool = False) -> Iterable[Path]:
    """
    Find all .log files under a directory.
    
    Args:
        root: Root directory to search
        recursive: Whether to search subdirectories
        
    Yields:
        Paths to .log files
    """
    if recursive:
        yield from root.rglob("*.log")
    else:
        yield from root.glob("*.log")


def main() -> None:
    """Main entry point for log conversion script."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Convert syslog text to CSV (file or folder)."
    )
    parser.add_argument(
        "--log-path", 
        default="raw_logs/linux",
        help="Path to a .log file or a folder containing .log files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (file mode only). Ignored when --log-path is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/linux/raw_csv",
        help="Base directory for outputs when converting a directory.",
    )
    parser.add_argument(
        "--recursive", 
        action="store_true", 
        help="Recurse into subdirectories when --log-path is a folder"
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--errors", 
        default="replace", 
        choices=["strict", "ignore", "replace"]
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        logger.error(f"Path does not exist: {log_path}")
        raise FileNotFoundError(f"{log_path} does not exist.")

    # Directory mode: convert all .log files under the folder
    if log_path.is_dir():
        out_base = Path(args.output_dir)
        count = 0
        for f in glob_logs(log_path, recursive=args.recursive):
            out_path = compute_default_output(f, str(out_base), root=log_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df = parse_log(f, args.encoding, args.errors)
            df.to_csv(out_path, index=False)
            count += 1
            logger.info(f"[+] {f} -> {out_path} ({len(df)} rows)")
        logger.info(f"✅ Converted {count} log file(s) to CSV under {out_base}")
        return

    # Single-file mode
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = compute_default_output(log_path, args.output_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = parse_log(log_path, args.encoding, args.errors)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
