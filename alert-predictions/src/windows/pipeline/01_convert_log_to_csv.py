#!/usr/bin/env python3
"""
01_convert_log_to_csv.py - Parse Windows CBS/CSI log files and export CSV.

Supports Windows CBS (Component-Based Servicing) and CSI (Component
Servicing Infrastructure) log formats commonly found in:
  - C:\Windows\Logs\CBS\CBS.log
  - C:\Windows\Logs\DISM\dism.log

Format: "2016-09-28 04:30:30, Info  CBS  Starting TrustedInstaller..."

Usage:
    # Single file
    python src/windows/pipeline/01_convert_log_to_csv.py \
        --log-path raw_logs/windows/windows20k.log \
        --output data/windows/raw_csv/windows20k.csv

    # Directory
    python src/windows/pipeline/01_convert_log_to_csv.py \
        --log-path raw_logs/windows \
        --output-dir data/windows/raw_csv \
        --recursive
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.windows.common.constants import (
        WINDOWS_LOG_PATTERN,
        setup_logging,
        get_logger,
    )
except ImportError:
    import re
    WINDOWS_LOG_PATTERN = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s*"
        r"(?P<level>\w+)\s+"
        r"(?P<proc>\w+)\s+"
        r"(?P<msg>.*)$"
    )

    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def iter_rows_from_log(path: Path, encoding: str, errors: str):
    """
    Stream rows from a Windows CBS/CSI log file (memory-safe).
    Yields dicts with the expected CSV columns.
    """
    last_timestamp = ""
    last_proc = ""

    with path.open("r", encoding=encoding, errors=errors, newline="") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            match = WINDOWS_LOG_PATTERN.match(line)
            if match:
                last_timestamp = match.group("ts").strip()
                last_proc = match.group("proc").strip()
                yield {
                    "line_no": line_no,
                    "timestamp": last_timestamp,
                    "host": "localhost",  # Windows CBS logs don't include hostname
                    "process": last_proc,
                    "pid": "",
                    "message": match.group("msg").strip(),
                    "raw": line,
                }
            else:
                # Continuation line (e.g., CSIPERF:TXCOMMIT;200)
                yield {
                    "line_no": line_no,
                    "timestamp": last_timestamp,
                    "host": "localhost",
                    "process": last_proc,
                    "pid": "",
                    "message": line.strip(),
                    "raw": line,
                }


def write_csv_stream(rows_iter, out_path: Path, chunk_rows: int = 500_000) -> int:
    """
    Write rows from an iterator to CSV in chunks to keep RAM low.

    Returns total rows written.
    """
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["line_no", "timestamp", "host", "process", "pid", "message", "raw"]
    count = 0
    buffer: list[dict] = []

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows_iter:
            buffer.append(row)
            count += 1
            if len(buffer) >= chunk_rows:
                writer.writerows(buffer)
                buffer.clear()
        if buffer:
            writer.writerows(buffer)

    return count


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
    """Main entry point for Windows log conversion script."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Convert Windows CBS/CSI log to CSV (file or folder)."
    )
    parser.add_argument(
        "--log-path",
        default="raw_logs/windows",
        help="Path to a .log file or a folder containing .log files"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path (file mode only). Ignored when --log-path is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/windows/raw_csv",
        help="Base directory for outputs when converting a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when --log-path is a folder"
    )
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument(
        "--errors",
        default="replace",
        choices=["strict", "ignore", "replace"]
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500_000,
        help="Rows per write chunk to limit RAM (streaming mode).",
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
            rows_written = write_csv_stream(
                iter_rows_from_log(f, args.encoding, args.errors),
                out_path,
                chunk_rows=args.chunk_rows,
            )
            count += 1
            logger.info(f"[+] {f} -> {out_path} ({rows_written} rows)")
        logger.info(f"✅ Converted {count} log file(s) to CSV under {out_base}")
        return

    # Single-file mode
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = compute_default_output(log_path, args.output_dir)

    rows_written = write_csv_stream(
        iter_rows_from_log(log_path, args.encoding, args.errors),
        output_path,
        chunk_rows=args.chunk_rows,
    )
    logger.info(f"✅ Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
