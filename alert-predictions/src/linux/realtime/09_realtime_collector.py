#!/usr/bin/env python3
"""
09_realtime_collector.py - Collect and stream live logs to CSV.

Supports collecting from file, journald, macOS log stream, or Docker containers.

Usage:
    python src/linux/realtime/09_realtime_collector.py \\
        --os linux \\
        --source file \\
        --value /var/log/syslog \\
        --out outputs/linux/collected.csv
"""

import argparse
import subprocess
import time
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Optional, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.linux.common.constants import (
        SYSLOG_PATTERN,
        DEFAULT_YEAR,
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
    DEFAULT_YEAR = 2026
    
    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)

# Buffer settings
BUFFER_SIZE = 100  # Lines to buffer before flushing
BUFFER_TIMEOUT = 5.0  # Seconds before force flush


def parse_syslog_line(line: str, year: int) -> Optional[Dict]:
    """
    Parse a syslog-formatted line into structured fields.
    
    Args:
        line: Raw log line
        year: Year to assume for timestamps
        
    Returns:
        Dict with parsed fields or None if parsing fails
    """
    m = SYSLOG_PATTERN.match(line.strip())
    if not m:
        return None
    
    try:
        ts = datetime.strptime(f"{m.group('ts')} {year}", "%b %d %H:%M:%S %Y")
    except ValueError:
        ts = datetime.now()
    
    return {
        "ts": ts,
        "host": m.group("host"),
        "process": m.group("proc"),
        "pid": m.group("pid") or "",
        "message": m.group("msg"),
        "raw": line.rstrip("\n")
    }


def tail_file(path: Path) -> Generator[str, None, None]:
    """
    Tail a file like 'tail -F'.
    
    Args:
        path: Path to file to tail
        
    Yields:
        New lines as they appear
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)  # Go to end
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            yield line


def stream_cmd(cmd: List[str]) -> Generator[str, None, None]:
    """
    Stream output from a command.
    
    Args:
        cmd: Command and arguments
        
    Yields:
        Lines from command stdout
    """
    p = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.DEVNULL, 
        text=True
    )
    try:
        assert p.stdout is not None
        for line in p.stdout:
            yield line
    finally:
        p.terminate()


def source_generator(source: str, value: str) -> Generator[str, None, None]:
    """
    Create a generator for the specified log source.
    
    Args:
        source: Source type (file, journald, mac, docker)
        value: Source-specific value (file path or container name)
        
    Yields:
        Log lines from the source
    """
    if source == "file":
        yield from tail_file(Path(value))
    elif source == "journald":
        yield from stream_cmd(["journalctl", "-f", "-o", "short"])
    elif source == "mac":
        yield from stream_cmd(["log", "stream", "--style", "syslog"])
    elif source == "docker":
        yield from stream_cmd(["docker", "logs", "-f", value])
    else:
        raise ValueError(f"Unknown source: {source}. Use: file|journald|mac|docker")


def normalize_event(os_name: str, source: str, parsed: Dict) -> Dict:
    """Add OS and source fields to parsed event."""
    parsed["os"] = os_name
    parsed["source"] = source
    return parsed


def safe_csv_value(x) -> str:
    """Escape value for CSV output."""
    return str(x).replace("\n", " ").replace(",", " ")


def write_buffered(
    out_path: Path, 
    buffer: List[Dict],
) -> None:
    """
    Write buffered events to CSV file.
    
    Args:
        out_path: Output file path
        buffer: List of events to write
    """
    with out_path.open("a", encoding="utf-8") as f:
        for ev in buffer:
            f.write(
                f"{safe_csv_value(ev['ts'])},"
                f"{safe_csv_value(ev['host'])},"
                f"{ev['os']},"
                f"{ev['source']},"
                f"{safe_csv_value(ev['process'])},"
                f"{safe_csv_value(ev['pid'])},"
                f"{safe_csv_value(ev['message'])},"
                f"{safe_csv_value(ev['raw'])}\n"
            )


def main() -> None:
    """Main entry point for realtime collector script."""
    setup_logging()
    
    ap = argparse.ArgumentParser(
        description="Collect and stream live logs to CSV."
    )
    ap.add_argument(
        "--os", 
        required=True, 
        choices=["linux", "windows", "mac"],
        help="Operating system type"
    )
    ap.add_argument(
        "--source", 
        required=True, 
        choices=["file", "journald", "mac", "docker"],
        help="Log source type"
    )
    ap.add_argument(
        "--value", 
        required=False, 
        default="",
        help="File path or container name"
    )
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize file with header if needed
    if not out_path.exists():
        out_path.write_text(
            "ts,host,os,source,process,pid,message,raw\n", 
            encoding="utf-8"
        )

    logger.info(f"Collecting from {args.source} -> {out_path}")

    buffer: List[Dict] = []
    last_flush = time.time()

    for line in source_generator(args.source, args.value):
        ev = parse_syslog_line(line, args.year)
        if not ev:
            # Fallback: store raw only
            ev = {
                "ts": datetime.utcnow(),
                "host": "",
                "process": "",
                "pid": "",
                "message": line.strip(),
                "raw": line.strip()
            }

        ev = normalize_event(args.os, args.source, ev)
        buffer.append(ev)

        # Flush buffer when full or timeout
        if len(buffer) >= BUFFER_SIZE or (time.time() - last_flush) > BUFFER_TIMEOUT:
            write_buffered(out_path, buffer)
            buffer.clear()
            last_flush = time.time()


if __name__ == "__main__":
    main()
