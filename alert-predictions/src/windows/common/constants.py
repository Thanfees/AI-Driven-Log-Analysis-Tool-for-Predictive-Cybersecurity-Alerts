"""
Windows-specific constants for the log-forecast pipeline.

This module centralizes regex patterns, labeling rules, and default configurations
for parsing and analyzing Windows CBS/CSI event logs.
"""

import re
import logging
from typing import Pattern

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# WINDOWS CBS/CSI LOG PARSING
# =============================================================================

# Windows CBS/CSI log format:
# 2016-09-28 04:30:30, Info                  CBS    Starting TrustedInstaller...
# Fields: timestamp, level, component, message
WINDOWS_LOG_PATTERN: Pattern = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s*"
    r"(?P<level>\w+)\s+"
    r"(?P<proc>\w+)\s+"
    r"(?P<msg>.*)$"
)

# CSI entries with sequence IDs (e.g., "00000001@2016/9/27:20:30:31.455")
CSI_SEQ_PATTERN: Pattern = re.compile(
    r"^(?P<seq>\d{8})(?:@\d{4}/\d{1,2}/\d{1,2}:\d{2}:\d{2}:\d{2}\.\d+)?\s+(?P<msg>.*)$"
)

# =============================================================================
# KEYWORD PATTERNS FOR FEATURE EXTRACTION
# =============================================================================

KW_PATTERNS: dict[str, str] = {
    "kw_error": r"\berror\b|\bfail(ed|ure)?\b|\bE_FAIL\b|\bHRESULT\b.*0x8",
    "kw_warning": r"\bwarning\b|\bcaution\b",
    "kw_update": r"\bWindowsUpdateAgent\b|\bWindows Update\b|\bwusa\b",
    "kw_service": r"\bTrustedInstaller\b|\bservice\b.*\b(start|stop|fail)\b",
    "kw_manifest": r"CBS_E_MANIFEST|INVALID_ITEM|\bmanifest\b.*\berror\b",
    "kw_corruption": r"\bcorrupt(ed|ion)?\b|\brepair\b|\bSFC\b|\bDISM\b",
    "kw_security": r"\baudit\b|\bpolicy\b.*\bfail\b|\baccess denied\b|\bsecurity\b.*\bviolation\b",
    "kw_store": r"\bstore\b.*\b(error|corrupt|fail)\b|\bCSI Store\b",
    "kw_reboot": r"\breboot\b|\brestart\b.*\bpending\b|\bshutdown\b",
}

# Pre-compiled keyword patterns for performance
KW_PATTERNS_COMPILED: dict[str, Pattern] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in KW_PATTERNS.items()
}

# =============================================================================
# ANOMALY LABELING RULES
# =============================================================================

LABELING_RULES: list[tuple[str, Pattern]] = [
    ("CBS_FAIL", re.compile(
        r"\bCBS\b.*\bfail|\bCBS\b.*\berror\b|\bfailed to\b.*\bCBS\b", re.IGNORECASE)),
    ("MANIFEST_ERROR", re.compile(
        r"CBS_E_MANIFEST|\bmanifest\b.*\b(invalid|error|corrupt)\b", re.IGNORECASE)),
    ("UPDATE_FAIL", re.compile(
        r"\bWindows\s*Update\b.*\b(fail|error)\b|\bupdate\b.*\bfail(ed|ure)\b"
        r"|\bHRESULT\s*=\s*0x80(070002|004005|073712|240016)\b", re.IGNORECASE)),
    ("CORRUPTION", re.compile(
        r"\bcorrupt(ed|ion)?\b.*\b(store|component|package)\b"
        r"|\bSFC\b.*\b(found|repair|fail)\b"
        r"|\bDISM\b.*\b(error|fail)\b", re.IGNORECASE)),
    ("STORE_ERROR", re.compile(
        r"\bstore\b.*\b(corrupt|error|fail|damage)\b"
        r"|\bcomponent store\b.*\b(error|repair)\b", re.IGNORECASE)),
    ("SERVICE_FAIL", re.compile(
        r"\bservice\b.*\b(fail|crash|unexpected|stop)\b"
        r"|\bTrustedInstaller\b.*\b(fail|error|crash)\b", re.IGNORECASE)),
    ("SECURITY_VIOLATION", re.compile(
        r"\baccess denied\b|\bsecurity\b.*\bviolation\b"
        r"|\bpolicy\b.*\bfail\b|\baudit\b.*\bfail\b", re.IGNORECASE)),
    ("HRESULT_ERROR", re.compile(
        r"\bHRESULT\s*=\s*0x8[0-9a-fA-F]{7}\b", re.IGNORECASE)),
    ("INSTALL_FAIL", re.compile(
        r"\binstall(ation)?\b.*\bfail(ed|ure)?\b"
        r"|\bpackage\b.*\bfail\b", re.IGNORECASE)),
    ("PENDING_REBOOT", re.compile(
        r"\breboot\b.*\bpending\b|\bpending\b.*\breboot\b"
        r"|\brestart\b.*\brequired\b", re.IGNORECASE)),
]

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_PROCS: list[str] = [
    "CBS", "CSI", "WindowsUpdateAgent", "TrustedInstaller",
    "DISM", "SFC", "WER", "WUSA", "msiexec", "DPX",
    "CMIV2", "Perf", "WCP",
]

# Import shared defaults
try:
    from src.common.constants import (
        DEFAULT_YEAR,
        DEFAULT_WINDOW,
        DEFAULT_HORIZON_MIN,
        DEFAULT_MIN_LINES,
        DEFAULT_K_CONFIRM,
        DEFAULT_TARGET_PRECISION,
        LOG_FORMAT,
        LOG_DATE_FORMAT,
        setup_logging,
        get_logger,
    )
except ImportError:
    DEFAULT_YEAR = 2026
    DEFAULT_WINDOW = "60s"
    DEFAULT_HORIZON_MIN = 15
    DEFAULT_MIN_LINES = 5
    DEFAULT_K_CONFIRM = 3
    DEFAULT_TARGET_PRECISION = 0.80
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def setup_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)
