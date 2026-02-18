"""
macOS-specific constants for the log-forecast pipeline.

This module centralizes regex patterns, labeling rules, and default configurations
for parsing and analyzing macOS syslog-style system logs.
"""

import re
import logging
from typing import Pattern

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# MAC SYSLOG PARSING
# =============================================================================

# Mac logs use the same syslog format as Linux, but process names can contain
# dots (e.g., com.apple.xpc.launchd) and brackets with PID
MAC_SYSLOG_PATTERN: Pattern = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<proc>[A-Za-z0-9_.:-]+)"
    r"(?:\[(?P<pid>\d+)\])?"
    r":\s*(?P<msg>.*)$"
)

# =============================================================================
# KEYWORD PATTERNS FOR FEATURE EXTRACTION
# =============================================================================

KW_PATTERNS: dict[str, str] = {
    "kw_error": r"\berror\b|\bfail(ed|ure)?\b|\bdenied\b|\bcritical\b",
    "kw_sandbox": r"\bSandbox\b.*\bdeny\b|\bSandbox violation\b",
    "kw_kernel": r"\bkernel panic\b|\bKernel panic\b|\bpanic\b.*\bcpu\b",
    "kw_power": r"\bSleep\b|\bWake\b|\bpower\b.*\bstate\b|\bthermal pressure\b|\bpowerChange\b",
    "kw_wifi": r"\bLink Down\b|\bAirPort\b.*\b(error|fail)\b|\bchannel changed\b|\bdisassociated\b",
    "kw_disk": r"\bdisk\d+s\d+\b.*\berror\b|\bI/O error\b|\bfilevault\b|\bRead-only file system\b",
    "kw_memory": r"\bmemory pressure\b|\bjetsam\b|\bVM Compressor\b|\bMemory pressure state:\s*[1-9]\b",
    "kw_bluetooth": r"\bBluetooth\b.*\b(error|fail|disconnect)\b|Bluetooth --",
    "kw_usb": r"\bUSB\b.*\b(error|fail|disconnect)\b|\bIOUSB\b",
    "kw_thunderbolt": r"\bThunderbolt\b.*\b(error|fail)\b|\bIOThunderbolt\b",
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
    ("SANDBOX_DENY", re.compile(
        r"\bSandbox\b.*\bdeny\b|\bSandbox violation\b", re.IGNORECASE)),
    ("KERNEL_PANIC", re.compile(
        r"\bKernel panic\b|\bpanic\b.*\bnot syncing\b|\bpanic\b.*\bcpu\b", re.IGNORECASE)),
    ("DISK_ERROR", re.compile(
        r"\bdisk\d+s\d+\b.*\berror\b|\bI/O error\b|\bBuffer I/O error\b"
        r"|\bRead-only file system\b|\bEXT4-fs error\b", re.IGNORECASE)),
    ("WIFI_FAILURE", re.compile(
        r"\bLink Down\b.*\bReason\b|\bAirPort\b.*\bfail\b"
        r"|\bwl0\b.*\berror\b|\bNo route to host\b", re.IGNORECASE)),
    ("MEMORY_PRESSURE", re.compile(
        r"\bmemory pressure\b.*\bstate:\s*[2-9]\b|\bjetsam\b.*\bkill\b"
        r"|\bVM Compressor\b.*\berror\b", re.IGNORECASE)),
    ("THERMAL_EVENT", re.compile(
        r"\bthermal pressure\b.*\bstate:\s*[2-9]\b"
        r"|\bthermal\b.*\bcritical\b", re.IGNORECASE)),
    ("APP_CRASH", re.compile(
        r"\bcrash(ed)?\b|\bSegmentation fault\b|\bcore dumped\b"
        r"|\bsignal\s+11\b|\bASSERTION FAILED\b", re.IGNORECASE)),
    ("SECURITY_VIOLATION", re.compile(
        r"\baccess denied\b|\bpermission denied\b"
        r"|\bauthentication\b.*\bfail\b|\bFailed password\b", re.IGNORECASE)),
    ("NETWORK_ERROR", re.compile(
        r"\bConnection refused\b|\btimed? out\b|\bnetwork\b.*\bunreachable\b"
        r"|\bDNS\b.*\bfail\b", re.IGNORECASE)),
    ("USB_ERROR", re.compile(
        r"\bUSB\b.*\b(error|fail|disconnect|reset)\b"
        r"|\bIOUSB\b.*\berror\b", re.IGNORECASE)),
    ("MDNS_ERROR", re.compile(
        r"\bmDNSResponder\b.*\b(error|fail)\b"
        r"|\bmDNS\b.*\bFrequent transitions\b", re.IGNORECASE)),
    ("PM_RESPONSE_SLOW", re.compile(
        r"\bPM response took\b.*\b\d{4,}\b.*\bms\b", re.IGNORECASE)),
]

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_PROCS: list[str] = [
    "kernel", "sandboxd", "symptomsd", "mDNSResponder", "WindowServer",
    "Safari", "AirPort", "bluetoothd", "apsd", "com.apple.CDScheduler",
    "com.apple.xpc.launchd", "com.apple.AddressBook.InternetAccountsBridge",
    "powerd", "loginwindow", "UserEventAgent", "configd",
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
