"""
Shared constants for the log-forecast pipeline.

This module centralizes regex patterns, labeling rules, and default configurations
used across multiple pipeline and realtime scripts.
"""

import re
import logging
from typing import Pattern

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# SYSLOG PARSING
# =============================================================================

SYSLOG_PATTERN: Pattern = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<proc>[A-Za-z0-9_.-]+)"
    r"(?:\[(?P<pid>\d+)\])?"
    r":\s*(?P<msg>.*)$"
)

# =============================================================================
# KEYWORD PATTERNS FOR FEATURE EXTRACTION
# =============================================================================

KW_PATTERNS: dict[str, str] = {
    "kw_error": r"\berror\b|\bfail(ed|ure)?\b|\bdenied\b|\bcritical\b|\bpanic\b",
    "kw_timeout": r"\btimeout\b|\btimed? out\b|\bNo route to host\b|\bConnection refused\b",
    "kw_auth": r"\bFailed password\b|\binvalid user\b|\bauthentication failure\b|\bpossible break-in attempt\b",
    "kw_kernel": r"\boom-killer\b|\bOut of memory\b|\bKernel panic\b",
    "kw_segfault": r"\bsegfault\b|\bcore dumped\b",
    "kw_firewall": r"\bUFW\b|\biptables\b",
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
    ("ABNORMAL_EXIT", re.compile(r"\bALERT\b.*\bexited abnormally\b", re.IGNORECASE)),
    ("SEGFAULT", re.compile(r"\bsegfault\b|\bcore dumped\b", re.IGNORECASE)),
    ("AUTH_FAIL", re.compile(r"\bFailed password\b|\binvalid user\b|\bauthentication failure\b|\bpossible break-in attempt\b", re.IGNORECASE)),
    ("SUSP_ROOT", re.compile(r"Accepted password for root\b", re.IGNORECASE)),
    ("PRIV_ESC", re.compile(r"\bsudo:.*\bUSER=root\b|sudo: pam_open_session.*failure", re.IGNORECASE)),
    ("REMOTE_SCRIPT", re.compile(r"\b(curl|wget)\b.*\bhttps?://.*(\\|\s*(bash|sh)|\\|\s*/bin/bash)\b", re.IGNORECASE)),
    ("FIREWALL", re.compile(r"\bufw\s+(disable|reset)\b|\biptables\s+(-F|DROP|REJECT)\b", re.IGNORECASE)),
    ("UFW_BLOCK", re.compile(r"\[UFW\s+BLOCK\]", re.IGNORECASE)),
    ("DISK_ERR", re.compile(r"\bEXT4-fs error\b|\bBuffer I/O error\b|\bI/O error\b|Read-only file system", re.IGNORECASE)),
    ("APPARMOR", re.compile(r'apparmor="DENIED"', re.IGNORECASE)),
    ("SUSP_CRON", re.compile(r"\bCRON\[\d+\].*CMD.*(\.cache/|/tmp/|python3\s+-c|wget|curl)", re.IGNORECASE)),
    ("OOM_KILL", re.compile(r"\boom-killer\b|\bOut of memory\b", re.IGNORECASE)),
    ("KERNEL_PANIC", re.compile(r"Kernel panic - not syncing", re.IGNORECASE)),
    ("NET_REFUSED", re.compile(r"Connection refused|No route to host|timed? out", re.IGNORECASE)),
]

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_PROCS: list[str] = [
    "CRON", "sshd", "sudo", "systemd", "systemd-timesyncd", "rsyslogd", "kernel",
    "NetworkManager", "logrotate", "ntpd", "chronyd", "cups", "dockerd", "docker",
    "smartd", "redis-server", "unattended-upgrades", "xinetd", "telnetd", "ftpd",
    "unix_chkpwd", "named", "gpm", "ufw"
]

DEFAULT_YEAR: int = 2026
DEFAULT_WINDOW: str = "60s"
DEFAULT_HORIZON_MIN: int = 15
DEFAULT_MIN_LINES: int = 5
DEFAULT_K_CONFIRM: int = 3
DEFAULT_TARGET_PRECISION: float = 0.80

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the log-forecast pipeline.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
