"""
Shared constants and utilities for the multi-OS log-forecast pipeline.

This module centralizes default configurations and logging setup
used across Linux, Mac, and Windows pipelines.
"""

import logging

# =============================================================================
# DEFAULT CONFIGURATION (shared across all OS pipelines)
# =============================================================================

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
