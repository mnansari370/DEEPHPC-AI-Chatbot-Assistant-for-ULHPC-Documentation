"""Logging utilities for DEEPHPC."""
import sys
from loguru import logger
from pathlib import Path


def get_logger(name: str, log_file: str = None, level: str = "INFO"):
    """
    Configure and return a loguru logger.

    Args:
        name: Logger name (used in log format).
        log_file: Optional path to write logs to file.
        level: Log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    logger.remove()  # Remove default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>{name}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, format=fmt, level=level, rotation="10 MB")

    return logger
