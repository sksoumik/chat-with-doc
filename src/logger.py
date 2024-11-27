"""Logging configuration for the application."""
import logging
import sys
from pathlib import Path
from typing import Optional

from .debug_config import DEBUG_CONFIG


def setup_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Sets up a logger with the specified name and debug settings.

    Args:
        name (str): Name of the logger
        log_level (Optional[int]): Override default log level

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    level = log_level or DEBUG_CONFIG.get("log_level", logging.INFO)
    logger.setLevel(level)

    Path("logs").mkdir(exist_ok=True)

    fh = logging.FileHandler(f"logs/{name}.log")
    fh.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if level == logging.DEBUG:
        fh.setFormatter(detailed_formatter)
        ch.setFormatter(detailed_formatter)
    else:
        fh.setFormatter(simple_formatter)
        ch.setFormatter(simple_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
