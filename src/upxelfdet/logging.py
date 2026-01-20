"""Enhanced logging configuration with file output support.

This module extends maldet's configure_logging with file output capabilities.
It inherits maldet's configuration and reconfigures structlog to use stdlib
LoggerFactory instead of PrintLoggerFactory to enable file logging.

Typical usage:
    from upxelfdet.logging import configure_logging

    configure_logging(
        level="INFO",
        format="console",
        log_dir="./output/logs"
    )
"""

import logging
import sys
from pathlib import Path
from typing import Literal

import structlog
from maldet.logging import configure_logging as maldet_configure_logging
from structlog.types import Processor


def configure_logging(
    level: str = "INFO",
    format: Literal["console", "json"] = "console",
    log_dir: Path | str | None = None,
) -> None:
    """Configure logging with optional file output.

    This function extends maldet's configure_logging by:
    1. Calling maldet's implementation to set up processors
    2. Reconfiguring structlog to use LoggerFactory (instead of PrintLoggerFactory)
    3. Adding file handlers if log_dir is specified

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Output format - "console" for human-readable, "json" for
            machine-readable structured logs.
        log_dir: Optional directory for log files. If provided, logs will be
            written to both console and files in this directory.
    """
    # First, let maldet configure the base logging setup
    maldet_configure_logging(level=level, format=format)

    # Get the log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure stdlib logging to work with file handlers
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
        force=True,
    )

    # Setup file handlers if log directory is specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger()

        # Create human-readable log file
        log_file = log_path / "detector.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        file_handler.setLevel(log_level)

        # Use appropriate formatter based on format setting
        if format == "json":
            file_formatter = logging.Formatter("%(message)s")
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Always create a JSON log file for machine processing
        json_log_file = log_path / "detector.json.log"
        json_handler = logging.FileHandler(json_log_file, encoding="utf-8", mode="a")
        json_handler.setLevel(log_level)
        json_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(json_handler)

    # Reconfigure structlog to use stdlib LoggerFactory
    # This inherits the processors from maldet but changes the logger factory
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if format == "json":
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ]

    # Reconfigure structlog with LoggerFactory to enable file logging
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),  # Key change from maldet
        cache_logger_on_first_use=False,  # Allow reconfiguration
    )
