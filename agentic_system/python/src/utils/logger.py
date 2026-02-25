"""
Structured logging with sensitive data filtering.

Security:
    - Filters API keys, passwords, tokens from logs
    - JSON format for production (easy to parse)
    - File rotation to prevent disk overflow
"""
import logging
import sys
import json
import re
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


class SensitiveDataFilter(logging.Filter):
    """
    Filter to remove sensitive data from logs.

    Patterns matched:
        - api_key, api-key
        - password
        - token
        - secret
        - authorization headers

    Example:
        >>> logger.info("API key: sk-1234")
        # Logs: "API key: [REDACTED]"
    """

    PATTERNS = [
        (re.compile(r'(api[_-]?key["\s:=]+)[\w\-]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(password["\s:=]+)\S+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(token["\s:=]+)[\w\-]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(secret["\s:=]+)\S+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(Bearer\s+)[\w\-\.]+', re.IGNORECASE), r'\1[REDACTED]'),
        (re.compile(r'(sk-|pk-|ak-)[\w\-]+', re.IGNORECASE), r'[REDACTED]'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize log message by removing sensitive data."""
        if record.msg:
            msg = str(record.msg)
            for pattern, replacement in self.PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Produces logs like:
        {"timestamp": "2024-01-01T12:00:00", "level": "INFO", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up logger with security and best practices.

    Features:
        - Sensitive data filtering (API keys, passwords)
        - Optional JSON format for production
        - File rotation to prevent disk overflow
        - Console and file handlers

    Args:
        name: Logger name (use __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path for file logging
        json_format: Use JSON format (recommended for production)
        max_bytes: Max file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Trade executed", extra={"symbol": "EURUSD"})
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Add sensitive data filter
    logger.addFilter(SensitiveDataFilter())

    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger for the application
_root_logger: Optional[logging.Logger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger, creating the root logger if needed.

    Args:
        name: Logger name (default: 'pipflow')

    Returns:
        Logger instance
    """
    global _root_logger

    if name is None:
        name = "pipflow"

    if _root_logger is None:
        from ..config.base import get_config

        config = get_config()
        _root_logger = setup_logger(
            "pipflow",
            level=getattr(logging, config.log_level.upper()),
            json_format=config.log_format == "json",
        )

    if name == "pipflow":
        return _root_logger

    return logging.getLogger(name)
