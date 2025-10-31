"""
Logging utilities for SHAP Analytics.

Provides structured JSON logging for cloud/serverless environments.
"""

import json
import logging
import sys
import traceback

from datetime import datetime
from typing import Any

__all__ = [
    "LambdaJsonFormatter",
    "log_performance",
    "setup_structured_logger",
]


class LambdaJsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in Lambda and cloud environments.

    Outputs logs in JSON format compatible with CloudWatch, ELK, and other
    log aggregation systems.

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(LambdaJsonFormatter())
        >>> logger.addHandler(handler)
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra context from record
        if hasattr(record, "context"):
            log_data["context"] = record.context

        # Add any custom fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName", "relativeCreated",
                "thread", "threadName", "exc_info", "exc_text", "stack_info",
                "context",
            ]:
                try:
                    # Only add JSON-serializable values
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, default=str)


def setup_structured_logger(
    name: str,
    level: int = logging.INFO,
    enable_json: bool = True,
) -> logging.Logger:
    """
    Set up a logger with structured JSON output.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        enable_json: Whether to use JSON formatting (True for production).

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_structured_logger(__name__)
        >>> logger.info("Processing request", extra={"user_id": 123})
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Set formatter
    formatter: logging.Formatter
    if enable_json:
        formatter = LambdaJsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    **extra: Any,
) -> None:
    """
    Log performance metrics in structured format.

    Args:
        logger: Logger instance.
        operation: Name of the operation being measured.
        duration_ms: Duration in milliseconds.
        **extra: Additional context to log.

    Example:
        >>> import time
        >>> start = time.perf_counter()
        >>> # ... do work ...
        >>> duration_ms = (time.perf_counter() - start) * 1000
        >>> log_performance(logger, "compute_shap", duration_ms, n_samples=100)
    """
    log_data = {
        "operation": operation,
        "duration_ms": round(duration_ms, 2),
        "metric_type": "performance",
        **extra,
    }

    logger.info(
        f"Performance: {operation} completed in {duration_ms:.2f}ms",
        extra={"context": log_data},
    )


# Context manager for performance logging
class PerformanceLogger:
    """
    Context manager for automatic performance logging.

    Example:
        >>> logger = setup_structured_logger(__name__)
        >>> with PerformanceLogger(logger, "compute_shap", n_samples=100):
        ...     compute_shap_values(model, X_train, X_test)
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        **extra: Any,
    ) -> None:
        """
        Initialize performance logger.

        Args:
            logger: Logger instance.
            operation: Operation name.
            **extra: Additional context.
        """
        self.logger = logger
        self.operation = operation
        self.extra = extra
        self.start_time: float | None = None

    def __enter__(self) -> "PerformanceLogger":
        """Start timing."""
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and log performance."""
        import time
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            log_performance(self.logger, self.operation, duration_ms, **self.extra)
