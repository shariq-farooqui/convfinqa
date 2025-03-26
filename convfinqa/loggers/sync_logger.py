import json
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Any

import pendulum

from .base_logger import BaseLogger


class CustomJsonFormatter(logging.Formatter):
    """Custom JSON formatter to include extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        log_data = {
            "logged_at": pendulum.now("Europe/London").isoformat(),
            "level": record.levelname,
            "message": record.msg,
            "function_name": record.funcName,
            "file_path": record.pathname,
            "line_number": record.lineno,
        }
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        return json.dumps(log_data)


class SyncJsonLogger(BaseLogger):
    """A singleton class for creating synchronous JSON logger."""

    def __init__(
        self,
        name: str,
        level: int | None = None,
        log_file_path: str | None = None,
    ) -> None:
        super().__init__(name=name, level=level, log_file_path=log_file_path)

        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(self._level)

        # Clear any existing handlers to avoid duplicates
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)

        json_formatter = CustomJsonFormatter()

        # File handler - use DEBUG level to ensure all logs are written to file
        # regardless of the overall logger level
        file_handler = TimedRotatingFileHandler(
            filename=self._filename,
            when=self._log_interval_unit,
            interval=self._log_interval,
            backupCount=self._log_file_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        self._logger.addHandler(file_handler)

        self._logger.propagate = False

    def get_logger(self) -> logging.Logger:
        """Return the configured logger."""
        return self._logger


def get_sync_singleton_logger(
    name: str, level: int | None = None, log_file_path: str | None = None
) -> logging.Logger:
    """Factory function to get a logger."""
    return SyncJsonLogger(name, level, log_file_path).get_logger()


class LoggerAdapter(logging.LoggerAdapter):
    """Custom LoggerAdapter to handle extra fields."""

    def process(
        self, msg: str, kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Process the log message and keyword arguments."""
        extra = kwargs.get("extra", {})
        kwargs["extra"] = {"extra": extra}
        return msg, kwargs


def get_singleton_logger(
    name: str, level: int | None = None, log_file_path: str | None = None
) -> LoggerAdapter:
    """Factory function to get a logger with adapter."""
    logger = get_sync_singleton_logger(name, level, log_file_path)
    return LoggerAdapter(logger, {})
