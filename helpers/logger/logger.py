"""Generic logger class for repository."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Literal

LogLevel = Literal["OFF", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class SingletonError(Exception):
    """Error raised when attempting to create a second instance of a singleton."""

class Logger:
    """A singleton logger class for consistent logging across projects."""

    _instance: ClassVar[WestworldLogger | None] = None
    logger: logging.Logger | None = None

    @classmethod
    def get_logger(cls) -> WestworldLogger:
        """
        Get the singleton logger instance.

        Returns
        -------
        WestworldLogger
            The configured logger instance.

        Raises
        ------
        RuntimeError
            If the logger hasn't been initialized.

        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, initial_level: LogLevel = "INFO") -> None:
        """
        Initialize the WestworldLogger.

        Parameters
        ----------
        initial_level : LogLevel, optional
            The initial logging level, by default "INFO"

        Raises
        ------
        SingletonError
            If an attempt is made to create a second instance of WestworldLogger.

        """
        if WestworldLogger._instance is not None:
            e = (
                "WestworldLogger is a singleton."
                "Use get_logger() to obtain the instance."
            )
            raise SingletonError(e)
        self._setup_logger(initial_level)

    def _setup_logger(self, level: LogLevel) -> None:
        """Set up the logger with console and file handlers."""
        logger = logging.getLogger("westworld")
        logger.setLevel(getattr(logging, level))
        logger.handlers.clear()  # Clear any existing handlers

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "westworld.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def set_level(self, level: LogLevel) -> None:
        """
        Set the logging level.

        Parameters
        ----------
        level : LogLevel
            The desired logging level (OFF, DEBUG, INFO, WARNING, ERROR, or CRITICAL).
            If set to OFF, logging will be disabled.

        """
        if self.logger is None:
            e = (
                "Logger has not been properly initialized."
                "Use get_logger() to obtain the instance."
            )
            raise RuntimeError(e)

        if level == "OFF":
            self.logger.setLevel(logging.CRITICAL + 1)  # Set to an unreachable level
            self.logger.disabled = True
        else:
            self.logger.setLevel(getattr(logging, level))
            self.logger.disabled = False

    def debug(self, msg: str) -> None:
        """Log a debug message."""
        if self.logger:
            self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log an info message."""
        if self.logger:
            self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        if self.logger:
            self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        if self.logger:
            self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message."""
        if self.logger:
            self.logger.critical(msg)

    def exception(self, msg: str) -> None:
        """Log an exception message."""
        if self.logger:
            self.logger.exception(msg)
