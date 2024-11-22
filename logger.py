"""Generic logger class for repository with coloured terminal output and custom colour options."""

##TODO: Update docstrings and tests, consider splitting out to make easier to follow.

##TODO: Debug mode should include DEBUG and WARNING, otherwise these should not be in normal running

##TODO: Split out the stdout/stderr suppression into a separate class that can be called, too complicated here
##TODO: Actually, do I even need stdout/stderr capture?  Python should handle all types of warnings and output capture natively,
####### so I can log/capture those instead of dealing with stdout/stderr directly and having recursion issues.

from __future__ import annotations
import warnings
import logging
import sys
from pathlib import Path
from typing import ClassVar, Literal, TextIO
from contextlib import contextmanager
import warnings
import io

LogLevel = Literal["OFF", "DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION", "CRITICAL"]


class Logger:
    """A singleton logger class for consistent logging across projects."""

    _instance: ClassVar[Logger | None] = None
    logger: logging.Logger | None = None
    module_name: str = ""
    _original_stdout: TextIO
    _original_stderr: TextIO
    _suppressed_stdout: SuppressedStdoutStderr
    _suppressed_stderr: SuppressedStdoutStderr
    _suppression_active: bool = False

    @staticmethod
    def get_colours() -> dict[str, str]:
        """Return a dictionary of available colour names."""
        return {k: k for k in ColouredFormatter.COLOURS if k != 'RESET'}

    @classmethod
    def get_logger(cls) -> Logger:
        """
        Get the singleton logger instance.

        Returns
        -------
        Logger
            The configured logger instance.

        Raises
        ------
        RuntimeError
            If the logger hasn't been initialized.

        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, initial_level: LogLevel = "INFO", module_name: str | None = None, log_folder: Path = Path('log')) -> None:
        """
        Initialize the Logger.

        Parameters
        ----------
        initial_level : LogLevel, optional
            The initial logging level, by default "INFO"
        module_name : str | None, optional
            The name of the module to prepend to log messages, by default None

        Raises
        ------
        SingletonError
            If an attempt is made to create a second instance of Logger.

        """
        if Logger._instance is not None:
            e = (
                "Logger is a singleton."
                "Use get_logger() to obtain the instance."
            )
            raise SingletonError(e)
        self.module_name = module_name or ""
        self.log_folder = log_folder
        self._setup_logger(initial_level)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._suppressed_stdout = SuppressedStdoutStderr(self, "stdout")
        self._suppressed_stderr = SuppressedStdoutStderr(self, "stderr")
        self._suppression_active = False
        self._setup_warning_capture()
        

    def _setup_logger(self, level: LogLevel) -> None:
        """Set up the logger with console and file handlers."""
        logger = logging.getLogger("logger")
        logger.setLevel(getattr(logging, level))
        logger.handlers.clear()  # Clear any existing handlers

        # Console handler with coloured output
        console_formatter = ColouredFormatter(
            "%(asctime)s :: %(levelname)10s :: %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(console_formatter)
        stream_handler.setLevel(logging.DEBUG)  # Set console handler to DEBUG level
        logger.addHandler(stream_handler)

        # File handler (without colours)
        file_formatter = logging.Formatter(
            "%(asctime)s :: %(levelname)10s :: %(message)s"
        )
        self.log_folder.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.log_folder / "global.log")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Set file handler to WARNING level
        logger.addHandler(file_handler)

        # # Warning filter when external stdout or stderr is being redirected
        warning_filter = WarningFilter()
        warning_filter.logger = logger
        logger.addFilter(warning_filter)

        self.logger = logger

    def _setup_warning_capture(self):
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            warning_message = warnings.formatwarning(message, category, filename, lineno, line)
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                # In DEBUG mode, log warnings to console
                self.warning(f"Captured warning: {warning_message.strip()}")
            else:
                # Otherwise, only log to file
                self.logger.handlers[1].handle(  # File handler
                    self.logger.makeRecord(
                        "logger", logging.WARNING, filename, lineno,
                        f"Suppressed warning: {warning_message.strip()}", (), None
                    )
                )
                module_prefix = f"[{self.module_name}] " if self.module_name else ""
                print(f"{module_prefix}Warning message suppressed. Check global.log for details or set log_level as DEBUG to view in console.")

        warnings.showwarning = warning_handler

    def set_level(self, level: LogLevel) -> None:
        """
        Set the logging level.

        Parameters
        ----------
        level : LogLevel
            The desired logging level (OFF, DEBUG, INFO, WARNING, ERROR, EXCEPTION, or CRITICAL).
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

        # Toggle output suppression based on debug mode
        if level == "DEBUG":
            self._disable_output_suppression()
        else:
            self._enable_output_suppression()

    def _enable_output_suppression(self):
        if not self._suppression_active:
            self._suppression_active = True
            sys.stdout = self._suppressed_stdout
            sys.stderr = self._suppressed_stderr
            self.debug("stdout and stderr suppression enabled.")

    def _disable_output_suppression(self):
        if self._suppression_active:
            self._suppression_active = False
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self.debug("stdout and stderr suppression has been disabled.")


    @contextmanager
    def capture_output(self):
        """
        Context manager to capture stdout and stderr.
        Output is always captured, but only suppressed when not in DEBUG mode.
        
        Usage:
        with logger.capture_output():
            # Your code here
        captured_stdout = logger.get_captured_stdout()
        captured_stderr = logger.get_captured_stderr()
        """
        old_stdout, old_stderr = sys.stdout, sys.stderr
        old_showwarning = warnings.showwarning
        temp_stdout = SuppressedStdoutStderr(self, "stdout")
        temp_stderr = SuppressedStdoutStderr(self, "stderr")
        sys.stdout, sys.stderr = temp_stdout, temp_stderr
        
        def captured_warning_handler(message, category, filename, lineno, file=None, line=None):
            warning_message = warnings.formatwarning(message, category, filename, lineno, line)
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                self.warning(f"Captured warning: {warning_message.strip()}")
            else:
                self.logger.handlers[1].handle(
                    self.logger.makeRecord(
                        "logger", logging.WARNING, filename, lineno,
                        f"Suppressed warning: {warning_message.strip()}", (), None
                    )
                )
                module_prefix = f"[{self.module_name}] " if self.module_name else ""
                print(f"{module_prefix}Warning message suppressed. Check global.log for details or set log_level as DEBUG to view in console.")

        warnings.showwarning = captured_warning_handler

        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            warnings.showwarning = old_showwarning
            temp_stdout.flush()
            temp_stderr.flush()
            
            stdout_content = temp_stdout.getvalue()
            stderr_content = temp_stderr.getvalue()
            if stdout_content:
                self.debug(f"Captured stdout: {stdout_content.rstrip()}")
            if stderr_content:
                self.debug(f"Captured stderr: {stderr_content.rstrip()}")


    def get_captured_stdout(self) -> str:
        """Get the captured stdout content."""
        return self._suppressed_stdout.getvalue()

    def get_captured_stderr(self) -> str:
        """Get the captured stderr content."""
        return self._suppressed_stderr.getvalue()

    def _log(self, level: str, msg: str, colour: str | None = None) -> None:
        """Log messages with the module name prepended and optional colour."""
        if self.logger:
            module_prefix = f"[{self.module_name}] " if self.module_name else ""
            extra = {"module_name": module_prefix}
            if colour:
                extra["colour"] = ColouredFormatter.COLOURS.get(colour.upper(), colour)
            getattr(self.logger, level)(f"{module_prefix}{msg}", extra=extra)

    def debug(self, msg: str, colour: str | None = None) -> None:
        """Log a debug message."""
        self._log("debug", msg, colour)

    def info(self, msg: str, colour: str | None = None) -> None:
        """Log an info message."""
        self._log("info", msg, colour)

    def warning(self, msg: str, colour: str | None = None) -> None:
        """Log a warning message."""
        self._log("warning", msg, colour)

    def error(self, msg: str, colour: str | None = None) -> None:
        """Log an error message."""
        self._log("error", msg, colour)

    def critical(self, msg: str, colour: str | None = None) -> None:
        """Log a critical message."""
        self._log("critical", msg, colour)

    def exception(self, msg: str, colour: str | None = None) -> None:
        """Log an exception message."""
        self._log("exception", msg, colour)

    def __del__(self):
        self._disable_output_suppression()


class SingletonError(Exception):
    """Error raised when attempting to create a second instance of a singleton."""

class ColouredFormatter(logging.Formatter):
    """Logging formatter adding console colours to the output."""

    COLOURS: ClassVar[dict] = {
        'DEFAULT': '\033[0m',       # Default colour (reset)
        'BLACK': '\033[30m',        # Black
        'RED': '\033[31m',          # Red
        'GREEN': '\033[32m',        # Green
        'YELLOW': '\033[33m',       # Yellow
        'BLUE': '\033[34m',         # Blue
        'MAGENTA': '\033[35m',      # Magenta
        'CYAN': '\033[36m',         # Cyan
        'WHITE': '\033[37m',        # White
        'ORANGE': '\033[38;5;208m', # Orange (using 256-color mode)
        'SILVER': '\033[38;5;7m',   # Silver (using 256-color mode)
        'RESET': '\033[0m'          # Reset to default colour
    }

    LEVEL_COLOURS:ClassVar[dict] = {
        'DEBUG': COLOURS['BLUE'],
        'INFO': COLOURS['GREEN'],
        'WARNING': COLOURS['YELLOW'],
        'EXCEPTION': COLOURS['ORANGE'],
        'ERROR': COLOURS['RED'],
        'CRITICAL': COLOURS['MAGENTA']
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record into a string.

        Args:
        ----
            record (logging.LogRecord): The log record to format.

        Returns:
        -------
            str: The formatted log message.

        """
        levelname = record.levelname
        message = super().format(record)
        colour = getattr(record, 'colour', self.LEVEL_COLOURS.get(levelname, self.COLOURS['DEFAULT']))
        return f"{colour}{message}{self.COLOURS['RESET']}"

class SuppressedStdoutStderr:
    def __init__(self, logger: Logger, stream_type: str):
        self.logger = logger
        self.stream_type = stream_type
        self.buffer = io.StringIO()
        self.line_buffer = ""

    def write(self, message):
        self.line_buffer += message
        while '\n' in self.line_buffer:
            line, self.line_buffer = self.line_buffer.split('\n', 1)
            if line:
                self.logger.warning(f"Suppressed {self.stream_type}: {line}")
            self.buffer.write(line + '\n')

    def flush(self):
        if self.line_buffer:
            self.logger.warning(f"Suppressed {self.stream_type}: {self.line_buffer}")
            self.buffer.write(self.line_buffer)
            self.line_buffer = ""

    def getvalue(self):
        self.flush()
        return self.buffer.getvalue()

class WarningFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.WARNING:
            if self.logger.getEffectiveLevel() > logging.DEBUG:
                record.msg = 'Warning message from outside logger scope suppressed, enable DEBUG mode to view full text'
            return True
        return True