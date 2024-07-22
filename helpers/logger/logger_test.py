"""Unit tests for the logger helper module."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import pytest

from ogger import LogLevel, Logger


class ParametrizeArgs(TypedDict):
    """
    A TypedDict representing the arguments for pytest.mark.parametrize decorator.

    This class defines the structure and types of arguments expected by the
    pytest.mark.parametrize decorator for compatability with the mypy static
    type checker.

    Attributes
    ----------
        argnames (str): A string containing comma-separated names of the arguments
            to be parametrized. These names should correspond to the function
            parameters that will receive the parametrized values.

        argvalues (list[str]): A list of strings representing the values to be
            passed to the parametrized arguments. Each item in this list
            corresponds to a test case.

    """

    argnames: str
    argvalues: list[str]


parametrize: Callable[..., Callable[[Callable[..., None]], Callable[..., None]]] = (
    pytest.mark.parametrize
)


def test_singleton() -> None:
    """
    Test the singleton behavior of Logger.

    This test ensures that multiple calls to Logger.get_logger()
    return the same instance.

    Returns
    -------
    None

    """
    logger1 = Logger.get_logger()
    logger2 = Logger.get_logger()
    assert logger1 is logger2


def test_initial_level() -> None:
    """
    Test the initial log level of Logger.

    This test verifies that the initial log level of the logger is set to INFO.

    Returns
    -------
    None

    """
    logger = Logger.get_logger()
    assert logger.logger is not None
    assert logger.logger.level == logging.INFO


def test_set_level() -> None:
    """
    Test the set_level method of Logger.

    This test checks if the log level can be properly set to different values,
    including the special 'OFF' level.

    Returns
    -------
    None

    """
    logger = Logger.get_logger()
    assert logger.logger is not None

    logger.set_level("DEBUG")
    assert logger.logger.level == logging.DEBUG

    logger.set_level("ERROR")
    assert logger.logger.level == logging.ERROR

    logger.set_level("OFF")
    assert logger.logger.level == logging.CRITICAL + 1
    assert logger.logger.disabled is True


def test_logger_setup() -> None:
    """
    Test the setup of Logger.

    This test ensures that the logger is properly set up with both a StreamHandler
    and a FileHandler.

    Returns
    -------
    None

    """
    logger = Logger.get_logger()
    test_value = 2
    assert logger.logger is not None
    assert len(logger.logger.handlers) == test_value
    assert isinstance(logger.logger.handlers[0], logging.StreamHandler)
    assert isinstance(logger.logger.handlers[1], logging.FileHandler)


@parametrize(
    **ParametrizeArgs(
        argnames="level", argvalues=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
)
def test_log_levels(
    level: LogLevel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Test logging at different levels.

    This test verifies that messages are properly logged at different log levels.

    Parameters
    ----------
    level : LogLevel
        The log level to test.
    caplog : pytest.LogCaptureFixture
        Pytest fixture for capturing log output.

    Returns
    -------
    None

    """
    logger = Logger.get_logger()
    logger.set_level(level)

    log_message = f"This is a {level} message"
    getattr(logger, level.lower())(log_message)

    assert log_message in caplog.text


def test_file_logging(tmp_path: Path) -> None:
    """
    Test file logging functionality.

    This test checks if log messages are properly written to a file at various
    log levels.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary directory path.

    Returns
    -------
    None

    """
    log_file = tmp_path / "test_log.txt"
    Logger._instance = None  # Reset the singleton # noqa: SLF001
    logger = Logger(initial_level="DEBUG")

    # Replace the existing file handler with one that writes to our test file
    assert logger.logger is not None
    for handler in logger.logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file)
    logger.logger.addHandler(file_handler)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    logger.exception("Exception message")

    with Path.open(log_file) as f:
        log_contents = f.read()

    assert "Debug message" in log_contents
    assert "Info message" in log_contents
    assert "Warning message" in log_contents
    assert "Error message" in log_contents
    assert "Critical message" in log_contents
    assert "Exception message" in log_contents
