from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


class _LogCapture:
    """Captured log records with an ``output`` attribute for assertion."""

    def __init__(self, records: list[logging.LogRecord]) -> None:
        self.records = records
        self.output = [r.getMessage() for r in records]


@contextmanager
def assert_logs(
    logger_name: str | None = None,
    level: str = "WARNING",
) -> Iterator[_LogCapture]:
    """Assert that at least one log message is emitted at or above *level*."""
    logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()

    def _emit(record: logging.LogRecord) -> None:
        records.append(record)

    handler.emit = _emit
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(getattr(logging, level.upper()))
    try:
        yield _LogCapture(records)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    if not records:
        msg = f"no log messages of level {level} or above emitted by {logger_name!r}"
        raise AssertionError(msg)


@contextmanager
def assert_no_logs(
    logger_name: str | None = None,
    level: str = "WARNING",
) -> Iterator[None]:
    """Assert that no log message is emitted at or above *level*."""
    logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()

    def _emit(record: logging.LogRecord) -> None:
        records.append(record)

    handler.emit = _emit
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(getattr(logging, level.upper()))
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    if records:
        messages = [r.getMessage() for r in records]
        raise AssertionError(
            f"Unexpected log messages at level {level} from {logger_name!r}: {messages}"
        )
