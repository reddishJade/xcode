from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

_file_locks: dict[Path, threading.Lock] = {}
_registry_lock = threading.Lock()


def _resolve_key(file_path: Path) -> Path:
    try:
        return file_path.resolve()
    except (OSError, ValueError):
        return file_path.absolute()


def with_file_mutation[T](file_path: Path, fn: Callable[[], T]) -> T:
    key = _resolve_key(file_path)
    with _registry_lock:
        mutex = _file_locks.get(key)
        if mutex is None:
            mutex = threading.Lock()
            _file_locks[key] = mutex
    with mutex:
        return fn()
