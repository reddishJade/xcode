from __future__ import annotations

import logging
import tempfile
from typing import Any

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail

logger = logging.getLogger("xcode.coding_agent.tools.output_accumulator")


class OutputAccumulator:
    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_lines: int = DEFAULT_MAX_LINES,
        temp_prefix: str = "xcode-bash-",
    ):
        self._max_bytes = max_bytes
        self._max_lines = max_lines
        self._temp_prefix = temp_prefix
        self._chunks: list[bytes] = []
        self._total_lines = 0
        self._total_bytes = 0
        self._truncated = False
        self._full_path: str | None = None
        self._file: Any = None

    @property
    def full_path(self) -> str | None:
        return self._full_path

    @property
    def total_lines(self) -> int:
        return self._total_lines

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def append(self, chunk: bytes) -> None:
        self._chunks.append(chunk)
        self._total_bytes += len(chunk)
        self._total_lines += chunk.count(b"\n")
        if self._file is not None:
            self._file.write(chunk)
            self._trim_tail()
            return
        if self._total_lines > self._max_lines or self._total_bytes > self._max_bytes:
            self._persist_full()

    def _persist_full(self) -> None:
        if self._file is None:
            self._file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".log", prefix=self._temp_prefix
            )
            self._full_path = self._file.name
            for chunk in self._chunks:
                self._file.write(chunk)
            self._file.flush()
        self._truncated = True
        self._trim_tail()

    def _trim_tail(self) -> None:
        text = b"".join(self._chunks)
        if len(text) > self._max_bytes:
            text = text[-self._max_bytes :]
            first_newline = text.find(b"\n")
            if first_newline > 0:
                text = text[first_newline + 1 :]
        lines = text.splitlines(keepends=True)
        if len(lines) > self._max_lines:
            text = b"".join(lines[-self._max_lines :])
        self._chunks = [text] if text else []

    def snapshot(self) -> str:
        if self._chunks:
            text = b"".join(self._chunks).decode("utf-8", errors="replace")
        else:
            text = ""

        if not text and not self._truncated:
            return "(no output)"

        result = truncate_tail(
            text, max_lines=self._max_lines, max_bytes=self._max_bytes
        )
        output_lines = result.count("\n") + 1 if result else 0

        if self._full_path and (self._truncated or self._total_lines > output_lines):
            footer = (
                f"\n[Showing {output_lines} of {self._total_lines} lines"
                f" ({self._max_bytes // 1024}KB limit)."
                f" Full output: {self._full_path}]"
            )
            return result + footer

        return result

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                logger.debug(
                    "failed to close temp file %s", self._full_path, exc_info=True
                )
