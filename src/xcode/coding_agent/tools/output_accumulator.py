from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from typing import Any

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail

logger = logging.getLogger("xcode.coding_agent.tools.output_accumulator")


@dataclass
class SnapshotResult:
    content: str
    total_lines: int
    total_bytes: int
    truncated: bool
    full_output_path: str | None


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
        self._current_line_bytes = 0

    @property
    def full_path(self) -> str | None:
        return self._full_path

    @property
    def total_lines(self) -> int:
        return self._total_lines

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def get_last_line_bytes(self) -> int:
        return self._current_line_bytes

    def append(self, chunk: bytes) -> None:
        self._chunks.append(chunk)
        self._total_bytes += len(chunk)
        newlines = chunk.count(b"\n")
        self._total_lines += newlines
        if newlines == 0:
            self._current_line_bytes += len(chunk)
        else:
            last_newline = chunk.rfind(b"\n")
            self._current_line_bytes = len(chunk) - last_newline - 1
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

    def snapshot(self, persist_if_truncated: bool = False) -> str:
        result = self.snapshot_detailed(persist_if_truncated=persist_if_truncated)
        return result.content

    def snapshot_detailed(self, persist_if_truncated: bool = False) -> SnapshotResult:
        if self._chunks:
            text = b"".join(self._chunks).decode("utf-8", errors="replace")
        else:
            text = ""

        has_content = bool(text) or self._truncated

        if not has_content:
            return SnapshotResult(
                content="(no output)",
                total_lines=0,
                total_bytes=0,
                truncated=False,
                full_output_path=self._full_path,
            )

        if persist_if_truncated and self._truncated and self._file is None:
            self._persist_full()

        tr = truncate_tail(text, max_lines=self._max_lines, max_bytes=self._max_bytes)

        full_path_hint = self._full_path and (
            self._truncated or self._total_lines > tr.output_lines
        )
        if full_path_hint:
            footer = (
                f"\n[Showing {tr.output_lines} of {self._total_lines} lines"
                f" ({self._max_bytes // 1024}KB limit)."
                f" Full output: {self._full_path}]"
            )
            return SnapshotResult(
                content=tr.content + footer,
                total_lines=self._total_lines,
                total_bytes=self._total_bytes,
                truncated=True,
                full_output_path=self._full_path,
            )

        return SnapshotResult(
            content=tr.content,
            total_lines=self._total_lines,
            total_bytes=self._total_bytes,
            truncated=self._truncated,
            full_output_path=self._full_path,
        )

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                logger.debug(
                    "failed to close temp file %s", self._full_path, exc_info=True
                )
