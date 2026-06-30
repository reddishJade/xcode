"""输出滚动缓冲区与截断。

OpenCode 风格的滚动缓冲区设计：
- keep = maxBytes * 2，保留截断余量
- 溢出后才写文件（惰性写入），小输出纯内存
- 始终保持最近 N 字节的预览文本供元数据推送
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO, cast

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail

logger = logging.getLogger("xcode.coding_agent.tools.output_accumulator")

# 元数据预览的最大长度（30KB）
MAX_PREVIEW_BYTES: int = 30 * 1024


@dataclass
class SnapshotResult:
    content: str
    total_lines: int
    total_bytes: int
    truncated: bool
    full_output_path: str | None


class _ChunkEntry:
    """滚动缓冲区中的单个条目。"""

    __slots__ = ("text", "size")

    def __init__(self, text: str, size: int) -> None:
        self.text = text
        self.size = size


class PersistedOutputFile:
    def __init__(self, temp_prefix: str) -> None:
        self._file = cast(
            BinaryIO,
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".log",
                prefix=temp_prefix,
            ),
        )
        self.path = str(self._file.name)

    def write_existing(self, chunks: list[bytes]) -> None:
        for chunk in chunks:
            self._file.write(chunk)
        self._file.flush()

    def append(self, chunk: bytes) -> None:
        self._file.write(chunk)

    def close(self) -> None:
        self._file.flush()
        self._file.close()


class OutputAccumulator:
    """滚动缓冲区输出累加器。

    设计遵循 OpenCode：
    - keep = max_bytes * 2：滚动窗口保留 2x 上限，给截断留余量
    - 溢出后惰性写文件（只在超过 max_bytes 时落盘）
    - last/preview 保持最近 30KB 供元数据推送
    """

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_lines: int = DEFAULT_MAX_LINES,
        temp_prefix: str = "xcode-bash-",
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_lines = max_lines
        self._temp_prefix = temp_prefix
        self._on_progress = on_progress

        # 滚动窗口（保留 max_bytes * 2 字节）
        self._keep = max_bytes * 2
        self._list: list[_ChunkEntry] = []
        self._used = 0  # 当前窗口总字节数

        # 完整累计统计（不因滚动而丢失）
        self._total_lines = 0
        self._total_bytes = 0

        # 预览字符串（最近 ~MAX_PREVIEW_BYTES）
        self._preview = ""

        # 惰性文件 spill
        self._truncated = False
        self._full_path: str | None = None
        self._persisted_output: PersistedOutputFile | None = None
        self._finished = False

    # ── 属性 ──

    @property
    def full_path(self) -> str | None:
        return self._full_path

    @property
    def total_lines(self) -> int:
        return self._total_lines

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def preview(self) -> str:
        """最近 MAX_PREVIEW_BYTES 的预览文本，用于元数据推送。"""
        return self._preview

    # ── 追加数据 ──

    def append(self, chunk: bytes) -> None:
        if self._finished:
            raise RuntimeError("cannot append after close")
        text = chunk.decode("utf-8", errors="replace")
        size = len(chunk)

        self._total_bytes += size
        self._total_lines += text.count("\n")

        # 追加到滚动窗口
        self._list.append(_ChunkEntry(text, size))
        self._used += size

        # 滚动：超出 keep 时从头部移除
        while self._used > self._keep and len(self._list) > 1:
            entry = self._list.pop(0)
            self._used -= entry.size

        # 更新预览（保持最近 MAX_PREVIEW_BYTES）
        self._preview = self._preview + text
        if len(self._preview) > MAX_PREVIEW_BYTES:
            self._preview = self._preview[-MAX_PREVIEW_BYTES:]

        # 如果已 spill 到文件，同步写入
        if self._persisted_output is not None:
            self._persisted_output.append(chunk)

        # 首次超限 → 惰性 spill
        if self._persisted_output is None and (
            self._total_bytes > self._max_bytes
            or self._total_lines > self._max_lines
        ):
            self._spill()

        # 进度回调
        if self._on_progress:
            self._on_progress(text)

    def _spill(self) -> None:
        """将完整内容写入文件，内存仅保留尾部窗口。"""
        if self._persisted_output is not None:
            return
        persisted = PersistedOutputFile(self._temp_prefix)
        # 写入所有已收集的数据
        all_bytes = b"".join(e.text.encode("utf-8", errors="replace") for e in self._list)
        # 但 PersistedOutputFile.write_existing 接受 list[bytes]
        # 我们用 append 逐个写
        for entry in self._list:
            persisted.append(entry.text.encode("utf-8", errors="replace"))
        self._persisted_output = persisted
        self._full_path = persisted.path
        self._truncated = True

    # ── 快照 ──

    def snapshot(self, persist_if_truncated: bool = False) -> str:
        return self.snapshot_detailed(persist_if_truncated=persist_if_truncated).content

    def snapshot_detailed(self, persist_if_truncated: bool = False) -> SnapshotResult:
        # 拼接滚动窗口内容
        text = "".join(e.text for e in self._list) if self._list else ""

        has_content = bool(text) or self._truncated

        if not has_content:
            return SnapshotResult(
                content="(no output)",
                total_lines=0,
                total_bytes=0,
                truncated=False,
                full_output_path=self._full_path,
            )

        if persist_if_truncated and self._truncated and self._persisted_output is None:
            self._spill()

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

    # ── 清理 ──

    def close(self) -> None:
        self._finished = True
        if self._persisted_output is not None:
            try:
                self._persisted_output.close()
            except Exception:
                logger.debug(
                    "failed to close temp file %s", self._full_path, exc_info=True
                )
