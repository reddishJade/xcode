from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompactionPreparation:
    """压缩准备结果。"""

    messages: list[Any]
    read_files: set[str] = field(default_factory=set)
    modified_files: set[str] = field(default_factory=set)


@dataclass
class CompactionResult:
    """压缩结果。"""

    summary: str
    messages: list[Any]
    tokens_saved: int = 0


def prepare_compaction(
    messages: list[Any],
    read_files: set[str] | None = None,
    modified_files: set[str] | None = None,
) -> CompactionPreparation | None:
    """准备压缩：检查是否需要压缩。

    当消息数超过阈值时返回 CompactionPreparation，否则返回 None。
    """
    if len(messages) < 50:
        return None

    return CompactionPreparation(
        messages=messages,
        read_files=read_files or set(),
        modified_files=modified_files or set(),
    )


def compact(prep: CompactionPreparation, summary: str = "") -> CompactionResult:
    """执行压缩（占位实现）。"""
    return CompactionResult(
        summary=summary or "[conversation compacted]",
        messages=prep.messages[-10:],
        tokens_saved=max(0, len(prep.messages) - 10) * 100,
    )
