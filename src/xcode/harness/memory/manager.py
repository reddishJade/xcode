"""面向长任务的最小持久记忆。

MEMORY.md 是唯一事实源。检索只使用确定性的 BM25；会话连续性由
session surface 负责，不在长期记忆中维护反馈、效用或生命周期状态。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Literal

import filelock
from rank_bm25 import BM25Okapi

from .parsing import MemoryRecord, parse_memory_blocks, tokenize

type MemoryLayer = Literal["project", "user"]
type MemoryLayerFilter = Literal["all", "project", "user"]
type MemoryFileSignature = tuple[str, int, int, int] | tuple[str, None, None, None]


@dataclass(frozen=True)
class _MemorySearchIndex:
    """可按文件签名安全失效的内存检索索引。"""

    signature: tuple[MemoryFileSignature, ...]
    records: tuple[MemoryRecord, ...]
    corpus: tuple[list[str], ...]
    index: BM25Okapi | None


class MemoryManager:
    """管理可审查的项目级与用户级 Markdown 记忆。"""

    def __init__(
        self,
        root: Path,
        *,
        user_memory_file: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.memory_file = self.root / "MEMORY.md"
        self.user_memory_file = user_memory_file or (
            Path.home() / ".xcode" / "memory" / "MEMORY.md"
        )
        self._search_indexes: dict[MemoryLayerFilter, _MemorySearchIndex] = {}
        self._search_indexes_lock = threading.Lock()

    def read_memory_blocks(
        self,
        layer: MemoryLayerFilter = "all",
    ) -> list[str]:
        """读取指定层级中的原始 H2 记忆块。"""
        return [record.block for record in self.read_memory_records(layer)]

    def read_memory_records(
        self,
        layer: MemoryLayerFilter = "all",
    ) -> list[MemoryRecord]:
        """读取指定层级的记忆记录。"""
        records: list[MemoryRecord] = []
        for current_layer in self._selected_layers(layer):
            path = self._memory_file(current_layer)
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            records.extend(parse_memory_blocks(text, layer=current_layer))
        return records

    def search_memory_records(
        self,
        query: str,
        *,
        limit: int = 5,
        layer: MemoryLayerFilter = "all",
        scope: str | None = None,
        **_ignored: object,
    ) -> list[MemoryRecord]:
        """使用 BM25 检索长期记忆。

        `scope` 仅作为附加检索词，不触发隐藏的重排策略。多余关键字参数
        被忽略，以便旧插件在迁移期间仍能调用这个只读接口。
        """
        normalized = query.strip()
        if not normalized or limit <= 0:
            return []
        search_index = self._search_index(layer)
        if not search_index.records or search_index.index is None:
            return []

        query_text = "\n".join(part for part in (normalized, scope or "") if part)
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []
        raw_scores = search_index.index.get_scores(query_tokens)
        lowered = normalized.casefold()

        ranked: list[MemoryRecord] = []
        for record, document_tokens, raw_score in zip(
            search_index.records,
            search_index.corpus,
            raw_scores,
            strict=True,
        ):
            exact = lowered in record.search_text.casefold()
            overlap = len(set(query_tokens).intersection(document_tokens))
            if not exact and overlap == 0:
                continue
            score = (
                max(float(raw_score), 0.0) + float(overlap) + (1.0 if exact else 0.0)
            )
            ranked.append(replace(record, score=score))
        ranked.sort(
            key=lambda item: (
                -item.score,
                0 if item.layer == "project" else 1,
                item.title.casefold(),
            )
        )
        return ranked[: min(limit, 10)]

    def read_budgeted(
        self,
        max_tokens: int,
        layer: MemoryLayerFilter = "all",
    ) -> list[str]:
        """按文件顺序读取可装入预算的记忆，用于 resume/rebuild。"""
        if max_tokens <= 0:
            return []
        from xcode.agent._compaction import estimate_tokens

        selected: list[str] = []
        remaining = max_tokens
        for record in self.read_memory_records(layer):
            packet = self.render_prompt_packet(record)
            cost = estimate_tokens(packet)
            if cost > remaining:
                continue
            selected.append(packet)
            remaining -= cost
        return selected

    def add_memory_block(
        self,
        block: str,
        *,
        layer: MemoryLayer = "project",
        **_ignored: object,
    ) -> bool:
        """显式追加一条记忆；拒绝空记录和标题重复。"""
        incoming = self._parse_incoming_block(block, layer)
        if incoming is None:
            return False
        path = self._memory_file(layer)
        with self._file_lock(path):
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            existing = parse_memory_blocks(current, layer=layer)
            if self._duplicates(incoming, existing):
                return False
            prefix = current.rstrip()
            content = (
                f"{prefix}\n\n{incoming.block.strip()}\n"
                if prefix
                else (
                    f"# {'Project' if layer == 'project' else 'User'} memory\n\n"
                    f"{incoming.block.strip()}\n"
                )
            )
            self._atomic_write(path, content)
        self._invalidate_search_indexes()
        return True

    def update_memory_block(
        self,
        title: str,
        block: str,
        *,
        layer: MemoryLayer = "project",
    ) -> bool:
        """按标题原子替换一条记忆，并保留文件中的其他内容。"""
        incoming = self._parse_incoming_block(block, layer)
        target_title = title.strip().casefold()
        if incoming is None or not target_title:
            return False
        path = self._memory_file(layer)
        with self._file_lock(path):
            if not path.is_file():
                return False
            current = path.read_text(encoding="utf-8")
            spans = self._memory_block_spans(current)
            target = next(
                (span for span in spans if span[0].casefold() == target_title),
                None,
            )
            if target is None:
                return False
            existing = [
                record
                for record in parse_memory_blocks(current, layer=layer)
                if record.title.casefold() != target_title
            ]
            other_titles = {
                span_title.casefold()
                for span_title, _, _ in spans
                if span_title.casefold() != target_title
            }
            if incoming.title.casefold() in other_titles or self._duplicates(
                incoming, existing
            ):
                return False
            _, start, end = target
            suffix = current[end:].lstrip("\r\n")
            content = current[:start].rstrip()
            if content:
                content += "\n\n"
            content += incoming.block.strip()
            content += f"\n\n{suffix}" if suffix else "\n"
            self._atomic_write(path, content)
        self._invalidate_search_indexes()
        return True

    def delete_memory_block(
        self,
        title: str,
        *,
        layer: MemoryLayer = "project",
    ) -> bool:
        """按标题原子删除一条记忆，并保留文件头和其他记录。"""
        target_title = title.strip().casefold()
        if not target_title:
            return False
        path = self._memory_file(layer)
        with self._file_lock(path):
            if not path.is_file():
                return False
            current = path.read_text(encoding="utf-8")
            target = next(
                (
                    span
                    for span in self._memory_block_spans(current)
                    if span[0].casefold() == target_title
                ),
                None,
            )
            if target is None:
                return False
            _, start, end = target
            prefix = current[:start].rstrip()
            suffix = current[end:].lstrip("\r\n")
            if prefix and suffix:
                content = f"{prefix}\n\n{suffix}"
            elif prefix:
                content = f"{prefix}\n"
            else:
                content = suffix
            self._atomic_write(path, content)
        self._invalidate_search_indexes()
        return True

    def render_prompt_packet(self, record: MemoryRecord) -> str:
        """渲染带来源的记忆块。"""
        return f"[{record.layer} memory · {record.memory_id}]\n{record.block.strip()}"

    def render_search_result(self, record: MemoryRecord) -> str:
        """渲染 memory 工具结果。"""
        path = self._memory_file(record.layer)
        return (
            f"[{record.layer}] {record.title} "
            f"(score={record.score:.3f}, path={path})\n{record.block.strip()}"
        )

    def _memory_file(self, layer: MemoryLayer | str) -> Path:
        if layer not in {"project", "user"}:
            raise ValueError(f"unsupported memory layer: {layer}")
        return self.memory_file if layer == "project" else self.user_memory_file

    @staticmethod
    def _selected_layers(layer: MemoryLayerFilter) -> tuple[MemoryLayer, ...]:
        if layer == "project":
            return ("project",)
        if layer == "user":
            return ("user",)
        return ("project", "user")

    def _search_index(self, layer: MemoryLayerFilter) -> _MemorySearchIndex:
        signature = self._memory_signature(layer)
        with self._search_indexes_lock:
            cached = self._search_indexes.get(layer)
            if cached is not None and cached.signature == signature:
                return cached

        stable_signature: tuple[MemoryFileSignature, ...] = ()
        records: tuple[MemoryRecord, ...] = ()
        for _ in range(3):
            signature_before = self._memory_signature(layer)
            records = tuple(self.read_memory_records(layer))
            signature_after = self._memory_signature(layer)
            if signature_before == signature_after:
                stable_signature = signature_after
                break

        corpus = tuple(tokenize(record.search_text) for record in records)
        index = BM25Okapi(list(corpus)) if corpus else None
        rebuilt = _MemorySearchIndex(
            signature=stable_signature,
            records=records,
            corpus=corpus,
            index=index,
        )
        with self._search_indexes_lock:
            self._search_indexes[layer] = rebuilt
        return rebuilt

    def _memory_signature(
        self,
        layer: MemoryLayerFilter,
    ) -> tuple[MemoryFileSignature, ...]:
        signatures: list[MemoryFileSignature] = []
        for current_layer in self._selected_layers(layer):
            path = self._memory_file(current_layer)
            try:
                stat = path.stat()
            except FileNotFoundError:
                signatures.append((str(path), None, None, None))
                continue
            signatures.append((str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size))
        return tuple(signatures)

    def _invalidate_search_indexes(self) -> None:
        with self._search_indexes_lock:
            self._search_indexes.clear()

    @staticmethod
    def _file_lock(path: Path) -> filelock.FileLock:
        path.parent.mkdir(parents=True, exist_ok=True)
        return filelock.FileLock(f"{path}.lock", timeout=10)

    @staticmethod
    def _parse_incoming_block(
        block: str,
        layer: MemoryLayer,
    ) -> MemoryRecord | None:
        parsed = parse_memory_blocks(block, layer=layer)
        if len(parsed) != 1 or len(parsed[0].body.strip()) < 3:
            return None
        return parsed[0]

    @staticmethod
    def _duplicates(
        incoming: MemoryRecord,
        existing: list[MemoryRecord],
    ) -> bool:
        return any(
            record.title.casefold() == incoming.title.casefold()
            or record.body.casefold() == incoming.body.casefold()
            for record in existing
        )

    @staticmethod
    def _memory_block_spans(text: str) -> list[tuple[str, int, int]]:
        matches = list(re.finditer(r"(?m)^##[ \t]+(.+?)[ \t]*$", text))
        return [
            (
                match.group(1).strip(),
                match.start(),
                matches[index + 1].start() if index + 1 < len(matches) else len(text),
            )
            for index, match in enumerate(matches)
        ]

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        replaced = False
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            replaced = True
        finally:
            if not replaced:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass


def build_memory_block(title: str, body: str) -> str:
    """构建供 CLI 和调用方使用的最小 Markdown 记忆块。"""
    clean_title = re.sub(r"[\r\n]+", " ", title).strip()
    clean_body = body.strip()
    return f"## {clean_title}\n{clean_body}\n"
