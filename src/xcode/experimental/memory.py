from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from pathlib import Path
from typing import Sequence
from ..experimental.bm25 import BM25Okapi

# 质量门阈值
_MIN_BLOCK_LENGTH = 50  # 最短有效记忆块字符数
_MIN_FIELD_CONTENT_LENGTH = 3  # 每个必填字段最少内容字符数（文件名如 tasks.py 只有 8 字符）
_NOVELTY_THRESHOLD = 0.85  # Jaccard containment 上限，超过视为重复

# 遗忘策略默认值
_DEFAULT_MAX_BLOCKS = 0  # 默认不限制；设为正数后启用 LRU 淘汰


@dataclass(frozen=True)
class MemoryRecord:
    block: str
    title: str
    fields: dict[str, str]
    score: float = 0.0


@dataclass(frozen=True)
class MemorySearchEvalCase:
    query: str
    expected_title_contains: str
    scope: str | None = None


@dataclass(frozen=True)
class MemorySearchEvalResult:
    query: str
    passed: bool
    expected_title_contains: str
    matched_titles: tuple[str, ...]


class MemoryManager:
    """基于 H2 契约校验、BM25 召回和元数据重排的 MEMORY.md 记忆系统。

    支持质量门（拒绝低质量/重复块）、冲突合并（同标题合并字段）、
    LRU 遗忘策略（超过 max_blocks 时淘汰最久未访问的块）。
    """

    def __init__(
        self,
        root: Path,
        max_blocks: int = _DEFAULT_MAX_BLOCKS,
    ) -> None:
        self.root = root
        self.memory_file = root / "MEMORY.md"
        self.archive_dir = root / ".local" / "memory_archive"
        self.lru_file = root / ".local" / "memory_lru.json"
        self.max_blocks = max_blocks  # 0 表示不限制

    # ── 读取 ──

    def read_memory_blocks(self) -> list[str]:
        """读取 MEMORY.md 并以 H2 (## ) 开头划分为独立 incident 块。"""
        if not self.memory_file.exists():
            return []

        content = self.memory_file.read_text(encoding="utf-8")
        blocks = []
        current_block: list[str] = []

        for line in content.splitlines():
            if line.startswith("## "):
                if current_block:
                    blocks.append("\n".join(current_block) + "\n")
                current_block = [line]
            else:
                if current_block or line.strip():
                    current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block) + "\n")

        return [b for b in blocks if b.strip()]

    def read_memory_records(self) -> list[MemoryRecord]:
        """读取记忆块并解析字段，保留原始块以便可追溯。"""
        return [_parse_memory_record(block) for block in self.read_memory_blocks()]

    # ── 检索 ──

    def search_memory(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
    ) -> list[str]:
        """检索 MEMORY.md 中的相关 incident，保持旧接口返回原始块。"""
        records = self.search_memory_records(query, limit=limit, scope=scope)
        self._touch_lru(records)
        return [record.block for record in records]

    def search_memory_records(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
    ) -> list[MemoryRecord]:
        """使用 BM25 召回，再按 scope、置信度和状态做轻量重排。"""
        records = self.read_memory_records()
        blocks = [record.block for record in records]
        if not blocks:
            return []

        corpus = []
        for block in blocks:
            words = [
                w.lower()
                for w in re.sub(r"[^\w\s-]", "", block).replace("-", " ").split()
                if len(w) >= 2
            ]
            corpus.append(words)

        query_words = [
            w.lower()
            for w in re.sub(r"[^\w\s-]", "", query).replace("-", " ").split()
            if len(w) >= 2
        ]

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_words)

        ranked: list[MemoryRecord] = []
        for score, record in zip(scores, records, strict=True):
            adjusted = _adjust_score(score, record, query, scope)
            if adjusted > 0:
                ranked.append(
                    MemoryRecord(
                        block=record.block,
                        title=record.title,
                        fields=record.fields,
                        score=round(adjusted, 6),
                    )
                )
        ranked.sort(key=lambda record: (-record.score, record.title))
        return ranked[:limit]

    # ── 评测 ──

    def evaluate_search(
        self,
        cases: list[MemorySearchEvalCase],
        limit: int = 3,
    ) -> list[MemorySearchEvalResult]:
        """对记忆召回做最小可重复评测，检查期望标题是否出现在 top-k。"""
        results: list[MemorySearchEvalResult] = []
        for case in cases:
            records = self.search_memory_records(
                case.query,
                limit=limit,
                scope=case.scope,
            )
            titles = tuple(record.title for record in records)
            expected = case.expected_title_contains
            results.append(
                MemorySearchEvalResult(
                    query=case.query,
                    passed=any(expected in title for title in titles),
                    expected_title_contains=expected,
                    matched_titles=titles,
                )
            )
        return results

    # ── 校验 ──

    def validate_memory_block(self, block: str) -> bool:
        """基础校验：H2 开头 + 4 个必填字段。"""
        content = block.strip()
        if not content.startswith("## "):
            return False

        mandatory_fields = ["Context/Query", "Solution", "Files", "Takeaways"]
        for field in mandatory_fields:
            if field not in content:
                return False
        return True

    def _content_quality_check(self, block: str) -> bool:
        """内容质量：长度和必填字段内容长度。不检查新颖性。"""
        content = block.strip()
        if len(content) < _MIN_BLOCK_LENGTH:
            return False
        for field_name in ("Context/Query", "Solution", "Files", "Takeaways"):
            field_value = _extract_field_content(content, field_name)
            if field_value is None or len(field_value.strip()) < _MIN_FIELD_CONTENT_LENGTH:
                return False
        return True

    def _quality_check(self, block: str, existing_records: list[MemoryRecord] | None = None) -> bool:
        """完整质量门：内容质量 + 新颖性。用于新标题块。"""
        if not self._content_quality_check(block):
            return False
        if existing_records and len(existing_records) > 0:
            if self._is_duplicate(block, existing_records):
                return False
        return True

    def _is_duplicate(self, block: str, existing_records: list[MemoryRecord]) -> bool:
        """使用 Jaccard containment 判定块是否与已有块重复。"""
        new_tokens = _tokenize_set(block)
        if not new_tokens:
            return False

        for record in existing_records:
            old_tokens = _tokenize_set(record.block)
            if not old_tokens:
                continue
            overlap = len(new_tokens & old_tokens) / min(len(new_tokens), len(old_tokens))
            if overlap >= _NOVELTY_THRESHOLD:
                return True
        return False

    # ── 合并与写入 ──

    def add_memory_block(
        self,
        block: str,
        *,
        source: str | None = None,
        scope: str | None = None,
        confidence: float | None = None,
    ) -> bool:
        """追加或合并记忆块。

        流程：
          基础校验
          -> 添加元数据
          -> 提取标题
          -> 如果同标题存在：直接字段 merge（跳过质量门/新颖性）
          -> 否则：质量门 + 新颖性校验 -> 追加
          -> LRU 遗忘

        校验失败或质量门不通过时归档至 archive 目录。
        """
        if not self.validate_memory_block(block):
            self._archive_block(block)
            return False

        # 先添加元数据（合并时也会用到）
        block = _with_metadata(
            block,
            source=source,
            scope=scope,
            confidence=confidence,
        )

        existing_records = self.read_memory_records()
        new_title = _extract_title(block)

        # 优先处理同标题合并（跳过新颖性，但保留内容质量检查）
        if new_title and existing_records:
            merged_block = self._merge_with_existing(
                block, new_title, existing_records
            )
            if merged_block is not None:
                # 合并后只做内容质量检查，跳过新颖性（同标题不判重）
                if not self._content_quality_check(merged_block):
                    self._archive_block(merged_block)
                    return False
                self._replace_block_by_title(new_title, merged_block)
                self._enforce_lru()
                return True

        # 新标题：走完整质量门（内容质量 + 新颖性）
        if not self._quality_check(block, existing_records):
            self._archive_block(block)
            return False

        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write("\n" + block.strip() + "\n")

        self._enforce_lru()
        return True

    def _replace_block_by_title(self, title: str, new_block: str) -> None:
        """在 MEMORY.md 中按标题替换块。"""
        blocks = self.read_memory_blocks()
        updated = []
        found = False
        for existing_block in blocks:
            existing_title = _extract_title(existing_block)
            if existing_title and existing_title.lower() == title.lower():
                updated.append(new_block.strip() + "\n")
                found = True
            else:
                updated.append(existing_block.rstrip() + "\n")
        if not found:
            updated.append(new_block.strip() + "\n")
        self.memory_file.write_text("".join(updated), encoding="utf-8")

    def _merge_with_existing(
        self,
        new_block: str,
        new_title: str,
        existing_records: list[MemoryRecord],
    ) -> str | None:
        """查找同标题已有块并合并字段。返回合并后的块，无匹配返回 None。

        合并策略：新字段中非空值覆盖旧字段，空值保留旧值。
        """
        new_lower = new_title.lower()
        for record in existing_records:
            if record.title.lower() == new_lower:
                new_fields = _parse_fields(new_block)
                old_fields = record.fields

                # 合并：只覆盖非空新值，其余保留旧值
                merged = {}
                merged.update(old_fields)
                for key, value in new_fields.items():
                    if value.strip():
                        merged[key] = value
                merged["last_modified"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime()
                )

                # 重建块内容
                lines = [f"## {new_title}"]
                mandatory = ["Context/Query", "Solution", "Files", "Takeaways"]
                for field in mandatory:
                    key = field.lower()
                    if key in merged:
                        lines.append(f"- {field}: {merged.pop(key)}")
                for key, value in merged.items():
                    display_key = key.capitalize() if "_" not in key else key
                    lines.append(f"- {display_key}: {value}")
                return "\n".join(lines) + "\n"
        return None

    def consolidate(self, summary: str) -> None:
        """从压缩产生的摘要中提取潜在的记忆块，交给 add_memory_block 处理。

        add_memory_block 内部已包含合并、质量门、新颖性和归档的全部逻辑，
        consolidate 不需要重复 _quality_check。
        """
        parts = summary.split("## ")
        if len(parts) <= 1:
            return

        for part in parts[1:]:
            segment = part.split("\n- ")[0].strip()
            block = "## " + segment

            for field in ["Context/Query", "Solution", "Files", "Takeaways"]:
                block = block.replace(f" - {field}", f"\n- {field}")
                block = block.replace(f"- {field}", f"\n- {field}")

            block = "\n".join(
                line.strip() for line in block.splitlines() if line.strip()
            )

            if self._is_memory_attempt(block):
                self.add_memory_block(block)

    def _is_memory_attempt(self, block: str) -> bool:
        """判定块是否为记忆块尝试。"""
        has_field = any(
            f in block for f in ["Context/Query", "Solution", "Files", "Takeaways"]
        )
        return block.strip().startswith("## ") and (
            has_field or "incident" in block.lower()
        )

    def _archive_block(self, block: str) -> None:
        """将不合格的记忆块归档。"""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        archive_file = self.archive_dir / f"corrupt_{timestamp}.md"
        archive_file.write_text(block, encoding="utf-8")

    # ── LRU 遗忘策略 ──

    def _touch_lru(self, records: Sequence[MemoryRecord]) -> None:
        """更新被检索到的块的访问时间。"""
        lru = self._read_lru()
        now = time.time()
        for record in records:
            lru[record.title] = now
        self._write_lru(lru)

    def _enforce_lru(self) -> None:
        """读取 MEMORY.md 的块数，超过 max_blocks 时淘汰最久未访问的块。

        max_blocks=0 时跳过（默认）。
        淘汰的块先归档再删除，防止信息丢失。
        """
        if self.max_blocks <= 0:
            return

        blocks = self.read_memory_blocks()
        block_titles = [_extract_title(b) for b in blocks]
        block_title_set = set(t for t in block_titles if t)

        # 任何时候都清理 stale LRU key
        lru = self._read_lru()
        cleaned = {title: ts for title, ts in lru.items() if title in block_title_set}
        if cleaned != lru:
            self._write_lru(cleaned)
            lru = cleaned

        if len(blocks) <= self.max_blocks:
            return
        now = time.time()
        for title in block_titles:
            if title and title not in lru:
                lru[title] = now

        # 按访问时间排序，保留最近 max_blocks 个
        sorted_titles = sorted(lru.keys(), key=lambda t: lru.get(t, 0.0))
        titles_to_evict = set(sorted_titles[: len(blocks) - self.max_blocks])

        kept_blocks = []
        for block in blocks:
            title = _extract_title(block)
            if title in titles_to_evict:
                # 淘汰前归档
                self._archive_block(block)
            else:
                kept_blocks.append(block)

        if len(kept_blocks) < len(blocks):
            self._write_blocks(kept_blocks)
            for title in titles_to_evict:
                lru.pop(title, None)
            self._write_lru(lru)

    def _read_lru(self) -> dict[str, float]:
        """读取 LRU 访问时间记录。"""
        if not self.lru_file.exists():
            return {}
        try:
            data = json.loads(self.lru_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            pass
        return {}

    def _write_lru(self, lru: dict[str, float]) -> None:
        """写入 LRU 访问时间记录。"""
        self.lru_file.parent.mkdir(parents=True, exist_ok=True)
        self.lru_file.write_text(
            json.dumps(lru, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_blocks(self, blocks: list[str]) -> None:
        """重写 MEMORY.md。"""
        content = "\n".join(b.rstrip() for b in blocks if b.strip())
        self.memory_file.write_text(content + "\n", encoding="utf-8")


# ── 辅助函数 ──


def _parse_memory_record(block: str) -> MemoryRecord:
    title = ""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
        elif line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return MemoryRecord(block=block, title=title, fields=fields)


def _extract_title(block: str) -> str:
    """从记忆块中提取 H2 标题。"""
    for line in block.splitlines():
        if line.startswith("## "):
            return line[3:].strip()
    return ""


def _extract_field_content(block: str, field_name: str) -> str | None:
    """提取指定字段的值内容。"""
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            key, value = stripped[2:].split(":", 1)
            if key.strip().lower() == field_name.lower():
                return value.strip()
    return None


def _parse_fields(block: str) -> dict[str, str]:
    """从块中解析所有键值字段。"""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            key, value = stripped[2:].split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields


def _adjust_score(
    score: float,
    record: MemoryRecord,
    query: str,
    scope: str | None,
) -> float:
    adjusted = score
    if adjusted <= 0:
        return 0.0
    status = record.fields.get("status", "").lower()
    if status in {"deprecated", "superseded", "obsolete"}:
        adjusted *= 0.2
    confidence = _parse_confidence(record.fields.get("confidence", ""))
    if confidence is not None:
        adjusted *= 0.75 + min(max(confidence, 0.0), 1.0) * 0.5
    if scope:
        adjusted *= _scope_multiplier(record, scope)
    query_terms = set(_tokenize(query))
    provenance_text = " ".join(
        record.fields.get(key, "")
        for key in ("source", "session", "validated", "validation")
    )
    if query_terms and query_terms.intersection(_tokenize(provenance_text)):
        adjusted *= 1.1
    return adjusted


def _scope_multiplier(record: MemoryRecord, scope: str) -> float:
    scope_terms = set(_tokenize(scope))
    if not scope_terms:
        return 1.0
    scoped_text = " ".join(
        record.fields.get(key, "")
        for key in ("scope", "files", "context/query", "takeaways")
    )
    scoped_terms = set(_tokenize(scoped_text))
    if scope_terms.intersection(scoped_terms):
        return 1.35
    if record.fields.get("scope"):
        return 0.75
    return 1.0


def _parse_confidence(value: str) -> float | None:
    text = value.strip().lower()
    if not text:
        return None
    named = {"low": 0.25, "medium": 0.5, "high": 0.85, "verified": 1.0}
    if text in named:
        return named[text]
    try:
        return float(text)
    except ValueError:
        return None


def _with_metadata(
    block: str,
    *,
    source: str | None,
    scope: str | None,
    confidence: float | None,
) -> str:
    lines = [line.rstrip() for line in block.strip().splitlines()]
    existing = {
        line[2:].split(":", 1)[0].strip().lower()
        for line in lines
        if line.startswith("- ") and ":" in line
    }
    additions = []
    if source and "source" not in existing:
        additions.append(f"- Source: {source}")
    if scope and "scope" not in existing:
        additions.append(f"- Scope: {scope}")
    if confidence is not None and "confidence" not in existing:
        bounded = min(max(confidence, 0.0), 1.0)
        additions.append(f"- Confidence: {bounded:.2f}")
    if "created" not in existing:
        additions.append(f"- Created: {time.strftime('%Y-%m-%d', time.localtime())}")
    if additions:
        lines.extend(additions)
    return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    return [
        w.lower()
        for w in re.sub(r"[^\w\s-]", "", text).replace("-", " ").split()
        if len(w) >= 2
    ]


def _tokenize_set(text: str) -> set[str]:
    return set(_tokenize(text))
