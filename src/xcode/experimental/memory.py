"""基于 BM25 的 MEMORY.md 记忆系统。

支持质量门（拒绝低质量/重复块）、冲突合并（同标题合并字段）、
LRU 遗忘策略（超过 max_blocks 时淘汰最久未访问的块）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from collections.abc import Sequence

from .bm25 import BM25Okapi
from .memory_parsing import (
    MemoryRecord,
    MemorySearchEvalCase,
    MemorySearchEvalResult,
    _DEFAULT_MAX_BLOCKS,
    _MIN_BLOCK_LENGTH,
    _MIN_FIELD_CONTENT_LENGTH,
    _NOVELTY_THRESHOLD,
    adjust_score,
    extract_field_content,
    extract_title,
    parse_fields,
    parse_memory_record,
    tokenize,
    tokenize_set,
    with_metadata,
)


class MemoryManager:
    """基于 H2 契约校验、BM25 召回和元数据重排的 MEMORY.md 记忆系统。"""

    def __init__(self, root: Path, max_blocks: int = _DEFAULT_MAX_BLOCKS) -> None:
        self.root = root
        self.memory_file = root / "MEMORY.md"
        self.archive_dir = root / ".local" / "memory_archive"
        self.lru_file = root / ".local" / "memory_lru.json"
        self.max_blocks = max_blocks

    # ── 读取 ──

    def read_memory_blocks(self) -> list[str]:
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
        return [parse_memory_record(block) for block in self.read_memory_blocks()]

    # ── 检索 ──

    def search_memory(
        self, query: str, limit: int = 3, scope: str | None = None
    ) -> list[str]:
        records = self.search_memory_records(query, limit=limit, scope=scope)
        self._touch_lru(records)
        return [record.block for record in records]

    def search_memory_records(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
    ) -> list[MemoryRecord]:
        records = self.read_memory_records()
        blocks = [record.block for record in records]
        if not blocks:
            return []

        corpus = [tokenize(block) for block in blocks]
        query_words = tokenize(query)

        bm25 = BM25Okapi(corpus)
        raw = list(bm25.get_scores(query_words))

        # 归一化 BM25 得分到 [0,1]。rank_bm25 的标准 IDF 在小语料下
        # 得分差异可能极小或全零，此时退化到 term 命中率来保持区分度
        if not raw:
            scores = []
        elif max(raw) - min(raw) > 1e-6:
            lo, hi = min(raw), max(raw)
            scores = [(s - lo) / (hi - lo) for s in raw]
        else:
            query_set = set(query_words)
            scores = [
                sum(q in b for q in query_set) / max(len(query_words), 1)
                for b in corpus
            ]

        ranked: list[MemoryRecord] = []
        for score, record in zip(scores, records, strict=True):
            adjusted = adjust_score(score, record, query, scope)
            if adjusted > 0:
                ranked.append(
                    MemoryRecord(
                        block=record.block,
                        title=record.title,
                        fields=record.fields,
                        score=round(adjusted, 6),
                    )
                )
        ranked.sort(key=lambda r: (-r.score, r.title))
        return ranked[:limit]

    # ── 评测 ──

    def evaluate_search(
        self,
        cases: list[MemorySearchEvalCase],
        limit: int = 3,
    ) -> list[MemorySearchEvalResult]:
        results: list[MemorySearchEvalResult] = []
        for case in cases:
            records = self.search_memory_records(
                case.query, limit=limit, scope=case.scope
            )
            titles = tuple(r.title for r in records)
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
        content = block.strip()
        if not content.startswith("## "):
            return False
        for field in ["Context/Query", "Solution", "Files", "Takeaways"]:
            if field not in content:
                return False
        return True

    def _content_quality_check(self, block: str) -> bool:
        content = block.strip()
        if len(content) < _MIN_BLOCK_LENGTH:
            return False
        for field_name in ("Context/Query", "Solution", "Files", "Takeaways"):
            field_value = extract_field_content(content, field_name)
            if (
                field_value is None
                or len(field_value.strip()) < _MIN_FIELD_CONTENT_LENGTH
            ):
                return False
        return True

    def _quality_check(
        self, block: str, existing_records: list[MemoryRecord] | None = None
    ) -> bool:
        if not self._content_quality_check(block):
            return False
        if existing_records and len(existing_records) > 0:
            if self._is_duplicate(block, existing_records):
                return False
        return True

    def _is_duplicate(self, block: str, existing_records: list[MemoryRecord]) -> bool:
        new_tokens = tokenize_set(block)
        if not new_tokens:
            return False
        for record in existing_records:
            old_tokens = tokenize_set(record.block)
            if not old_tokens:
                continue
            overlap = len(new_tokens & old_tokens) / min(
                len(new_tokens), len(old_tokens)
            )
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
        if not self.validate_memory_block(block):
            self._archive_block(block)
            return False

        block = with_metadata(block, source=source, scope=scope, confidence=confidence)
        existing_records = self.read_memory_records()
        new_title = extract_title(block)

        if new_title and existing_records:
            merged_block = self._merge_with_existing(block, new_title, existing_records)
            if merged_block is not None:
                if not self._content_quality_check(merged_block):
                    self._archive_block(merged_block)
                    return False
                self._replace_block_by_title(new_title, merged_block)
                self._enforce_lru()
                return True

        if not self._quality_check(block, existing_records):
            self._archive_block(block)
            return False

        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write("\n" + block.strip() + "\n")
        self._enforce_lru()
        return True

    def _replace_block_by_title(self, title: str, new_block: str) -> None:
        blocks = self.read_memory_blocks()
        updated = []
        found = False
        for existing_block in blocks:
            existing_title = extract_title(existing_block)
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
        new_lower = new_title.lower()
        for record in existing_records:
            if record.title.lower() == new_lower:
                new_fields = parse_fields(new_block)
                old_fields = record.fields
                merged = {}
                merged.update(old_fields)
                for key, value in new_fields.items():
                    if value.strip():
                        merged[key] = value
                merged["last_modified"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime()
                )

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
        has_field = any(
            f in block for f in ["Context/Query", "Solution", "Files", "Takeaways"]
        )
        return block.strip().startswith("## ") and (
            has_field or "incident" in block.lower()
        )

    def _archive_block(self, block: str) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        archive_file = self.archive_dir / f"corrupt_{timestamp}.md"
        archive_file.write_text(block, encoding="utf-8")

    # ── LRU 遗忘策略 ──

    def _touch_lru(self, records: Sequence[MemoryRecord]) -> None:
        lru = self._read_lru()
        now = time.time()
        for record in records:
            lru[record.title] = now
        self._write_lru(lru)

    def _enforce_lru(self) -> None:
        if self.max_blocks <= 0:
            return
        blocks = self.read_memory_blocks()
        block_titles = [extract_title(b) for b in blocks]
        block_title_set = set(t for t in block_titles if t)

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

        sorted_titles = sorted(lru.keys(), key=lambda t: lru.get(t, 0.0))
        titles_to_evict = set(sorted_titles[: len(blocks) - self.max_blocks])

        kept_blocks = []
        for block in blocks:
            title = extract_title(block)
            if title in titles_to_evict:
                self._archive_block(block)
            else:
                kept_blocks.append(block)

        if len(kept_blocks) < len(blocks):
            self._write_blocks(kept_blocks)
            for title in titles_to_evict:
                lru.pop(title, None)
            self._write_lru(lru)

    def _read_lru(self) -> dict[str, float]:
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
        self.lru_file.parent.mkdir(parents=True, exist_ok=True)
        self.lru_file.write_text(
            json.dumps(lru, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_blocks(self, blocks: list[str]) -> None:
        content = "\n".join(b.rstrip() for b in blocks if b.strip())
        self.memory_file.write_text(content + "\n", encoding="utf-8")
