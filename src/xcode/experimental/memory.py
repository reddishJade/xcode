from __future__ import annotations

from dataclasses import dataclass
import re
import time
from pathlib import Path
from ..experimental.bm25 import BM25Okapi


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
    """基于 H2 契约校验、BM25 召回和元数据重排的 MEMORY.md 记忆系统。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.memory_file = root / "MEMORY.md"
        self.archive_dir = root / ".local" / "memory_archive"

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

    def search_memory(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
    ) -> list[str]:
        """检索 MEMORY.md 中的相关 incident，保持旧接口返回原始块。"""
        return [
            record.block
            for record in self.search_memory_records(query, limit=limit, scope=scope)
        ]

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

    def validate_memory_block(self, block: str) -> bool:
        """校验内存块是否包含全部 4 个必填字段且以 H2 开头。"""
        content = block.strip()
        if not content.startswith("## "):
            return False

        mandatory_fields = ["Context/Query", "Solution", "Files", "Takeaways"]
        for field in mandatory_fields:
            if field not in content:
                return False
        return True

    def add_memory_block(
        self,
        block: str,
        *,
        source: str | None = None,
        scope: str | None = None,
        confidence: float | None = None,
    ) -> bool:
        """追加内存块。如果校验失败，则归档至 archive 目录。"""
        if self.validate_memory_block(block):
            block = _with_metadata(
                block,
                source=source,
                scope=scope,
                confidence=confidence,
            )
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write("\n" + block.strip() + "\n")
            return True
        else:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            archive_file = self.archive_dir / f"corrupt_{timestamp}.md"
            archive_file.write_text(block, encoding="utf-8")
            return False

    def consolidate(self, summary: str) -> None:
        """从压缩产生的摘要中提取并校验潜在的记忆块，写入 MEMORY.md，失败则归档。"""
        # 通过 H2 标题定位潜在记忆块。
        parts = summary.split("## ")
        if len(parts) <= 1:
            return

        for part in parts[1:]:
            # 摘要中出现下一条消息边界时，截断当前候选块。
            segment = part.split("\n- ")[0].strip()
            block = "## " + segment

            # 压缩预览可能把字段拍平成单行，这里恢复字段换行。
            for field in ["Context/Query", "Solution", "Files", "Takeaways"]:
                block = block.replace(f" - {field}", f"\n- {field}")
                block = block.replace(f"- {field}", f"\n- {field}")

            block = "\n".join(
                line.strip() for line in block.splitlines() if line.strip()
            )

            if self._is_memory_attempt(block):
                self.add_memory_block(block)

    def _is_memory_attempt(self, block: str) -> bool:
        """判定块是否为记忆块尝试（包含 H2 开头且包含字段或 Incident 关键词）。"""
        has_field = any(
            f in block for f in ["Context/Query", "Solution", "Files", "Takeaways"]
        )
        return block.strip().startswith("## ") and (
            has_field or "incident" in block.lower()
        )


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
