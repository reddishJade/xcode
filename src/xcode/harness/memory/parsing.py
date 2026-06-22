"""记忆块解析与评分辅助。

从 manager.py 提取的数据类型、字段解析、评分调整和分词工具函数。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

# 质量门阈值
_MIN_BLOCK_LENGTH = 50
_MIN_FIELD_CONTENT_LENGTH = 3
_NOVELTY_THRESHOLD = 0.85
_DEFAULT_MAX_BLOCKS = 0


@dataclass(frozen=True)
class MemoryRecord:
    """表示一个可检索的结构化记忆块。"""

    block: str
    title: str
    fields: dict[str, str]
    score: float = 0.0
    layer: str = "project"


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


# ── 解析 ──


def parse_memory_record(block: str) -> MemoryRecord:
    title = ""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
        elif line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return MemoryRecord(block=block, title=title, fields=fields)


def extract_title(block: str) -> str:
    for line in block.splitlines():
        if line.startswith("## "):
            return line[3:].strip()
    return ""


def extract_field_content(block: str, field_name: str) -> str | None:
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            key, value = stripped[2:].split(":", 1)
            if key.strip().lower() == field_name.lower():
                return value.strip()
    return None


def parse_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            key, value = stripped[2:].split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields


# ── 评分 ──


def adjust_score(
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
    confidence = parse_confidence(record.fields.get("confidence", ""))
    if confidence is not None:
        adjusted *= 0.75 + min(max(confidence, 0.0), 1.0) * 0.5
    if scope:
        adjusted *= scope_multiplier(record, scope)
    query_terms = set(tokenize(query))
    provenance_text = " ".join(
        record.fields.get(key, "")
        for key in ("source", "session", "validated", "validation")
    )
    if query_terms and query_terms.intersection(tokenize(provenance_text)):
        adjusted *= 1.1
    return adjusted


def scope_multiplier(record: MemoryRecord, scope: str) -> float:
    scope_terms = set(tokenize(scope))
    if not scope_terms:
        return 1.0
    scoped_text = " ".join(
        record.fields.get(key, "")
        for key in ("scope", "files", "context/query", "takeaways")
    )
    scoped_terms = set(tokenize(scoped_text))
    if scope_terms.intersection(scoped_terms):
        return 1.35
    if record.fields.get("scope"):
        return 0.75
    return 1.0


def parse_confidence(value: str) -> float | None:
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


# ── 元数据 ──


def with_metadata(
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


# ── 分词 ──


def tokenize(text: str) -> list[str]:
    return [
        w.lower()
        for w in re.sub(r"[^\w\s-]", "", text).replace("-", " ").split()
        if len(w) >= 2
    ]


def tokenize_set(text: str) -> set[str]:
    return set(tokenize(text))
