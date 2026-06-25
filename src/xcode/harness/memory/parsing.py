"""记忆块解析与评分辅助。

从 manager.py 提取的数据类型、字段解析、评分调整和分词工具函数。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

# 质量门阈值
_MIN_BLOCK_LENGTH = 50
_MIN_FIELD_CONTENT_LENGTH = 3
_NOVELTY_THRESHOLD = 0.85
_DEFAULT_MAX_BLOCKS = 0

type MemoryType = Literal["semantic", "episodic", "procedural", "preference"]


@dataclass(frozen=True)
class MemoryEvidence:
    """结构化证据引用。"""

    kind: str
    reference: str


@dataclass(frozen=True)
class MemoryRecord:
    """表示一个可检索的结构化记忆块。"""

    block: str
    title: str
    fields: dict[str, str]
    memory_id: str
    memory_type: MemoryType
    scope: str | None = None
    source_session: str | None = None
    related_files: tuple[str, ...] = ()
    related_symbols: tuple[str, ...] = ()
    created_at: str | None = None
    modified_at: str | None = None
    confidence_value: float | None = None
    status: str = "active"
    validity: str = "unknown"
    supersedes: tuple[str, ...] = ()
    evidence: tuple[MemoryEvidence, ...] = ()
    retrieval_count: int = 0
    injection_count: int = 0
    adoption_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    correction_count: int = 0
    utility: float = 0.0
    last_outcome: str | None = None
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


type MemoryTraceEventType = Literal[
    "candidate_created",
    "accepted",
    "rejected",
    "quarantined",
    "retrieved",
    "injected",
    "tool_searched",
    "used",
    "superseded",
    "forgotten",
]


@dataclass(frozen=True)
class MemoryTraceEvent:
    """描述 memory 生命周期中的结构化事件。"""

    type: MemoryTraceEventType
    memory_id: str | None = None
    layer: str | None = None
    title: str | None = None
    score: float | None = None
    token_count: int | None = None
    latency_ms: float | None = None
    rejection_reason: str | None = None
    source: str | None = None
    timestamp: float = field(default_factory=time.time)


def build_memory_id(*, layer: str, title: str) -> str:
    """基于层级和标题生成不泄露内容的临时 memory id。"""
    digest = sha256(f"{layer}:{title.strip().lower()}".encode("utf-8")).hexdigest()
    return f"mem_{digest[:12]}"


# ── 解析 ──


def parse_memory_record(block: str, *, layer: str = "project") -> MemoryRecord:
    title = ""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
        elif line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            fields[key.strip().lower()] = value.strip()
    memory_id = fields.get("memory-id") or build_memory_id(layer=layer, title=title)
    memory_type = parse_memory_type(
        fields.get("memory-type"), title=title, fields=fields
    )
    return MemoryRecord(
        block=block,
        title=title,
        fields=fields,
        memory_id=memory_id,
        memory_type=memory_type,
        scope=fields.get("scope"),
        source_session=fields.get("source-session") or fields.get("source"),
        related_files=_parse_list_field(
            fields.get("related-files") or fields.get("files")
        ),
        related_symbols=_parse_list_field(fields.get("related-symbols")),
        created_at=fields.get("created"),
        modified_at=fields.get("modified") or fields.get("last_modified"),
        confidence_value=parse_confidence(fields.get("confidence", "")),
        status=(fields.get("status") or "active").strip().lower(),
        validity=(fields.get("validity") or "unknown").strip().lower(),
        supersedes=_parse_list_field(fields.get("supersedes")),
        evidence=parse_evidence_field(fields.get("evidence")),
        retrieval_count=_parse_int_field(fields.get("retrieval-count")),
        injection_count=_parse_int_field(fields.get("injection-count")),
        adoption_count=_parse_int_field(fields.get("adoption-count")),
        success_count=_parse_int_field(fields.get("success-count")),
        failure_count=_parse_int_field(fields.get("failure-count")),
        correction_count=_parse_int_field(fields.get("correction-count")),
        utility=_parse_float_field(fields.get("utility")) or 0.0,
        last_outcome=fields.get("last-outcome"),
        layer=layer,
    )


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


def parse_memory_type(
    value: str | None,
    *,
    title: str,
    fields: dict[str, str],
) -> MemoryType:
    text = (value or "").strip().lower()
    if text in {"semantic", "episodic", "procedural", "preference"}:
        return text
    combined = " ".join(
        [
            title.lower(),
            fields.get("context/query", "").lower(),
            fields.get("solution", "").lower(),
            fields.get("takeaways", "").lower(),
        ]
    )
    if "preference" in combined or "prefer " in combined:
        return "preference"
    if any(
        token in combined
        for token in ("incident", "error", "bug", "crash", "timeout", "failure")
    ):
        return "episodic"
    if any(
        token in combined
        for token in ("always", "steps", "procedure", "workflow", "checklist", "run ")
    ) or title.lower().startswith("how to"):
        return "procedural"
    return "semantic"


def parse_evidence_field(value: str | None) -> tuple[MemoryEvidence, ...]:
    if not value:
        return ()
    items: list[MemoryEvidence] = []
    for part in value.split(";"):
        entry = part.strip()
        if not entry:
            continue
        kind, separator, reference = entry.partition(":")
        if not separator:
            continue
        normalized_kind = kind.strip().lower()
        normalized_reference = reference.strip()
        if normalized_kind and normalized_reference:
            items.append(MemoryEvidence(normalized_kind, normalized_reference))
    return tuple(items)


def format_evidence_field(evidence: tuple[MemoryEvidence, ...]) -> str:
    return "; ".join(f"{item.kind}:{item.reference}" for item in evidence)


def _parse_list_field(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_field(value: str | None) -> int:
    if not value:
        return 0
    try:
        return max(0, int(value.strip()))
    except ValueError:
        return 0


def _parse_float_field(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


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
    status = record.status
    if status in {"deprecated", "superseded", "obsolete"}:
        adjusted *= 0.2
    confidence = record.confidence_value
    if confidence is not None:
        adjusted *= 0.75 + min(max(confidence, 0.0), 1.0) * 0.5
    if record.status == "needs_review":
        adjusted *= 0.7
    if record.utility != 0.0:
        adjusted *= max(0.4, min(1.6, 1.0 + record.utility * 0.25))
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
    layer: str,
    source: str | None,
    scope: str | None,
    confidence: float | None,
    memory_type: MemoryType | None = None,
    status: str | None = None,
    validity: str | None = None,
    supersedes: tuple[str, ...] = (),
    evidence: tuple[MemoryEvidence, ...] = (),
    retrieval_count: int | None = None,
    injection_count: int | None = None,
    adoption_count: int | None = None,
    success_count: int | None = None,
    failure_count: int | None = None,
    correction_count: int | None = None,
    utility: float | None = None,
    last_outcome: str | None = None,
) -> str:
    lines = [line.rstrip() for line in block.strip().splitlines()]
    existing = {
        line[2:].split(":", 1)[0].strip().lower()
        for line in lines
        if line.startswith("- ") and ":" in line
    }
    title = extract_title(block)
    additions = []
    if title and "memory-id" not in existing:
        additions.append(f"- Memory-ID: {build_memory_id(layer=layer, title=title)}")
    normalized_type = memory_type or parse_memory_type(
        None, title=title, fields=parse_fields(block)
    )
    if "memory-type" not in existing:
        additions.append(f"- Memory-Type: {normalized_type}")
    if source and "source" not in existing:
        additions.append(f"- Source: {source}")
    if source and "source-session" not in existing:
        additions.append(f"- Source-Session: {source}")
    if scope and "scope" not in existing:
        additions.append(f"- Scope: {scope}")
    if confidence is not None and "confidence" not in existing:
        bounded = min(max(confidence, 0.0), 1.0)
        additions.append(f"- Confidence: {bounded:.2f}")
    if status and "status" not in existing:
        additions.append(f"- Status: {status}")
    if validity and "validity" not in existing:
        additions.append(f"- Validity: {validity}")
    if supersedes and "supersedes" not in existing:
        additions.append(f"- Supersedes: {', '.join(supersedes)}")
    if evidence and "evidence" not in existing:
        additions.append(f"- Evidence: {format_evidence_field(evidence)}")
    if retrieval_count is not None and "retrieval-count" not in existing:
        additions.append(f"- Retrieval-Count: {max(0, retrieval_count)}")
    if injection_count is not None and "injection-count" not in existing:
        additions.append(f"- Injection-Count: {max(0, injection_count)}")
    if adoption_count is not None and "adoption-count" not in existing:
        additions.append(f"- Adoption-Count: {max(0, adoption_count)}")
    if success_count is not None and "success-count" not in existing:
        additions.append(f"- Success-Count: {max(0, success_count)}")
    if failure_count is not None and "failure-count" not in existing:
        additions.append(f"- Failure-Count: {max(0, failure_count)}")
    if correction_count is not None and "correction-count" not in existing:
        additions.append(f"- Correction-Count: {max(0, correction_count)}")
    if utility is not None and "utility" not in existing:
        additions.append(f"- Utility: {utility:.2f}")
    if last_outcome and "last-outcome" not in existing:
        additions.append(f"- Last-Outcome: {last_outcome}")
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
