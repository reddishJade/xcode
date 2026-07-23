"""MEMORY.md 的最小解析与分词。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re

_LEGACY_METADATA = {
    "memory-id",
    "memory-type",
    "source-session",
    "source-message",
    "created",
    "modified",
    "last_modified",
    "confidence",
    "status",
    "validity",
    "supersedes",
    "evidence",
    "retrieval-count",
    "injection-count",
    "reference-count",
    "adoption-count",
    "success-count",
    "failure-count",
    "correction-count",
    "utility",
    "last-outcome",
}
_RETIRED_LEGACY_STATUSES = {
    "candidate",
    "needs_review",
    "deprecated",
    "superseded",
    "obsolete",
}


@dataclass(frozen=True)
class MemoryRecord:
    """一条可审查、可检索的 Markdown 记忆。"""

    block: str
    title: str
    body: str
    memory_id: str
    layer: str = "project"
    score: float = 0.0

    @property
    def search_text(self) -> str:
        return f"{self.title}\n{self.body}"


def parse_memory_blocks(text: str, *, layer: str) -> list[MemoryRecord]:
    """把 Markdown 中的 H2 节解析为独立记录。"""
    matches = list(re.finditer(r"(?m)^##[ \t]+(.+?)[ \t]*$", text))
    records: list[MemoryRecord] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        if not title:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_body = text[match.end() : end].strip()
        fields = _legacy_fields(raw_body)
        if fields.get("status", "").casefold() in _RETIRED_LEGACY_STATUSES:
            continue
        body = _strip_legacy_metadata(raw_body)
        if not body:
            continue
        block = f"## {title}\n{body}"
        digest = sha256(f"{layer}:{title.casefold()}".encode("utf-8")).hexdigest()[:12]
        records.append(
            MemoryRecord(
                block=block,
                title=title,
                body=body,
                memory_id=f"mem_{digest}",
                layer=layer,
            )
        )
    return records


def _legacy_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in body.splitlines():
        match = re.match(r"^-[ \t]+([^:]+):[ \t]*(.*)$", line)
        if match is not None:
            fields[match.group(1).strip().casefold()] = match.group(2).strip()
    return fields


def _strip_legacy_metadata(body: str) -> str:
    lines: list[str] = []
    for line in body.splitlines():
        match = re.match(r"^-[ \t]+([^:]+):", line)
        key = match.group(1).strip().casefold() if match is not None else ""
        if key in _LEGACY_METADATA:
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def tokenize(text: str) -> list[str]:
    """生成适合代码、英文和中文的确定性 BM25 token。"""
    normalized = text.casefold()
    tokens = re.findall(r"[a-z0-9_./:-]+|[\u3400-\u9fff]+", normalized)
    expanded: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            expanded.extend(token)
            expanded.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            expanded.append(token)
    return expanded
