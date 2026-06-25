"""基于 BM25 的 MEMORY.md 记忆系统。

支持质量门（拒绝低质量/重复块）、冲突合并（同标题合并字段）、
LRU 遗忘策略（超过 max_blocks 时淘汰最久未访问的块）。
"""

from __future__ import annotations

from html import escape
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rank_bm25 import BM25Okapi

from .parsing import (
    MemoryEvidence,
    MemoryRecord,
    MemorySearchEvalCase,
    MemorySearchEvalResult,
    MemoryTraceEvent,
    MemoryType,
    _DEFAULT_MAX_BLOCKS,
    _MIN_BLOCK_LENGTH,
    _MIN_FIELD_CONTENT_LENGTH,
    _NOVELTY_THRESHOLD,
    adjust_score,
    build_memory_id,
    extract_field_content,
    extract_title,
    parse_fields,
    parse_memory_record,
    tokenize,
    tokenize_set,
    with_metadata,
)

type MemoryLayer = Literal["project", "user"]
type MemoryLayerFilter = Literal["all", "project", "user"]


class MemoryManager:
    """基于 H2 契约校验、BM25 召回和元数据重排的 MEMORY.md 记忆系统。"""

    def __init__(
        self,
        root: Path,
        max_blocks: int = _DEFAULT_MAX_BLOCKS,
        user_memory_file: Path | None = None,
        min_retrieval_score: float = 0.2,
        min_confidence: float = 0.0,
    ) -> None:
        """创建项目级与用户级并行的记忆管理器。"""
        self.root = root
        self.memory_file = root / "MEMORY.md"
        self.user_memory_file = user_memory_file or (
            Path.home() / ".xcode" / "memory" / "MEMORY.md"
        )
        self.archive_dir = root / ".local" / "memory_archive"
        self.lru_file = root / ".local" / "memory_lru.json"
        self.max_blocks = max_blocks
        self.min_retrieval_score = min_retrieval_score
        self.min_confidence = min_confidence
        self._trace_events: list[MemoryTraceEvent] = []

    # ── 读取 ──

    def read_memory_blocks(
        self,
        layer: MemoryLayerFilter = "all",
    ) -> list[str]:
        """读取指定层级的记忆块；默认合并项目级与用户级。"""
        blocks: list[str] = []
        for current_layer in self._selected_layers(layer):
            blocks.extend(self._read_blocks_from_file(self._memory_file(current_layer)))
        return blocks

    def read_memory_records(
        self,
        layer: MemoryLayerFilter = "all",
    ) -> list[MemoryRecord]:
        """读取指定层级并保留来源信息。"""
        records: list[MemoryRecord] = []
        for current_layer in self._selected_layers(layer):
            memory_file = self._memory_file(current_layer)
            for block in self._read_blocks_from_file(memory_file):
                record = parse_memory_record(block, layer=current_layer)
                records.append(record)
        return records

    def _read_blocks_from_file(self, memory_file: Path) -> list[str]:
        """从单个 MEMORY.md 文件解析 H2 记忆块。"""
        if not memory_file.exists():
            return []
        content = memory_file.read_text(encoding="utf-8")
        blocks: list[str] = []
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
        return [block for block in blocks if block.strip()]

    # ── 检索 ──

    def search_memory(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
        layer: MemoryLayerFilter = "all",
    ) -> list[str]:
        """跨项目级与用户级记忆检索匹配块。"""
        records = self.search_memory_records(
            query,
            limit=limit,
            scope=scope,
            layer=layer,
        )
        return [record.block for record in records]

    def search_memory_records(
        self,
        query: str,
        limit: int = 3,
        scope: str | None = None,
        layer: MemoryLayerFilter = "all",
        *,
        source: str = "api",
        track_usage: bool = True,
    ) -> list[MemoryRecord]:
        """跨层级执行 BM25 检索并返回带来源的记录。"""
        started_at = time.perf_counter()
        records = self.read_memory_records(layer=layer)
        blocks = [record.block for record in records]
        if not blocks or not query.strip() or limit <= 0:
            if source == "tool":
                self._emit_trace(
                    MemoryTraceEvent(
                        type="tool_searched",
                        latency_ms=self._elapsed_ms(started_at),
                        source=source,
                    )
                )
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
            if adjusted >= self.min_retrieval_score and self._passes_confidence_gate(
                record
            ):
                ranked.append(
                    MemoryRecord(
                        block=record.block,
                        title=record.title,
                        fields=record.fields,
                        memory_id=record.memory_id,
                        memory_type=record.memory_type,
                        scope=record.scope,
                        source_session=record.source_session,
                        related_files=record.related_files,
                        related_symbols=record.related_symbols,
                        created_at=record.created_at,
                        modified_at=record.modified_at,
                        confidence_value=record.confidence_value,
                        status=record.status,
                        validity=record.validity,
                        supersedes=record.supersedes,
                        evidence=record.evidence,
                        score=round(adjusted, 6),
                        layer=record.layer,
                    )
                )
        ranked.sort(key=lambda r: (-r.score, r.title))
        limited = ranked[:limit]
        elapsed_ms = self._elapsed_ms(started_at)
        if track_usage and limited:
            self._touch_lru(limited)
        for record in limited:
            self._emit_trace(
                MemoryTraceEvent(
                    type="retrieved",
                    memory_id=self._memory_id(record.layer, record.title),
                    layer=record.layer,
                    title=record.title,
                    score=record.score,
                    latency_ms=elapsed_ms,
                    source=source,
                )
            )
            if track_usage:
                self._emit_trace(
                    MemoryTraceEvent(
                        type="used",
                        memory_id=record.memory_id,
                        layer=record.layer,
                        title=record.title,
                        score=record.score,
                        source=source,
                    )
                )
        if source == "tool":
            self._emit_trace(
                MemoryTraceEvent(
                    type="tool_searched",
                    latency_ms=elapsed_ms,
                    source=source,
                )
            )
        return limited

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
        return self._quality_rejection_reason(block, existing_records) is None

    def _quality_rejection_reason(
        self,
        block: str,
        existing_records: list[MemoryRecord] | None = None,
    ) -> str | None:
        if not self._content_quality_check(block):
            return "content_quality_failed"
        if existing_records and len(existing_records) > 0:
            if self._is_duplicate(block, existing_records):
                return "duplicate_block"
        return None

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
        memory_type: MemoryType | None = None,
        status: str | None = None,
        validity: str | None = None,
        supersedes: Sequence[str] = (),
        evidence: Sequence[MemoryEvidence] = (),
        layer: MemoryLayer = "project",
    ) -> bool:
        """校验并写入指定记忆层级。"""
        title = extract_title(block)
        self._emit_trace(
            MemoryTraceEvent(
                type="candidate_created",
                memory_id=self._memory_id(layer, title) if title else None,
                layer=layer,
                title=title or None,
                source=source,
            )
        )
        if not self.validate_memory_block(block):
            self._emit_trace(
                MemoryTraceEvent(
                    type="rejected",
                    memory_id=self._memory_id(layer, title) if title else None,
                    layer=layer,
                    title=title or None,
                    rejection_reason="schema_validation_failed",
                    source=source,
                )
            )
            self._archive_block(block, layer)
            return False

        block = with_metadata(
            block,
            layer=layer,
            source=source,
            scope=scope,
            confidence=confidence,
            memory_type=memory_type,
            status=status,
            validity=validity,
            supersedes=tuple(supersedes),
            evidence=tuple(evidence),
        )
        existing_records = self.read_memory_records(layer=layer)
        new_title = extract_title(block)

        if new_title and existing_records:
            merged_block = self._merge_with_existing(block, new_title, existing_records)
            if merged_block is not None:
                if not self._content_quality_check(merged_block):
                    self._emit_trace(
                        MemoryTraceEvent(
                            type="rejected",
                            memory_id=self._memory_id(layer, new_title),
                            layer=layer,
                            title=new_title,
                            rejection_reason="merged_quality_gate_failed",
                            source=source,
                        )
                    )
                    self._archive_block(merged_block, layer)
                    return False
                self._emit_trace(
                    MemoryTraceEvent(
                        type="superseded",
                        memory_id=self._memory_id(layer, new_title),
                        layer=layer,
                        title=new_title,
                        source=source,
                    )
                )
                self._replace_block_by_title(new_title, merged_block, layer)
                self._enforce_lru(layer)
                self._emit_trace(
                    MemoryTraceEvent(
                        type="accepted",
                        memory_id=self._memory_id(layer, new_title),
                        layer=layer,
                        title=new_title,
                        source=source,
                    )
                )
                return True

        rejection_reason = self._quality_rejection_reason(block, existing_records)
        if rejection_reason is not None:
            self._emit_trace(
                MemoryTraceEvent(
                    type="rejected",
                    memory_id=self._memory_id(layer, new_title) if new_title else None,
                    layer=layer,
                    title=new_title or None,
                    rejection_reason=rejection_reason,
                    source=source,
                )
            )
            self._archive_block(block, layer)
            return False

        memory_file = self._memory_file(layer)
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        with memory_file.open("a", encoding="utf-8") as file:
            file.write("\n" + block.strip() + "\n")
        self._enforce_lru(layer)
        self._emit_trace(
            MemoryTraceEvent(
                type="accepted",
                memory_id=self._memory_id(layer, new_title) if new_title else None,
                layer=layer,
                title=new_title or None,
                source=source,
            )
        )
        return True

    def _replace_block_by_title(
        self,
        title: str,
        new_block: str,
        layer: MemoryLayer,
    ) -> None:
        """按标题替换指定层级中的现有记忆块。"""
        blocks = self.read_memory_blocks(layer=layer)
        updated: list[str] = []
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
        self._memory_file(layer).write_text("".join(updated), encoding="utf-8")

    def migrate_legacy_records(
        self,
        layer: MemoryLayerFilter = "all",
    ) -> int:
        """一次性补齐缺失的 memory_id / type / status / validity 元数据。"""
        updated_count = 0
        for current_layer in self._selected_layers(layer):
            blocks = self.read_memory_blocks(layer=current_layer)
            rewritten: list[str] = []
            changed = False
            for block in blocks:
                normalized = with_metadata(
                    block,
                    layer=current_layer,
                    source=None,
                    scope=None,
                    confidence=None,
                    status="active",
                    validity="unknown",
                ).strip() + "\n"
                rewritten.append(normalized)
                if normalized != block:
                    changed = True
                    updated_count += 1
            if changed:
                self._write_blocks(rewritten, current_layer)
        return updated_count

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
        """从压缩摘要中提取结构化块并写入项目级记忆。"""
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
                self.add_memory_block(block, source="consolidation", layer="project")

    def _is_memory_attempt(self, block: str) -> bool:
        has_field = any(
            f in block for f in ["Context/Query", "Solution", "Files", "Takeaways"]
        )
        return block.strip().startswith("## ") and (
            has_field or "incident" in block.lower()
        )

    def _archive_block(self, block: str, layer: MemoryLayer) -> None:
        """将无效或淘汰块归档到对应层级。"""
        archive_dir = self._archive_dir(layer)
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        archive_file = archive_dir / f"corrupt_{timestamp}.md"
        archive_file.write_text(block, encoding="utf-8")

    # ── LRU 遗忘策略 ──

    def _touch_lru(self, records: Sequence[MemoryRecord]) -> None:
        lru = self._read_lru()
        now = time.time()
        for record in records:
            lru[self._lru_key(record.layer, record.memory_id)] = now
        self._write_lru(lru)

    def _enforce_lru(self, layer: MemoryLayer) -> None:
        if self.max_blocks <= 0:
            return
        records = self.read_memory_records(layer=layer)
        blocks = [record.block for record in records]
        record_keys = {self._lru_key(layer, record.memory_id) for record in records}

        lru = self._read_lru()
        other_layer_lru = {
            key: timestamp
            for key, timestamp in lru.items()
            if not key.startswith(f"{layer}:")
        }
        layer_lru = {
            key: timestamp for key, timestamp in lru.items() if key in record_keys
        }
        cleaned = other_layer_lru | layer_lru
        if cleaned != lru:
            self._write_lru(cleaned)
            lru = cleaned

        if len(blocks) <= self.max_blocks:
            return
        now = time.time()
        for record in records:
            key = self._lru_key(layer, record.memory_id)
            if key not in lru:
                lru[key] = now

        sorted_keys = sorted(
            (key for key in lru if key.startswith(f"{layer}:")),
            key=lambda key: lru.get(key, 0.0),
        )
        keys_to_evict = set(sorted_keys[: len(blocks) - self.max_blocks])

        kept_blocks: list[str] = []
        for record in records:
            key = self._lru_key(layer, record.memory_id)
            if key in keys_to_evict:
                self._emit_trace(
                    MemoryTraceEvent(
                        type="forgotten",
                        memory_id=record.memory_id,
                        layer=layer,
                        title=record.title,
                        source="lru",
                    )
                )
                self._archive_block(record.block, layer)
            else:
                kept_blocks.append(record.block)

        if len(kept_blocks) < len(blocks):
            self._write_blocks(kept_blocks, layer)
            for key in keys_to_evict:
                lru.pop(key, None)
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

    def _write_blocks(self, blocks: list[str], layer: MemoryLayer) -> None:
        """覆盖写入指定层级的全部记忆块。"""
        content = "\n".join(b.rstrip() for b in blocks if b.strip())
        self._memory_file(layer).write_text(content + "\n", encoding="utf-8")

    def _memory_file(self, layer: MemoryLayer) -> Path:
        """返回指定层级的 MEMORY.md 路径。"""
        if layer == "project":
            return self.memory_file
        return self.user_memory_file

    def _archive_dir(self, layer: MemoryLayer) -> Path:
        """返回指定层级的归档目录。"""
        if layer == "project":
            return self.archive_dir
        return self.user_memory_file.parent / "archive"

    def _selected_layers(
        self,
        layer: MemoryLayerFilter,
    ) -> tuple[MemoryLayer, ...]:
        """将读取过滤器转换为确定的层级顺序。"""
        if layer == "all":
            return ("project", "user")
        return (layer,)

    def _lru_key(self, layer: str, title: str) -> str:
        """生成跨层级无冲突的 LRU 键。"""
        return f"{layer}:{title}"

    def record_injected_records(self, records: Sequence[MemoryRecord]) -> None:
        """记录进入 prompt 上下文的记忆摘要。"""
        for record in records:
            self._emit_trace(
                MemoryTraceEvent(
                    type="injected",
                    memory_id=self._memory_id(record.layer, record.title),
                    layer=record.layer,
                    title=record.title,
                    score=record.score,
                    token_count=self._estimate_block_tokens(record.block),
                    source="prompt",
                )
            )

    def drain_trace_events(self) -> tuple[MemoryTraceEvent, ...]:
        """返回并清空当前进程内累积的 memory trace 事件。"""
        events = tuple(self._trace_events)
        self._trace_events.clear()
        return events

    def _emit_trace(self, event: MemoryTraceEvent) -> None:
        self._trace_events.append(event)

    def _estimate_block_tokens(self, block: str) -> int:
        from xcode.agent.compaction import estimate_tokens

        return estimate_tokens(block)

    def _memory_id(self, layer: str, title: str) -> str:
        return build_memory_id(layer=layer, title=title)

    def _elapsed_ms(self, started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 3)

    def render_prompt_packet(self, record: MemoryRecord) -> str:
        """将记忆渲染为短小、可审计的 prompt packet。"""
        lines = [
            (
                f'<record id="{escape(record.memory_id)}" '
                f'type="{escape(record.memory_type)}" '
                f'layer="{escape(record.layer)}" '
                f'score="{record.score:.3f}">'
            ),
            f"<conclusion>{escape(self._conclusion_text(record))}</conclusion>",
        ]
        if record.scope:
            lines.append(f"<scope>{escape(record.scope)}</scope>")
        evidence_summary = self._evidence_summary(record)
        if evidence_summary:
            lines.append(f"<evidence>{escape(evidence_summary)}</evidence>")
        source_summary = self._source_summary(record)
        if source_summary:
            lines.append(f"<source>{escape(source_summary)}</source>")
        lines.append("</record>")
        return "\n".join(lines)

    def render_search_result(self, record: MemoryRecord) -> str:
        """渲染显式检索结果，保留完整记录并补充结构化头部。"""
        lines = [
            (
                f"[{record.layer}] id={record.memory_id} type={record.memory_type} "
                f"score={record.score:.3f}"
            ),
            f"title: {record.title}",
        ]
        evidence_summary = self._evidence_summary(record)
        if evidence_summary:
            lines.append(f"evidence: {evidence_summary}")
        lines.append(record.block.strip())
        return "\n".join(lines)

    def _conclusion_text(self, record: MemoryRecord) -> str:
        for key in ("solution", "takeaways"):
            value = record.fields.get(key, "").strip()
            if value:
                return value
        return record.title

    def _evidence_summary(self, record: MemoryRecord) -> str:
        if record.evidence:
            return "; ".join(
                f"{item.kind}:{item.reference}" for item in record.evidence[:3]
            )
        fallback = []
        for key in ("validated", "validation", "source-session", "source"):
            value = record.fields.get(key, "").strip()
            if value:
                fallback.append(value)
        return "; ".join(fallback[:2])

    def _source_summary(self, record: MemoryRecord) -> str:
        if record.source_session:
            return f"{record.layer}:{record.source_session}"
        return record.layer

    def get_last_used_at(self, record: MemoryRecord) -> float | None:
        """返回某条记忆最近一次被检索使用的时间戳。"""
        return self._read_lru().get(self._lru_key(record.layer, record.memory_id))

    def _passes_confidence_gate(self, record: MemoryRecord) -> bool:
        if record.confidence_value is None:
            return True
        return record.confidence_value >= self.min_confidence
