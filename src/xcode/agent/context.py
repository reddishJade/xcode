"""结构化上下文块收集、组装与管理。"""

from __future__ import annotations

import logging
import json
import subprocess
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Literal, Protocol

from xcode.agent._compaction import estimate_tokens
from xcode.agent.messages import (
    AgentMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import AgentTool, materialize_json_mapping

logger = logging.getLogger(__name__)


# ── 来源枚举 ──


class ContextBlockSource(StrEnum):
    INSTRUCTION = "instruction"
    SKILL = "skill"
    ACTIVE_DIFF = "active_diff"
    NOTES = "notes"
    RECENT_VALIDATION = "recent_validation"


# ── 注入目标枚举 ──


class ContextBlockTarget(StrEnum):
    SYSTEM = "system"
    USER_CONTEXT = "user_context"


# ── 优先级枚举 ──


class ContextPriority(IntEnum):
    CRITICAL = 0
    HIGH = 10
    MEDIUM = 20
    LOW = 30
    BACKGROUND = 40


# ── 过期策略 ──


@dataclass
class ContextExpiry:
    max_turns: int = 0
    max_steps: int = 0

    @property
    def never(self) -> bool:
        return self.max_turns <= 0 and self.max_steps <= 0


# ── 上下文块 ──


@dataclass
class ContextBlock:
    source: ContextBlockSource
    priority: ContextPriority
    content: str
    target: ContextBlockTarget = ContextBlockTarget.USER_CONTEXT
    token_count: int | None = None
    expiry: ContextExpiry | None = None
    created_turn: int = 0
    created_step: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
    block_id: str = ""
    provenance: str = ""
    truncated: bool = False
    truncation_reason: str | None = None

    def get_token_count(self) -> int:
        if self.token_count is not None:
            return self.token_count
        return estimate_tokens(self.content)


# ── 收集输入 ──


@dataclass
class ContextCollectionInput:
    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)
    current_turn: int = 0
    current_step: int = 0
    project_root: Path | None = None
    state: dict[str, object] = field(default_factory=dict)


# ── 组装输入/输出 ──


@dataclass
class ContextAssemblyInput:
    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)
    context_blocks: list[ContextBlock] = field(default_factory=list)
    current_turn: int = 0
    current_step: int = 0
    token_budget: int = 0
    state: dict[str, object] = field(default_factory=dict)


@dataclass
class ContextAssemblyResult:
    messages: list[AgentMessage] = field(default_factory=list)
    blocks_used: list[ContextBlock] = field(default_factory=list)
    blocks_dropped: list[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0
    token_budget: int = 0
    budget_remaining: int = 0
    base_tokens: int = 0


# ── 收集器协议 ──


class ContextCollector(Protocol):
    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]: ...


class ContextCollectorSource(Protocol):
    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]: ...


# ── 收集器注册表 ──


class ContextCollectorRegistry:
    def __init__(self) -> None:
        self._collectors: list[ContextCollector] = []

    def register(self, collector: ContextCollector) -> None:
        self._collectors.append(collector)

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        all_blocks: list[ContextBlock] = []
        for collector in self._collectors:
            try:
                blocks = collector.collect(input)
                all_blocks.extend(blocks)
            except Exception:
                logger.exception(
                    "ContextCollector %s raised; skipping",
                    type(collector).__name__,
                )
        return all_blocks

    def __len__(self) -> int:
        return len(self._collectors)

    def __bool__(self) -> bool:
        return len(self._collectors) > 0

    def freeze(self) -> FrozenContextCollectorRegistry:
        """发布不可再注册 collector 的组合快照。"""
        return FrozenContextCollectorRegistry(tuple(self._collectors))


@dataclass(frozen=True)
class FrozenContextCollectorRegistry:
    """请求组装使用的不可变 collector generation。"""

    collectors: tuple[ContextCollector, ...] = ()

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        all_blocks: list[ContextBlock] = []
        for collector in self.collectors:
            try:
                all_blocks.extend(collector.collect(input))
            except Exception:
                logger.exception(
                    "ContextCollector %s raised; skipping",
                    type(collector).__name__,
                )
        return all_blocks

    def __len__(self) -> int:
        return len(self.collectors)

    def __bool__(self) -> bool:
        return bool(self.collectors)


# ── 组装器协议 ──


class ContextAssembler(Protocol):
    def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult: ...


# ── 预算裁剪 ──


def trim_to_budget(
    blocks: list[ContextBlock],
    budget: int,
    base_tokens: int,
) -> tuple[list[ContextBlock], list[ContextBlock]]:
    # Python 的 sorted 是稳定排序；只按优先级排序可以保留 collector 的原始顺序。
    sorted_blocks = sorted(blocks, key=lambda b: b.priority)

    if budget <= 0:
        return sorted_blocks, []

    used: list[ContextBlock] = []
    dropped: list[ContextBlock] = []
    remaining = budget - base_tokens

    if remaining <= 0:
        return [], sorted_blocks

    for block in sorted_blocks:
        tokens = block.get_token_count()
        if tokens <= remaining:
            used.append(block)
            remaining -= tokens
        else:
            dropped.append(block)

    return used, dropped


# ── 默认组装器 ──


class DefaultContextAssembler:
    def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult:
        messages = list(input.messages)
        base_tokens = _estimate_base_tokens(input, messages)
        budget = input.token_budget

        if not input.context_blocks:
            return ContextAssemblyResult(
                messages=messages,
                total_tokens=base_tokens,
                token_budget=budget,
                budget_remaining=budget - base_tokens if budget > 0 else 0,
                base_tokens=base_tokens,
            )

        valid_blocks: list[ContextBlock] = []
        dropped: list[ContextBlock] = []
        for block in input.context_blocks:
            if _is_expired(block, input.current_turn, input.current_step):
                dropped.append(block)
            else:
                valid_blocks.append(block)

        used_blocks: list[ContextBlock]
        budget_dropped: list[ContextBlock]
        used_blocks, budget_dropped = trim_to_budget(valid_blocks, budget, base_tokens)
        dropped.extend(budget_dropped)

        if used_blocks:
            system_blocks = [
                b for b in used_blocks if b.target == ContextBlockTarget.SYSTEM
            ]
            user_blocks = [
                b for b in used_blocks if b.target != ContextBlockTarget.SYSTEM
            ]

            insert_idx = 0
            for i, m in enumerate(messages):
                role = getattr(m, "role", "")
                if role == "system":
                    insert_idx = i + 1
                else:
                    break

            if system_blocks:
                system_messages = [
                    SystemMessage(content=b.content) for b in system_blocks
                ]
                messages[insert_idx:insert_idx] = system_messages
                insert_idx += len(system_messages)

            if user_blocks:
                user_messages = [
                    UserMessage(content=_block_to_text(b)) for b in user_blocks
                ]
                messages[insert_idx:insert_idx] = user_messages

        final_total = _estimate_base_tokens(input, messages)

        return ContextAssemblyResult(
            messages=messages,
            blocks_used=used_blocks,
            blocks_dropped=dropped,
            total_tokens=final_total,
            token_budget=budget,
            budget_remaining=max(0, budget - final_total) if budget > 0 else 0,
            base_tokens=base_tokens,
        )


# ── 共享辅助函数 ──


def _is_expired(block: ContextBlock, turn: int, step: int) -> bool:
    if block.expiry is None:
        return False
    if block.expiry.never:
        return False
    if (
        block.expiry.max_turns > 0
        and turn - block.created_turn >= block.expiry.max_turns
    ):
        return True
    return bool(
        block.expiry.max_steps > 0
        and step - block.created_step >= block.expiry.max_steps
    )


def _block_to_text(block: ContextBlock) -> str:
    source_tag = f"[{block.source.value}]"
    if block.metadata:
        meta_str = " ".join(f"{k}={v}" for k, v in block.metadata.items())
        return f"{source_tag} ({meta_str})\n{block.content}"
    return f"{source_tag}\n{block.content}"


def _estimate_messages_tokens(messages: list[AgentMessage]) -> int:
    total = 0
    for msg in messages:
        if isinstance(msg, (CompactionSummaryMessage, BranchSummaryMessage)):
            total += estimate_tokens(msg.summary)
        else:
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += estimate_tokens(raw)
    return total


def _estimate_base_tokens(
    input: ContextAssemblyInput,
    messages: list[AgentMessage],
) -> int:
    """估算不含结构化上下文块的完整请求基线。"""
    total = _estimate_messages_tokens(messages)
    if input.system_prompt:
        total += estimate_tokens(input.system_prompt)
    total += _estimate_tools_tokens(input.tools)
    return total


def _estimate_tools_tokens(tools: list[AgentTool]) -> int:
    total = 0
    for tool in tools:
        description = str(getattr(tool, "description", ""))
        examples = getattr(tool, "examples", [])
        if examples:
            example_lines = ["\n", "Examples:"]
            for example in examples:
                example_lines.append(
                    f"  - {example.get('name', '')}: "
                    f"input={json.dumps(example.get('input', {}), ensure_ascii=False)}, "
                    f'output="{example.get("output", "")}"'
                )
            description += "\n".join(example_lines)
        payload = {
            "name": str(getattr(tool, "name", "")),
            "description": description,
            "parameters": materialize_json_mapping(getattr(tool, "parameters", {})),
        }
        builtin = getattr(tool, "builtin", None)
        if isinstance(builtin, dict):
            payload["builtin"] = builtin
        total += estimate_tokens(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        )
    return total


def _apply_size_budget(content: str, max_bytes: int, marker: str) -> str:
    if not content:
        return ""
    if _utf8_size(content) <= max_bytes:
        return content
    marker_bytes = _utf8_size(marker)
    budget = max_bytes - marker_bytes
    if budget <= 0:
        return ""
    return _utf8_prefix(content, budget) + marker


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _utf8_prefix(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")


# ── 内置收集器：指令 ──

_INSTRUCTION_PRIORITY_MAP: dict[str, ContextPriority] = {
    "critical": ContextPriority.CRITICAL,
    "high": ContextPriority.HIGH,
    "medium": ContextPriority.MEDIUM,
    "low": ContextPriority.LOW,
}


@dataclass
class InstructionSource:
    type: Literal["file", "inline"]
    path: str | None = None
    content: str | None = None
    priority: ContextPriority = ContextPriority.CRITICAL


MANIFEST_MAX_BYTES: int = 32 * 1024


class InstructionCollector:
    def __init__(
        self,
        sources: tuple[dict, ...] = (),
        project_root: Path | None = None,
    ) -> None:
        self._project_root = project_root
        self._sources: list[InstructionSource] = []
        for entry in sources:
            typ = entry["type"]
            priority_str = entry.get("priority", "critical")
            priority = _INSTRUCTION_PRIORITY_MAP[priority_str.lower()]
            if typ == "file":
                self._sources.append(
                    InstructionSource(
                        type="file",
                        path=entry["path"],
                        priority=priority,
                    )
                )
            else:
                self._sources.append(
                    InstructionSource(
                        type="inline",
                        content=entry["content"],
                        priority=priority,
                    )
                )

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        root = input.project_root or self._project_root
        if root is None:
            return []

        blocks: list[ContextBlock] = []
        configured_paths: set[Path] = set()
        remaining_bytes = MANIFEST_MAX_BYTES

        for source in self._sources:
            if remaining_bytes <= 0:
                break
            if source.type == "file":
                assert source.path is not None
                path = root / source.path
                resolved = path.resolve()
                if resolved in configured_paths:
                    continue
                configured_paths.add(resolved)
                source_blocks, consumed_bytes = self._collect_file(
                    path, source.priority, remaining_bytes
                )
            else:
                assert source.content is not None
                source_blocks, consumed_bytes = self._collect_inline(
                    source.content, source.priority, remaining_bytes
                )
            blocks.extend(source_blocks)
            remaining_bytes -= consumed_bytes

        agents_path = root / "AGENTS.md"
        if remaining_bytes > 0 and agents_path.is_file():
            if agents_path.resolve() not in configured_paths:
                source_blocks, _consumed_bytes = self._collect_file(
                    agents_path, ContextPriority.CRITICAL, remaining_bytes
                )
                blocks.extend(source_blocks)

        return blocks

    @staticmethod
    def _collect_file(
        path: Path, priority: ContextPriority, max_bytes: int
    ) -> tuple[list[ContextBlock], int]:
        try:
            data = path.read_bytes()
        except OSError:
            return [], 0
        content, consumed_bytes = _prepare_manifest_bytes(data, max_bytes, path)
        if not content:
            return [], consumed_bytes
        return [
            ContextBlock(
                source=ContextBlockSource.INSTRUCTION,
                target=ContextBlockTarget.SYSTEM,
                priority=priority,
                content=content,
                provenance=str(path),
                truncated=len(data) > max_bytes,
                truncation_reason="byte_budget" if len(data) > max_bytes else None,
            )
        ], consumed_bytes

    @staticmethod
    def _collect_inline(
        content: str, priority: ContextPriority, max_bytes: int
    ) -> tuple[list[ContextBlock], int]:
        data = content.encode("utf-8")
        prepared = _prepare_manifest(content, max_bytes, "inline instruction")
        consumed_bytes = min(len(data), max_bytes) if max_bytes > 0 else 0
        if not prepared:
            return [], consumed_bytes
        return [
            ContextBlock(
                source=ContextBlockSource.INSTRUCTION,
                target=ContextBlockTarget.SYSTEM,
                priority=priority,
                content=prepared,
                provenance="inline instruction",
                truncated=len(data) > max_bytes,
                truncation_reason="byte_budget" if len(data) > max_bytes else None,
            )
        ], consumed_bytes


def _prepare_manifest_bytes(
    data: bytes, max_bytes: int, source: Path | str
) -> tuple[str, int]:
    if max_bytes <= 0 or not data:
        return "", 0
    consumed_bytes = min(len(data), max_bytes)
    if len(data) > max_bytes:
        logger.warning(
            "project instruction exceeds remaining byte budget; truncating: %s",
            source,
        )
    return data[:consumed_bytes].decode("utf-8", errors="replace"), consumed_bytes


def _prepare_manifest(
    text: str,
    max_bytes: int = MANIFEST_MAX_BYTES,
    source: Path | str = "project instruction",
) -> str:
    prepared, _consumed_bytes = _prepare_manifest_bytes(
        text.encode("utf-8"), max_bytes, source
    )
    return prepared


# ── 内置收集器：活动 diff ──


ACTIVE_DIFF_MAX_BYTES: int = 8 * 1024
_DIFF_CMD_TIMEOUT: int = 5

_ACTIVE_DIFF_TRUNCATED_MARKER = (
    "<active-diff-truncated>Diff truncated because it exceeded the maximum "
    "allowed size. Use bash git diff for full details.</active-diff-truncated>"
)


def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            errors="replace",
            timeout=_DIFF_CMD_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


class ActiveDiffCollector:
    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        root = input.project_root or self._project_root
        if root is None:
            return []

        stat_unstaged = _run_git(root, "diff", "--stat")
        stat_staged = _run_git(root, "diff", "--cached", "--stat")
        has_staged = stat_staged is not None and bool(stat_staged.strip())
        has_unstaged = stat_unstaged is not None and bool(stat_unstaged.strip())

        if not has_staged and not has_unstaged:
            return []

        stat_parts: list[str] = []
        if has_staged:
            assert stat_staged is not None
            stat_parts.append("[staged]")
            stat_parts.append(stat_staged.strip())
        if has_unstaged:
            assert stat_unstaged is not None
            if stat_parts:
                stat_parts.append("")
            stat_parts.append("[unstaged]")
            stat_parts.append(stat_unstaged.strip())
        stat_summary = "\n".join(stat_parts)

        excerpt_block = _build_diff_excerpt_block(root, has_staged, has_unstaged)
        if excerpt_block is not None:
            ideal = stat_summary + "\n\n" + excerpt_block
        else:
            ideal = stat_summary

        marker = _ACTIVE_DIFF_TRUNCATED_MARKER
        marker_bytes = _utf8_size(marker)
        ideal_bytes = _utf8_size(ideal)

        if ideal_bytes <= ACTIVE_DIFF_MAX_BYTES:
            body = ideal
        else:
            content_budget = ACTIVE_DIFF_MAX_BYTES - marker_bytes
            if content_budget <= 0:
                return []
            body = _utf8_prefix(ideal, content_budget) + marker

        if not body.strip():
            return []

        return [
            ContextBlock(
                source=ContextBlockSource.ACTIVE_DIFF,
                target=ContextBlockTarget.USER_CONTEXT,
                priority=ContextPriority.HIGH,
                content=body,
                provenance=f"git diff: {root}",
                truncated=ideal_bytes > ACTIVE_DIFF_MAX_BYTES,
                truncation_reason=(
                    "byte_budget" if ideal_bytes > ACTIVE_DIFF_MAX_BYTES else None
                ),
            )
        ]


def _build_diff_excerpt_block(
    root: Path, has_staged: bool, has_unstaged: bool
) -> str | None:
    if has_unstaged:
        raw = _run_git(root, "diff", "--unified=1", "--no-color")
    elif has_staged:
        raw = _run_git(root, "diff", "--cached", "--unified=1", "--no-color")
    else:
        return None

    if raw is None or not raw.strip():
        return None

    lines = raw.splitlines()
    excerpt: str
    if len(lines) <= 30:
        excerpt = raw.strip()
    else:
        excerpt = "\n".join(lines[:30]) + (
            f"\n[... {len(lines) - 30} diff lines omitted ...]"
        )
    return "<diff-excerpt>\n" + excerpt + "\n</diff-excerpt>"


# ── 内置收集器：最近验证/测试失败 ──


RECENT_VALIDATION_MAX_BYTES: int = 4 * 1024
_RECENT_VALIDATION_TRUNCATED_MARKER = (
    "<validation-truncated>Failure excerpt truncated. "
    "Use the original command to see full output."
    "</validation-truncated>"
)

_VALIDATION_TOOL_NAMES: frozenset[str] = frozenset({"bash", "shell"})


def _extract_tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None) or getattr(item, "content", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return str(content)


class RecentValidationCollector:
    def __init__(
        self,
        max_bytes: int = RECENT_VALIDATION_MAX_BYTES,
    ) -> None:
        self._max_bytes = max_bytes

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        for msg in reversed(input.messages):
            if not isinstance(msg, ToolResultMessage):
                continue
            if not msg.is_error:
                continue
            if msg.tool_name not in _VALIDATION_TOOL_NAMES:
                continue
            return [self._build_block(msg)]
        return []

    def _build_block(self, msg: ToolResultMessage) -> ContextBlock:
        command = msg.tool_name
        raw = _extract_tool_result_text(msg.content)
        excerpt = _apply_size_budget(
            raw,
            self._max_bytes,
            _RECENT_VALIDATION_TRUNCATED_MARKER,
        )
        if not excerpt:
            excerpt = "(error output empty)"
        body = f"Command: {command}\n{excerpt}"
        return ContextBlock(
            source=ContextBlockSource.RECENT_VALIDATION,
            target=ContextBlockTarget.USER_CONTEXT,
            priority=ContextPriority.HIGH,
            content=body,
            provenance=f"{command}:{msg.tool_call_id}",
            truncated=_utf8_size(raw) > self._max_bytes,
            truncation_reason=(
                "byte_budget" if _utf8_size(raw) > self._max_bytes else None
            ),
        )


# ── 内置收集器：笔记 ──


NOTES_MAX_BYTES: int = 4 * 1024
NOTES_MAX_FILE_BYTES: int = 64 * 1024
_NOTES_TRUNCATED_MARKER = (
    "<notes-truncated>Notes truncated. "
    "Read individual files for full content.</notes-truncated>"
)

_NOTES_ALLOWED_SUFFIXES: frozenset[str] = frozenset({".md", ".txt"})


class NotesCollector:
    def __init__(
        self,
        project_root: Path | None = None,
        max_bytes: int = NOTES_MAX_BYTES,
    ) -> None:
        self._project_root = project_root
        self._max_bytes = max_bytes

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        root = input.project_root or self._project_root
        if root is None:
            return []
        notes_dir = root / ".xcode" / "notes"
        if not notes_dir.is_dir():
            return []
        try:
            files = sorted(
                p
                for p in notes_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in _NOTES_ALLOWED_SUFFIXES
                and p.stat().st_size <= NOTES_MAX_FILE_BYTES
            )
        except Exception:
            logger.exception("NotesCollector: failed to list notes dir")
            return []

        parts: list[str] = []
        total_bytes = 0
        marker = _NOTES_TRUNCATED_MARKER

        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                logger.debug("NotesCollector: failed to read %s", f, exc_info=True)
                text = ""
            if not text:
                continue
            header = f"--- {f.name} ---"
            item = f"{header}\n{text}"
            item_bytes = _utf8_size(item) + 1
            if total_bytes + item_bytes > self._max_bytes:
                remaining = self._max_bytes - total_bytes
                if _utf8_size(marker) <= remaining:
                    parts.append(marker)
                break
            parts.append(item)
            total_bytes += item_bytes

        if not parts:
            return []
        body = "\n".join(parts)

        return [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                target=ContextBlockTarget.USER_CONTEXT,
                priority=ContextPriority.MEDIUM,
                content=body,
                provenance=str(notes_dir),
                truncated=_NOTES_TRUNCATED_MARKER in body,
                truncation_reason=(
                    "byte_budget" if _NOTES_TRUNCATED_MARKER in body else None
                ),
            )
        ]
