"""上下文收集器模块——可插拔的 ContextBlock 来源注册与管理。

提供 ContextCollector 协议和 ContextCollectorRegistry，
用于从多个来源收集结构化上下文块，然后注入 ContextAssembler。

本模块是 prompt 构建的两个系统之一（另一个见 prompting/builder.py）。
职责边界：
- 项目指令 → InstructionCollector（workspace_policy；请求 SYSTEM 但会被 assembler 降级）
- 活动 diff 摘要  → ActiveDiffCollector（USER_CONTEXT 目标）
- 最近验证失败   → RecentValidationCollector（USER_CONTEXT 目标）
- 任务/计划状态   → TaskStateCollector（USER_CONTEXT 目标）
- 笔记文件        → NotesCollector（USER_CONTEXT 目标）
- 技能摘要      → SkillIndexCollector（USER_CONTEXT 目标）

每个上下文来源有且仅有一个注入路径。
不属于本模块（由 prompting/builder 管理）：
agent 身份、工具纪律、工具列表、搜索策略、环境信息、CWD 快照、
git preflight、contextual retrieval、session 通知。

不要重新引入：
- instructions.py（旧版提示词注入路径）
- skill catalog injection（技能目录现由 SkillIndexCollector 注入）
- <active-tasks-graph> 或 <post-compact-metadata>（旧版压缩元数据）

设计原则：
- 未注册 collector 时，collect() 返回空列表，不影响现有行为
- collector 按注册顺序运行，输出拼接后传递给 assembler 做优先级排序
- 单个 collector 失败时跳过（log + continue），不阻断其他 collector
- workspace 文件只能细化项目工作方式，不能获得 host/system authority
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from xcode.agent.context_assembly import (
    ContextAuthority,
    ContextBlock,
    ContextBlockSource,
    ContextBlockTarget,
    ContextPriority,
    ContextProvenance,
    ContextScope,
    ContextTrust,
)
from xcode.agent.messages import AgentMessage, ToolResultMessage
from xcode.agent.protocols import AgentTool

logger = logging.getLogger(__name__)


# ── 收集输入 ──


@dataclass
class ContextCollectionInput:
    """上下文收集器的输入。

    与 ContextAssemblyInput 类似，但不包含 context_blocks 和 token_budget，
    因为收集阶段只负责产出块，不负责消费。
    """

    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)
    current_turn: int = 0
    current_step: int = 0
    project_root: Path | None = None
    state: dict[str, object] = field(default_factory=dict)


# ── 收集器协议 ──


class ContextCollector(Protocol):
    """上下文收集器协议。

    实现此协议的类型可从特定来源提取 ContextBlock 列表。
    所有 collector 在当前线程同步执行；如需异步 I/O，实现方自行管理。
    """

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        """从特定来源收集 ContextBlock 列表。返回空列表表示无可用块。"""
        ...


# ── 收集器注册表 ──


class ContextCollectorRegistry:
    """上下文收集器注册表。

    维护有序的 collector 列表，按注册顺序依次调用。
    注册表本身可被序列化为配置项传给 AgentLoopConfig。

    错误处理（log + skip，非 fail-fast）：

    与 tool_execution._execute_one、_provider._collect_provider_events
    采用相同约定：单个 component 失败时记录日志并产生 fallback 值，
    不阻断整体流程。异常永不传播给调用方。
    """

    def __init__(self) -> None:
        self._collectors: list[ContextCollector] = []

    def register(self, collector: ContextCollector) -> None:
        """注册一个 collector，追加到调用链末尾。"""
        self._collectors.append(collector)

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        """依次调用所有注册的 collector，返回合并后的块列表。"""
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


# ── 指令收集器 ──


_INSTRUCTION_PRIORITY_MAP: dict[str, ContextPriority] = {
    "critical": ContextPriority.CRITICAL,
    "high": ContextPriority.HIGH,
    "medium": ContextPriority.MEDIUM,
    "low": ContextPriority.LOW,
}


@dataclass
class InstructionSource:
    """单个指令源的描述。"""

    type: Literal["file", "inline"]
    path: str | None = None
    content: str | None = None
    priority: ContextPriority = ContextPriority.CRITICAL


class InstructionCollector:
    """从配置的指令源 + 默认 AGENTS.md 收集 workspace policy。

    该 collector 读取的所有内容都来自 workspace 或用户项目配置，不是 host
    policy。为了保持旧调用方的 target 兼容性，它仍请求 SYSTEM；但会显式标记
    `workspace_policy + workspace_untrusted`，DefaultContextAssembler 因而将其
    非旁路地降级为 USER_CONTEXT。
    """

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
        scope_key = str(root.resolve())

        # 1. 配置源（按解析路径去重，首个配置源优先）
        for index, source in enumerate(self._sources):
            if source.type == "file":
                assert source.path is not None
                path = root / source.path
                resolved = path.resolve()
                if resolved in configured_paths:
                    continue
                configured_paths.add(resolved)
                blocks.extend(self._collect_file(path, source.priority, root, scope_key))
            else:
                assert source.content is not None
                blocks.extend(
                    self._collect_inline(
                        source.content,
                        source.priority,
                        scope_key,
                        f"prompt.instructions[{index}]",
                    )
                )

        # 2. 回退 AGENTS.md
        agents_path = root / "AGENTS.md"
        if agents_path.is_file() and agents_path.resolve() not in configured_paths:
            content = agents_path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                blocks.append(
                    _workspace_instruction_block(
                        content=_prepare_manifest(content),
                        priority=ContextPriority.CRITICAL,
                        scope_key=scope_key,
                        locator="AGENTS.md",
                    )
                )

        return blocks

    @staticmethod
    def _collect_file(
        path: Path,
        priority: ContextPriority,
        root: Path,
        scope_key: str,
    ) -> list[ContextBlock]:
        if not path.is_file():
            return []
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return []
        try:
            locator = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            locator = str(path.resolve())
        return [
            _workspace_instruction_block(
                content=_prepare_manifest(content),
                priority=priority,
                scope_key=scope_key,
                locator=locator,
            )
        ]

    @staticmethod
    def _collect_inline(
        content: str,
        priority: ContextPriority,
        scope_key: str,
        locator: str,
    ) -> list[ContextBlock]:
        return [
            _workspace_instruction_block(
                content=_prepare_manifest(content),
                priority=priority,
                scope_key=scope_key,
                locator=locator,
            )
        ]


def _workspace_instruction_block(
    *,
    content: str,
    priority: ContextPriority,
    scope_key: str,
    locator: str,
) -> ContextBlock:
    """构造明确受限的 workspace instruction block。"""
    return ContextBlock(
        source=ContextBlockSource.INSTRUCTION,
        target=ContextBlockTarget.SYSTEM,
        priority=priority,
        content=content,
        authority=ContextAuthority.WORKSPACE_POLICY,
        trust=ContextTrust.WORKSPACE_UNTRUSTED,
        scope=ContextScope.REPOSITORY,
        scope_key=scope_key,
        provenance=ContextProvenance(
            origin="instruction_collector",
            locator=locator,
        ),
    )


# 大小治理常量
MANIFEST_MAX_BYTES: int = 32 * 1024  # 32KB —— 超此阈值时压缩
MANIFEST_OPENING_BYTES: int = 6 * 1024  # 6KB —— 压缩时保留开头部分

MANIFEST_KEY_SECTIONS: frozenset[str] = frozenset(
    {
        "priority",
        "conversation style",
        "python coding principles",
        "checklist",
        "project rules",
        "comments and docstrings",
        "git safety",
        "commit rules",
        "validation",
        "working rules",
        "debugging approach",
    }
)

_MANIFEST_TRUNCATED_MARKER = (
    "<manifest-truncated>Content was truncated because it exceeded the "
    "maximum allowed size. Opening context and key sections are preserved. "
    "Use read_file to see the full file before acting on omitted details."
    "</manifest-truncated>"
)


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _utf8_prefix(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")


def _condense_manifest(text: str) -> str:
    """压缩超出预算的指令文本。

    先为压缩标记预留字节，再依次添加开头内容和关键节段，
    确保标记始终完整存在于输出中。
    """
    marker = _MANIFEST_TRUNCATED_MARKER
    marker_bytes = _utf8_size(marker)

    max_separator_bytes = _utf8_size("\n\n") * 2
    reserved = marker_bytes + max_separator_bytes
    content_budget = MANIFEST_MAX_BYTES - reserved

    if content_budget <= 0:
        return _utf8_prefix(marker, MANIFEST_MAX_BYTES)

    parts: list[str] = []

    opening_budget = min(MANIFEST_OPENING_BYTES, content_budget)
    opening = _utf8_prefix(text, opening_budget).strip()
    if opening:
        parts.append(opening)
        content_budget -= _utf8_size(opening)

    sections = _extract_key_sections(text)
    if sections and content_budget > 0:
        section_header = "## Preserved Key Sections\n\n"
        header_bytes = _utf8_size(section_header)
        if content_budget > header_bytes:
            chosen: list[str] = []
            used = 0
            for i, section in enumerate(sections):
                prefix = "\n\n" if i > 0 else ""
                item = prefix + section
                item_bytes = _utf8_size(item)
                if used + item_bytes + header_bytes <= content_budget:
                    chosen.append(section)
                    used += item_bytes
            if chosen:
                section_text = section_header + "\n\n".join(chosen)
                parts.append(section_text)
                content_budget -= _utf8_size(section_text)

    parts.append(marker)

    return "\n\n".join(parts)


def _extract_key_sections(text: str) -> list[str]:
    """提取 Markdown 中匹配的 ##/### 关键节。"""
    sections: list[str] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_lines
        if not current_heading or not current_lines:
            return
        heading_key = current_heading.strip("# ").strip().lower()
        if heading_key not in MANIFEST_KEY_SECTIONS:
            return
        section = "\n".join(_drop_fenced_blocks(current_lines)).strip()
        if section:
            sections.append(section)

    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            flush()
            current_heading = line
            current_lines = [line]
        elif current_heading:
            current_lines.append(line)
    flush()
    return sections


def _drop_fenced_blocks(lines: list[str]) -> list[str]:
    result: list[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        result.append(line)
    return result


def _prepare_manifest(text: str) -> str:
    """根据两级大小策略准备指令文本。"""
    source_bytes = _utf8_size(text)
    if source_bytes <= MANIFEST_MAX_BYTES:
        return text
    return _condense_manifest(text)


# ── 活动 diff 收集器 ──

ACTIVE_DIFF_MAX_BYTES: int = 8 * 1024
_DIFF_CMD_TIMEOUT: int = 5

_ACTIVE_DIFF_TRUNCATED_MARKER = (
    "<active-diff-truncated>Diff truncated because it exceeded the maximum "
    "allowed size. Use bash git diff for full details.</active-diff-truncated>"
)


def _run_git(root: Path, *args: str) -> str | None:
    """运行 git 命令，返回 stdout 字符串或 None（失败时）。"""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_DIFF_CMD_TIMEOUT,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


class ActiveDiffCollector:
    """从当前 git 工作树收集活动修改摘要的 collector。"""

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
        ideal = stat_summary + "\n\n" + excerpt_block if excerpt_block is not None else stat_summary

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
            )
        ]


def _build_diff_excerpt_block(
    root: Path, has_staged: bool, has_unstaged: bool
) -> str | None:
    """构建 diff 摘录块（含 <diff-excerpt> 包装）。"""
    if has_unstaged:
        raw = _run_git(root, "diff", "--unified=1", "--no-color")
    elif has_staged:
        raw = _run_git(root, "diff", "--cached", "--unified=1", "--no-color")
    else:
        return None

    if raw is None or not raw.strip():
        return None

    lines = raw.splitlines()
    if len(lines) <= 30:
        excerpt = raw.strip()
    else:
        excerpt = "\n".join(lines[:30]) + f"\n[... {len(lines) - 30} diff lines omitted ...]"
    return "<diff-excerpt>\n" + excerpt + "\n</diff-excerpt>"


# ── 共享大小预算辅助函数 ──


def _apply_size_budget(content: str, max_bytes: int, marker: str) -> str:
    """将内容限制在 max_bytes 字节内，超出时截断并追加完整标记。"""
    if not content:
        return ""
    if _utf8_size(content) <= max_bytes:
        return content
    marker_bytes = _utf8_size(marker)
    budget = max_bytes - marker_bytes
    if budget <= 0:
        return ""
    return _utf8_prefix(content, budget) + marker


# ── 最近验证/测试失败收集器 ──

RECENT_VALIDATION_MAX_BYTES: int = 4 * 1024
_RECENT_VALIDATION_TRUNCATED_MARKER = (
    "<validation-truncated>Failure excerpt truncated. "
    "Use the original command to see full output."
    "</validation-truncated>"
)
_VALIDATION_TOOL_NAMES: frozenset[str] = frozenset({"bash", "shell"})


def _extract_tool_result_text(content: object) -> str:
    """从工具结果内容块中提取纯文本。"""
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
    """收集最近失败的验证/测试命令。"""

    def __init__(self, max_bytes: int = RECENT_VALIDATION_MAX_BYTES) -> None:
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
        )


# ── 任务状态收集器 ──

TASK_STATE_MAX_BYTES: int = 4 * 1024
_TASK_STATE_TRUNCATED_MARKER = (
    "<task-state-truncated>Task state truncated. "
    "Use list_tasks for complete state.</task-state-truncated>"
)


class TaskStateCollector:
    """收集当前任务/计划状态。"""

    def __init__(
        self,
        provider: Callable[[], str] | None = None,
        max_bytes: int = TASK_STATE_MAX_BYTES,
    ) -> None:
        self._provider = provider
        self._max_bytes = max_bytes

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        if self._provider is None:
            return []
        try:
            state_text = self._provider()
        except Exception:
            logger.exception("TaskStateCollector provider raised")
            return []
        if not state_text.strip():
            return []
        body = _apply_size_budget(
            state_text,
            self._max_bytes,
            _TASK_STATE_TRUNCATED_MARKER,
        )
        if not body:
            return []
        return [
            ContextBlock(
                source=ContextBlockSource.TASK_STATE,
                target=ContextBlockTarget.USER_CONTEXT,
                priority=ContextPriority.HIGH,
                content=body,
            )
        ]


# ── 笔记收集器 ──

NOTES_MAX_BYTES: int = 4 * 1024
NOTES_MAX_FILE_BYTES: int = 64 * 1024
_NOTES_TRUNCATED_MARKER = (
    "<notes-truncated>Notes truncated. "
    "Read individual files for full content.</notes-truncated>"
)
_NOTES_ALLOWED_SUFFIXES: frozenset[str] = frozenset({".md", ".txt"})


class NotesCollector:
    """从 .local/notes/ 收集笔记文件。"""

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
        notes_dir = root / ".local" / "notes"
        if not notes_dir.is_dir():
            return []
        try:
            files = sorted(
                path
                for path in notes_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in _NOTES_ALLOWED_SUFFIXES
                and path.stat().st_size <= NOTES_MAX_FILE_BYTES
            )
        except Exception:
            logger.exception("NotesCollector: failed to list notes dir")
            return []

        parts: list[str] = []
        total_bytes = 0
        marker = _NOTES_TRUNCATED_MARKER
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            if not text:
                continue
            header = f"--- {path.name} ---"
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
            )
        ]


# ── 技能收集器 ──
# 摘要注入在此完成；完整正文通过 load_skill 工具懒加载。
# SkillIndexCollector 定义在 xcode.harness.agent_skills 中。
