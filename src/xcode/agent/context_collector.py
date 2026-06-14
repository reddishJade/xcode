"""上下文收集器模块——可插拔的 ContextBlock 来源注册与管理。

提供 ContextCollector 协议和 ContextCollectorRegistry，
用于从多个来源（技能、active diff、notes 等）收集结构化上下文块，
然后注入 ContextAssembler。

设计原则：
- 未注册 collector 时，collect() 返回空列表，不影响现有行为
- collector 按注册顺序运行，输出拼接后传递给 assembler 做优先级排序
- 单个 collector 失败时跳过（log + continue），不阻断其他 collector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from xcode.agent.context_assembly import ContextBlock, ContextBlockSource, ContextBlockTarget, ContextPriority
from xcode.agent.messages import AgentMessage
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
        """从特定来源收集上下文块。返回空列表表示无可用块。"""
        ...


# ── 收集器注册表 ──


class ContextCollectorRegistry:
    """上下文收集器注册表。

    维护有序的 collector 列表，按注册顺序依次调用。
    注册表本身可被序列化为配置项传给 AgentLoopConfig。

    错误处理（log + skip，非 fail-fast）：

    与 tool_execution._execute_one（行 185-190）、
    _provider._collect_provider_events（行 119-120）
    采用相同约定：单个 component 失败时记录日志并产生 fallback 值，
    不阻断整体流程。异常永不传播给调用方。
    """

    def __init__(self) -> None:
        self._collectors: list[ContextCollector] = []

    def register(self, collector: ContextCollector) -> None:
        """注册一个 collector，追加到调用链末尾。"""
        self._collectors.append(collector)

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        """依次调用所有注册的 collector，返回合并后的块列表。

        单个 collector 抛异常时记录日志并跳过，不阻断其他 collector。
        """
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


# ── 项目指令收集器 ──

# 大小治理常量
MANIFEST_MAX_BYTES: int = 32 * 1024  # 32KB —— 超此阈值时压缩
MANIFEST_OPENING_BYTES: int = 6 * 1024  # 6KB —— 压缩时保留开头部分

MANIFEST_KEY_SECTIONS: frozenset[str] = frozenset({
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
})

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
    """压缩超出预算的项目指令文本。

    先为压缩标记预留字节，再依次添加开头内容和关键节段，
    确保标记始终完整存在于输出中。
    """
    marker = _MANIFEST_TRUNCATED_MARKER
    marker_bytes = _utf8_size(marker)

    # 预留标记字节 + 最大段落分隔符开销（3段 × 2 分隔符 = 4 字节）
    max_separator_bytes = _utf8_size("\n\n") * 2
    reserved = marker_bytes + max_separator_bytes
    content_budget = MANIFEST_MAX_BYTES - reserved

    if content_budget <= 0:
        return _utf8_prefix(marker, MANIFEST_MAX_BYTES)

    parts: list[str] = []

    # 开头内容（压缩到自身预算或剩余预算中较小者）
    opening_budget = min(MANIFEST_OPENING_BYTES, content_budget)
    opening = _utf8_prefix(text, opening_budget).strip()
    if opening:
        parts.append(opening)
        content_budget -= _utf8_size(opening)

    # 关键节段（逐段检查是否适合剩余预算）
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

    # 压缩标记始终在末尾
    parts.append(marker)

    return "\n\n".join(parts)


def _extract_key_sections(text: str) -> list[str]:
    """提取 Markdown 中匹配 MANIFEST_KEY_SECTIONS 的 ## 节。

    每个节段用 _utf8_prefix 裁剪到 SECTION_BUDGET_BYTES 以避免单节
    占用过多预算。该裁剪由 ProjectManifestCollector 调用方自行决定
    是否使用，本函数只做初筛。
    """
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


class ProjectManifestCollector:
    """从 AGENTS.md / CLAUDE.md 收集项目指令的 collector。

    读取配置路径下的 AGENTS.md 和 CLAUDE.md 文件，产出 SYSTEM 目标的
    ContextBlock。CLAUDE.md 仅包含 @AGENTS.md 引用时不重复发出块。

    大小治理（两级策略）：
    - ≤ MANIFEST_MAX_BYTES（32KB）：内容原样保留
    - > MANIFEST_MAX_BYTES：自动压缩，保留开头和关键节段，末尾始终
      包含 <manifest-truncated> 标记

    优先级为 CRITICAL（最高优先等级）：项目指令定义 agent 在仓库中的
    行为规则。优先级仅影响裁剪顺序——高优先级的块最后被丢弃。若 base
    messages 已耗尽预算，CRITICAL 块也可能被丢弃。
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = project_root

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        root = input.project_root or self._project_root
        if root is None:
            return []

        blocks: list[ContextBlock] = []

        agents_path = root / "AGENTS.md"
        if agents_path.is_file():
            content = agents_path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                blocks.append(
                    ContextBlock(
                        source=ContextBlockSource.PROJECT_MANIFEST,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content=_prepare_manifest(content),
                    )
                )

        claude_path = root / "CLAUDE.md"
        if claude_path.is_file():
            content = claude_path.read_text(encoding="utf-8", errors="replace").strip()
            if content and not _is_agents_reference(content):
                blocks.append(
                    ContextBlock(
                        source=ContextBlockSource.PROJECT_MANIFEST,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content=_prepare_manifest(content),
                    )
                )

        return blocks


def _prepare_manifest(text: str) -> str:
    """根据两级大小策略准备清单文本。

    ≤ MANIFEST_MAX_BYTES：原样返回
    > MANIFEST_MAX_BYTES：压缩后返回
    """
    source_bytes = _utf8_size(text)
    if source_bytes <= MANIFEST_MAX_BYTES:
        return text
    return _condense_manifest(text)


_AGENTS_REF_PATTERNS = frozenset({"@AGENTS.md"})


def _is_agents_reference(content: str) -> bool:
    """判断 CLAUDE.md 内容是否仅引用 AGENTS.md。

    Claude Desktop 约定：CLAUDE.md 中单独一行 @AGENTS.md 表示
    该文件内容与 AGENTS.md 相同。此时不重复发出块。
    """
    stripped = content.strip()
    return stripped in _AGENTS_REF_PATTERNS
