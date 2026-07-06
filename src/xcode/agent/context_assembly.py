"""上下文组装模块——结构化上下文块管理与组装。

提供 ContextBlock、ContextAssembler 等数据模型和基础组件，
用于按来源、权限边界、信任级别、作用域、token 预算和过期策略管理上下文窗口。

设计原则：
- 默认无行为变更：未配置 assembler 时，消息流完全不改变
- 增量接入：transform_context 继续完全正常工作
- 确定性：优先级排序、预算裁剪、过期过滤均为纯函数
- 可审计：每个上下文块都带有 authority、trust、scope 和 provenance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Protocol

from xcode.agent.compaction import estimate_tokens
from xcode.agent.messages import (
    AgentMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    SystemMessage,
    UserMessage,
)
from xcode.agent.protocols import AgentTool


# ── 来源、权限与信任模型 ──


class ContextBlockSource(StrEnum):
    """上下文块的来源类别。"""

    INSTRUCTION = "instruction"
    SKILL = "skill"
    ACTIVE_DIFF = "active_diff"
    NOTES = "notes"
    RECENT_VALIDATION = "recent_validation"
    TASK_STATE = "task_state"
    MEMORY = "memory"


class ContextBlockTarget(StrEnum):
    """上下文块的注入目标。

    SYSTEM 仅保留给 host/core policy。来自 workspace、memory、工具结果或
    外部内容的块不得在迁移完成后以 SYSTEM authority 注入。
    """

    SYSTEM = "system"
    USER_CONTEXT = "user_context"


class ContextAuthority(StrEnum):
    """上下文可以影响 Agent 的权限来源。

    authority 描述“它是谁”，不代表其内容一定为真；真实性由 trust 描述。
    """

    HOST_POLICY = "host_policy"
    USER_REQUEST = "user_request"
    WORKSPACE_POLICY = "workspace_policy"
    MEMORY = "memory"
    OBSERVATION = "observation"


class ContextTrust(StrEnum):
    """上下文内容的信任等级。"""

    TRUSTED_HOST = "trusted_host"
    TRUSTED_USER = "trusted_user"
    VERIFIED_TOOL = "verified_tool"
    RUNTIME_INTERNAL = "runtime_internal"
    WORKSPACE_UNTRUSTED = "workspace_untrusted"
    EXTERNAL_UNTRUSTED = "external_untrusted"


class ContextScope(StrEnum):
    """上下文块可复用的最大作用域。"""

    SESSION = "session"
    WORKTREE = "worktree"
    REPOSITORY = "repository"
    USER_GLOBAL = "user_global"


@dataclass(frozen=True, slots=True)
class ContextProvenance:
    """可审计的上下文来源。

    origin 是产生该块的系统组件；locator 指向文件、会话、工具调用或其他
    可复查位置。evidence_ids 保留未来 Memory Evidence Ledger 的稳定入口。
    """

    origin: str = ""
    locator: str = ""
    evidence_ids: tuple[str, ...] = ()
    content_hash: str = ""


_DEFAULT_AUTHORITY_BY_SOURCE: dict[ContextBlockSource, ContextAuthority] = {
    ContextBlockSource.INSTRUCTION: ContextAuthority.WORKSPACE_POLICY,
    ContextBlockSource.SKILL: ContextAuthority.WORKSPACE_POLICY,
    ContextBlockSource.ACTIVE_DIFF: ContextAuthority.OBSERVATION,
    ContextBlockSource.NOTES: ContextAuthority.OBSERVATION,
    ContextBlockSource.RECENT_VALIDATION: ContextAuthority.OBSERVATION,
    ContextBlockSource.TASK_STATE: ContextAuthority.OBSERVATION,
    ContextBlockSource.MEMORY: ContextAuthority.MEMORY,
}

_DEFAULT_TRUST_BY_SOURCE: dict[ContextBlockSource, ContextTrust] = {
    ContextBlockSource.INSTRUCTION: ContextTrust.WORKSPACE_UNTRUSTED,
    ContextBlockSource.SKILL: ContextTrust.WORKSPACE_UNTRUSTED,
    ContextBlockSource.ACTIVE_DIFF: ContextTrust.VERIFIED_TOOL,
    ContextBlockSource.NOTES: ContextTrust.WORKSPACE_UNTRUSTED,
    ContextBlockSource.RECENT_VALIDATION: ContextTrust.VERIFIED_TOOL,
    ContextBlockSource.TASK_STATE: ContextTrust.RUNTIME_INTERNAL,
    ContextBlockSource.MEMORY: ContextTrust.RUNTIME_INTERNAL,
}

_DEFAULT_SCOPE_BY_SOURCE: dict[ContextBlockSource, ContextScope] = {
    ContextBlockSource.INSTRUCTION: ContextScope.REPOSITORY,
    ContextBlockSource.SKILL: ContextScope.REPOSITORY,
    ContextBlockSource.ACTIVE_DIFF: ContextScope.WORKTREE,
    ContextBlockSource.NOTES: ContextScope.REPOSITORY,
    ContextBlockSource.RECENT_VALIDATION: ContextScope.WORKTREE,
    ContextBlockSource.TASK_STATE: ContextScope.SESSION,
    ContextBlockSource.MEMORY: ContextScope.SESSION,
}


# ── 优先级与过期策略 ──


class ContextPriority(IntEnum):
    """上下文块的优先级等级。

    数值越小优先级越高。在预算紧张时低优先级块先被裁剪。
    注意：即使 CRITICAL 块也可能因整个预算被 base messages 耗尽而被丢弃。
    """

    CRITICAL = 0
    HIGH = 10
    MEDIUM = 20
    LOW = 30
    BACKGROUND = 40


@dataclass
class ContextExpiry:
    """上下文块的过期策略（相对期限）。"""

    max_turns: int = 0
    max_steps: int = 0

    @property
    def never(self) -> bool:
        """是否永不过期。"""
        return self.max_turns <= 0 and self.max_steps <= 0


# ── 上下文块 ──


@dataclass
class ContextBlock:
    """单个上下文块。

    authority、trust、scope 与 provenance 是上下文信任契约的一部分。当前
    Phase 0a 仅完成类型化和来源默认值；后续迁移会由 assembler 强制执行
    SYSTEM 注入边界与 Memory 的 USER_CONTEXT-only 约束。
    """

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
    authority: ContextAuthority | None = None
    trust: ContextTrust | None = None
    scope: ContextScope | None = None
    scope_key: str = ""
    provenance: ContextProvenance = field(default_factory=ContextProvenance)

    def __post_init__(self) -> None:
        """补齐基于来源的安全默认值，保留调用方的显式声明。"""
        if self.authority is None:
            self.authority = _DEFAULT_AUTHORITY_BY_SOURCE[self.source]
        if self.trust is None:
            self.trust = _DEFAULT_TRUST_BY_SOURCE[self.source]
        if self.scope is None:
            self.scope = _DEFAULT_SCOPE_BY_SOURCE[self.source]
        if not self.provenance.origin:
            self.provenance = ContextProvenance(
                origin=self.source.value,
                locator=self.provenance.locator,
                evidence_ids=self.provenance.evidence_ids,
                content_hash=self.provenance.content_hash,
            )

    def get_token_count(self) -> int:
        """获取 token 数，未预计算时即时估算。"""
        if self.token_count is not None:
            return self.token_count
        return estimate_tokens(self.content)


# ── 组装输入/输出 ──


@dataclass
class ContextAssemblyInput:
    """上下文组装器的输入。"""

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
    """上下文组装器的输出。"""

    messages: list[AgentMessage] = field(default_factory=list)
    blocks_used: list[ContextBlock] = field(default_factory=list)
    blocks_dropped: list[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0
    token_budget: int = 0
    budget_remaining: int = 0


class ContextAssembler(Protocol):
    """上下文组装器协议。"""

    def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult:
        """组装上下文，返回结构化结果。"""
        ...


# ── 预算裁剪 ──


def trim_to_budget(
    blocks: list[ContextBlock],
    budget: int,
    base_tokens: int,
) -> tuple[list[ContextBlock], list[ContextBlock]]:
    """按预算裁剪块列表，返回 (used, dropped)。

    按优先级从高到低依次尝试放入；同优先级内按 token 数升序。不能放入
    的块被丢弃，但继续检查后续较小块以最大化实际可用信息量。
    """
    sorted_blocks = sorted(blocks, key=lambda b: (b.priority, b.get_token_count()))

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
    """默认上下文组装器。

    - 未配置 context_blocks 时，messages 原样返回。
    - 配置了 context_blocks 时，按优先级排序后注入。
    - 超出 token_budget 时按贪心策略裁剪。
    - 过期块自动排除。
    """

    def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult:
        messages = list(input.messages)
        total_tokens = _estimate_messages_tokens(messages)
        budget = input.token_budget

        if not input.context_blocks:
            return ContextAssemblyResult(
                messages=messages,
                total_tokens=total_tokens,
                token_budget=budget,
                budget_remaining=budget - total_tokens if budget > 0 else 0,
            )

        valid_blocks: list[ContextBlock] = []
        dropped: list[ContextBlock] = []
        for block in input.context_blocks:
            if _is_expired(block, input.current_turn, input.current_step):
                dropped.append(block)
            else:
                valid_blocks.append(block)

        used_blocks, budget_dropped = trim_to_budget(valid_blocks, budget, total_tokens)
        dropped.extend(budget_dropped)

        if used_blocks:
            system_blocks = [
                block for block in used_blocks if block.target == ContextBlockTarget.SYSTEM
            ]
            user_blocks = [
                block for block in used_blocks if block.target != ContextBlockTarget.SYSTEM
            ]

            insert_idx = 0
            for index, message in enumerate(messages):
                if getattr(message, "role", "") == "system":
                    insert_idx = index + 1
                else:
                    break

            if system_blocks:
                messages[insert_idx:insert_idx] = [
                    SystemMessage(content=block.content) for block in system_blocks
                ]
                insert_idx += len(system_blocks)

            if user_blocks:
                messages[insert_idx:insert_idx] = [
                    UserMessage(content=_block_to_text(block)) for block in user_blocks
                ]

        final_total = _estimate_messages_tokens(messages)
        return ContextAssemblyResult(
            messages=messages,
            blocks_used=used_blocks,
            blocks_dropped=dropped,
            total_tokens=final_total,
            token_budget=budget,
            budget_remaining=max(0, budget - final_total) if budget > 0 else 0,
        )


# ── 辅助函数 ──


def _is_expired(block: ContextBlock, turn: int, step: int) -> bool:
    """判断块是否过期（相对期限，从 created_turn/created_step 算起）。"""
    if block.expiry is None or block.expiry.never:
        return False
    if block.expiry.max_turns > 0 and turn - block.created_turn >= block.expiry.max_turns:
        return True
    return block.expiry.max_steps > 0 and step - block.created_step >= block.expiry.max_steps


def _block_to_text(block: ContextBlock) -> str:
    """将辅助上下文渲染为可审计、不可伪装为 host policy 的用户上下文。"""
    fields = [
        f"source={block.source.value}",
        f"authority={block.authority.value}",
        f"trust={block.trust.value}",
        f"scope={block.scope.value}",
    ]
    if block.scope_key:
        fields.append(f"scope_key={block.scope_key}")
    if block.provenance.locator:
        fields.append(f"locator={block.provenance.locator}")
    if block.metadata:
        fields.extend(f"{key}={value}" for key, value in block.metadata.items())
    return f"[context {' '.join(fields)}]\n{block.content}"


def _estimate_messages_tokens(messages: list[AgentMessage]) -> int:
    total = 0
    for message in messages:
        if isinstance(message, (CompactionSummaryMessage, BranchSummaryMessage)):
            total += estimate_tokens(message.summary)
        else:
            raw = message.content if isinstance(message.content, str) else str(message.content)
            total += estimate_tokens(raw)
    return total
