"""结构化上下文块的信任边界、预算组装与 provider 前规范化。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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


class ContextBlockSource(StrEnum):
    """上下文块的来源类别。"""

    INSTRUCTION = "instruction"
    WORKSPACE_INSTRUCTION = "workspace_instruction"
    SKILL = "skill"
    ACTIVE_DIFF = "active_diff"
    NOTES = "notes"
    RECENT_VALIDATION = "recent_validation"
    TASK_STATE = "task_state"
    MEMORY = "memory"


class ContextBlockTarget(StrEnum):
    """上下文块请求的注入目标。

    SYSTEM 不是权限授予；真正的 system authority 由 authority 与 trust 的
    组合决定。无资格请求会在 provider 前规范化为 USER_CONTEXT。
    """

    SYSTEM = "system"
    USER_CONTEXT = "user_context"


class ContextAuthority(StrEnum):
    """上下文影响 Agent 行为时声明的权威来源。"""

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
    """可审计的上下文来源。"""

    origin: str = ""
    locator: str = ""
    evidence_ids: tuple[str, ...] = ()
    content_hash: str = ""


_DEFAULT_AUTHORITY_BY_SOURCE: dict[ContextBlockSource, ContextAuthority] = {
    ContextBlockSource.INSTRUCTION: ContextAuthority.HOST_POLICY,
    ContextBlockSource.WORKSPACE_INSTRUCTION: ContextAuthority.WORKSPACE_POLICY,
    ContextBlockSource.SKILL: ContextAuthority.WORKSPACE_POLICY,
    ContextBlockSource.ACTIVE_DIFF: ContextAuthority.OBSERVATION,
    ContextBlockSource.NOTES: ContextAuthority.OBSERVATION,
    ContextBlockSource.RECENT_VALIDATION: ContextAuthority.OBSERVATION,
    ContextBlockSource.TASK_STATE: ContextAuthority.OBSERVATION,
    ContextBlockSource.MEMORY: ContextAuthority.MEMORY,
}

_DEFAULT_TRUST_BY_SOURCE: dict[ContextBlockSource, ContextTrust] = {
    ContextBlockSource.INSTRUCTION: ContextTrust.TRUSTED_HOST,
    ContextBlockSource.WORKSPACE_INSTRUCTION: ContextTrust.WORKSPACE_UNTRUSTED,
    ContextBlockSource.SKILL: ContextTrust.WORKSPACE_UNTRUSTED,
    ContextBlockSource.ACTIVE_DIFF: ContextTrust.VERIFIED_TOOL,
    ContextBlockSource.NOTES: ContextTrust.WORKSPACE_UNTRUSTED,
    ContextBlockSource.RECENT_VALIDATION: ContextTrust.VERIFIED_TOOL,
    ContextBlockSource.TASK_STATE: ContextTrust.RUNTIME_INTERNAL,
    ContextBlockSource.MEMORY: ContextTrust.RUNTIME_INTERNAL,
}

_DEFAULT_SCOPE_BY_SOURCE: dict[ContextBlockSource, ContextScope] = {
    ContextBlockSource.INSTRUCTION: ContextScope.SESSION,
    ContextBlockSource.WORKSPACE_INSTRUCTION: ContextScope.REPOSITORY,
    ContextBlockSource.SKILL: ContextScope.REPOSITORY,
    ContextBlockSource.ACTIVE_DIFF: ContextScope.WORKTREE,
    ContextBlockSource.NOTES: ContextScope.REPOSITORY,
    ContextBlockSource.RECENT_VALIDATION: ContextScope.WORKTREE,
    ContextBlockSource.TASK_STATE: ContextScope.SESSION,
    ContextBlockSource.MEMORY: ContextScope.SESSION,
}


class ContextPriority(IntEnum):
    """上下文块的优先级等级。数值越小优先级越高。"""

    CRITICAL = 0
    HIGH = 10
    MEDIUM = 20
    LOW = 30
    BACKGROUND = 40


@dataclass
class ContextExpiry:
    """上下文块的相对过期策略。0 表示不限。"""

    max_turns: int = 0
    max_steps: int = 0

    @property
    def never(self) -> bool:
        return self.max_turns <= 0 and self.max_steps <= 0


@dataclass
class ContextBlock:
    """带来源、权限边界与可审计 provenance 的上下文块。"""

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

    @property
    def resolved_authority(self) -> ContextAuthority:
        assert self.authority is not None
        return self.authority

    @property
    def resolved_trust(self) -> ContextTrust:
        assert self.trust is not None
        return self.trust

    @property
    def resolved_scope(self) -> ContextScope:
        assert self.scope is not None
        return self.scope

    def get_token_count(self) -> int:
        if self.token_count is not None:
            return self.token_count
        return estimate_tokens(self.content)


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


class ContextAssembler(Protocol):
    def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult:
        """组装并返回实际发送给 provider 的消息。"""
        ...


def _is_system_eligible(block: ContextBlock) -> bool:
    """只有可信 host policy 可成为 SystemMessage。"""
    return (
        block.target is ContextBlockTarget.SYSTEM
        and block.resolved_authority is ContextAuthority.HOST_POLICY
        and block.resolved_trust is ContextTrust.TRUSTED_HOST
    )


def normalize_context_blocks(blocks: list[ContextBlock]) -> list[ContextBlock]:
    """在任何 assembler 运行前，消除不可信内容的 system 注入请求。

    该函数不丢弃上下文，而是将无资格 SYSTEM 请求转为 USER_CONTEXT，并在
    metadata 中留下不可伪装的 `system_target=demoted` 审计标记。
    """
    normalized: list[ContextBlock] = []
    for block in blocks:
        if block.target is not ContextBlockTarget.SYSTEM or _is_system_eligible(block):
            normalized.append(block)
            continue
        metadata = dict(block.metadata)
        metadata["system_target"] = "demoted"
        normalized.append(
            replace(
                block,
                target=ContextBlockTarget.USER_CONTEXT,
                metadata=metadata,
            )
        )
    return normalized


def sanitize_assembled_messages(
    *,
    original_messages: list[AgentMessage],
    assembled_messages: list[AgentMessage],
    trusted_system_blocks: list[ContextBlock],
) -> list[AgentMessage]:
    """在 provider 边界复核 custom assembler 新增的 SystemMessage。

    允许原始消息中已有的 system 内容，以及本轮已被判定为 host policy 的
    system block。任何其他新增 system message 都被转为审计化 USER_CONTEXT。
    """
    allowed: dict[str, int] = {}
    for message in original_messages:
        if getattr(message, "role", "") == "system":
            content = str(getattr(message, "content", ""))
            allowed[content] = allowed.get(content, 0) + 1
    for block in trusted_system_blocks:
        allowed[block.content] = allowed.get(block.content, 0) + 1

    sanitized: list[AgentMessage] = []
    for message in assembled_messages:
        if getattr(message, "role", "") != "system":
            sanitized.append(message)
            continue
        content = str(getattr(message, "content", ""))
        remaining = allowed.get(content, 0)
        if remaining > 0:
            allowed[content] = remaining - 1
            sanitized.append(message)
            continue
        sanitized.append(
            UserMessage(
                content=(
                    "[context source=assembler authority=untrusted "
                    "system_target=demoted]\n"
                    f"{content}"
                )
            )
        )
    return sanitized


def trim_to_budget(
    blocks: list[ContextBlock],
    budget: int,
    base_tokens: int,
) -> tuple[list[ContextBlock], list[ContextBlock]]:
    """按优先级和 token 预算贪心保留上下文块。"""
    sorted_blocks = sorted(blocks, key=lambda block: (block.priority, block.get_token_count()))
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


class DefaultContextAssembler:
    """默认 assembler：只允许已经通过 trust contract 的 system blocks。"""

    def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult:
        messages = list(input.messages)
        total_tokens = _estimate_messages_tokens(messages)
        budget = input.token_budget
        normalized_blocks = normalize_context_blocks(input.context_blocks)
        if not normalized_blocks:
            return ContextAssemblyResult(
                messages=messages,
                total_tokens=total_tokens,
                token_budget=budget,
                budget_remaining=budget - total_tokens if budget > 0 else 0,
            )

        valid_blocks: list[ContextBlock] = []
        dropped: list[ContextBlock] = []
        for block in normalized_blocks:
            if _is_expired(block, input.current_turn, input.current_step):
                dropped.append(block)
            else:
                valid_blocks.append(block)

        used_blocks, budget_dropped = trim_to_budget(valid_blocks, budget, total_tokens)
        dropped.extend(budget_dropped)
        if used_blocks:
            system_blocks = [block for block in used_blocks if _is_system_eligible(block)]
            user_blocks = [block for block in used_blocks if not _is_system_eligible(block)]
            insert_idx = _last_system_insert_index(messages)
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


def _last_system_insert_index(messages: list[AgentMessage]) -> int:
    insert_idx = 0
    for index, message in enumerate(messages):
        if getattr(message, "role", "") == "system":
            insert_idx = index + 1
        else:
            break
    return insert_idx


def _is_expired(block: ContextBlock, turn: int, step: int) -> bool:
    if block.expiry is None or block.expiry.never:
        return False
    if block.expiry.max_turns > 0 and turn - block.created_turn >= block.expiry.max_turns:
        return True
    return block.expiry.max_steps > 0 and step - block.created_step >= block.expiry.max_steps


def _block_to_text(block: ContextBlock) -> str:
    source_tag = f"[{block.source.value}]"
    fields = [
        f"authority={block.resolved_authority.value}",
        f"trust={block.resolved_trust.value}",
        f"scope={block.resolved_scope.value}",
    ]
    if block.metadata.get("system_target") == "demoted":
        fields.append("system_target=demoted")
    if block.scope_key:
        fields.append(f"scope_key={block.scope_key}")
    if block.provenance.locator:
        fields.append(f"locator={block.provenance.locator}")
    if block.metadata:
        fields.extend(
            f"{key}={value}"
            for key, value in block.metadata.items()
            if key != "system_target"
        )
    return f"{source_tag} ({' '.join(fields)})\n{block.content}"


def _estimate_messages_tokens(messages: list[AgentMessage]) -> int:
    total = 0
    for message in messages:
        if isinstance(message, (CompactionSummaryMessage, BranchSummaryMessage)):
            total += estimate_tokens(message.summary)
        else:
            raw = message.content if isinstance(message.content, str) else str(message.content)
            total += estimate_tokens(raw)
    return total
