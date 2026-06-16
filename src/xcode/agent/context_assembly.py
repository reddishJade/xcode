"""上下文组装模块——结构化上下文块管理与组装。

提供 ContextBlock、ContextAssembler 等数据模型和基础组件，
用于按来源、优先级、token 预算和过期策略管理上下文窗口。

设计原则：
- 默认无行为变更：未配置 assembler 时，消息流完全不改变
- 增量接入：transform_context 继续完全正常工作
- 确定性：优先级排序、预算裁剪、过期过滤均为纯函数
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


# ── 来源枚举 ──


class ContextBlockSource(StrEnum):
    """上下文块的来源类别。"""

    INSTRUCTION = "instruction"
    SKILL = "skill"
    ACTIVE_DIFF = "active_diff"
    NOTES = "notes"
    RECENT_VALIDATION = "recent_validation"
    TASK_STATE = "task_state"


# ── 注入目标枚举 ──


class ContextBlockTarget(StrEnum):
    """上下文块的注入目标。

    SYSTEM:       块作为 SystemMessage 注入（用于项目指令等系统级上下文）。
    USER_CONTEXT: 块作为 UserMessage 注入（用于技能、计划、提示等辅助上下文）。
    """

    SYSTEM = "system"
    USER_CONTEXT = "user_context"


# ── 优先级枚举 ──


class ContextPriority(IntEnum):
    """上下文块的优先级等级。

    数值越小优先级越高。在预算紧张时低优先级块先被裁剪。
    注意：即使 CRITICAL 块也可能因整个预算被 base messages 耗尽而被丢弃。
    居中留白便于将来在现有等级之间插入新等级。
    """

    CRITICAL = 0
    HIGH = 10
    MEDIUM = 20
    LOW = 30
    BACKGROUND = 40


# ── 过期策略 ──


@dataclass
class ContextExpiry:
    """上下文块的过期策略（相对期限）。

    每个字段表示从块创建 (created_turn/created_step) 起经过的轮/步数上限。
    所有字段默认 0 表示永不过期。

    expires_after_use 暂未实现（需要跨轮状态追踪），已从数据模型中移除。
    """

    max_turns: int = 0
    """创建后最多保留 N 轮。0 表示不限。"""

    max_steps: int = 0
    """创建后最多保留 N 步。0 表示不限。"""

    @property
    def never(self) -> bool:
        """是否永不过期。"""
        return self.max_turns <= 0 and self.max_steps <= 0


# ── 上下文块 ──


@dataclass
class ContextBlock:
    """单个上下文块。

    source:     来源标识
    priority:   优先级（排序用）
    content:    文本内容
    token_count:预计算的 token 数，None 表示首次使用时估算
    expiry:     过期策略（相对期限，从 created_turn/created_step 算起）
    created_turn:  块创建时的轮数（由调用方设置，用于相对过期判断）
    created_step:  块创建时的步数（由调用方设置，用于相对过期判断）
    metadata:   附加元数据
    block_id:   可选的唯一 ID，用于去重和审计
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

    def get_token_count(self) -> int:
        """获取 token 数，未预计算时即时估算。"""
        if self.token_count is not None:
            return self.token_count
        return estimate_tokens(self.content)


# ── 组装输入/输出 ──


@dataclass
class ContextAssemblyInput:
    """上下文组装器的输入。

    system_prompt: 系统提示词
    messages:      当前消息列表
    tools:         当前可用工具
    context_blocks:待注入的上下文块（由调用方根据场景提供）
    current_turn:  当前回合数（用于过期判断）
    current_step:  当前步骤数（用于过期判断）
    token_budget:  token 预算上限，0 表示不限
    state:         扩展状态字典
    """

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
    """上下文组装器的输出。

    messages:        最终消息列表（发送给 provider）
    blocks_used:     实际使用的上下文块
    blocks_dropped:  因预算或过期被丢弃的块
    total_tokens:    总 token 数
    token_budget:    本次的 token 预算
    budget_remaining:剩余预算
    """

    messages: list[AgentMessage] = field(default_factory=list)
    blocks_used: list[ContextBlock] = field(default_factory=list)
    blocks_dropped: list[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0
    token_budget: int = 0
    budget_remaining: int = 0


# ── 组装器协议 ──


class ContextAssembler(Protocol):
    """上下文组装器协议。

    实现此协议的类型可以插入到 AgentLoopConfig.context_assembler 中。
    """

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

    算法：始终按优先级从高到低排序，依次尝试放入；块能放入则保留，
    不能放入则丢弃且**继续检查低优先级块**（greedy priority-fill）。
    同优先级内保持原始顺序。纯函数，不修改输入。

    注意：不会因为高优先级块放不下就跳过其后的低优先级块。
    这意味着当预算有限时，一个刚好超出预算的小块可能导致所有大块被丢弃。
    """
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
    """默认上下文组装器。

    - 未配置 context_blocks 时，messages 原样返回
    - 配置了 context_blocks 时，按优先级排序后注入
    - 超出 token_budget 时从低优先级开始裁剪
    - 过期的块自动排除
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

        # 过滤过期块
        valid_blocks: list[ContextBlock] = []
        dropped: list[ContextBlock] = []
        for block in input.context_blocks:
            if _is_expired(block, input.current_turn, input.current_step):
                dropped.append(block)
            else:
                valid_blocks.append(block)

        # 预算裁剪
        used_blocks: list[ContextBlock]
        budget_dropped: list[ContextBlock]
        used_blocks, budget_dropped = trim_to_budget(valid_blocks, budget, total_tokens)
        dropped.extend(budget_dropped)

        # 注入上下文块到消息列表
        # 核心系统提示词保持第一；
        # SYSTEM 目标块 -> SystemMessage（在已有系统消息之后注入）
        # USER_CONTEXT 目标块 -> UserMessage（在所有 SystemMessage 之后注入）
        if used_blocks:
            system_blocks = [
                b for b in used_blocks if b.target == ContextBlockTarget.SYSTEM
            ]
            user_blocks = [
                b for b in used_blocks if b.target != ContextBlockTarget.SYSTEM
            ]

            # 找最后一个连续 SystemMessage 之后的位置
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

        # 重新计算总 token
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
    if block.expiry is None:
        return False
    if block.expiry.never:
        return False
    if (
        block.expiry.max_turns > 0
        and turn - block.created_turn >= block.expiry.max_turns
    ):
        return True
    if (
        block.expiry.max_steps > 0
        and step - block.created_step >= block.expiry.max_steps
    ):
        return True
    return False


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
