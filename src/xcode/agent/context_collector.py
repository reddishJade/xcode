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
from typing import Protocol

from xcode.agent.context_assembly import ContextBlock
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
