"""上下文组装模块测试。

测试 ContextBlock、DefaultContextAssembler 以及 context_assembler hook
在 agent 循环中的行为。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any
from xcode.agent.agent_loop import run_agent_loop
from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.context_assembly import (
    ContextAssemblyInput,
    ContextAssemblyResult,
    ContextBlock,
    ContextBlockSource,
    ContextBlockTarget,
    ContextExpiry,
    ContextPriority,
    DefaultContextAssembler,
    trim_to_budget,
)
from xcode.agent.context_collector import (
    ActiveDiffCollector,
    ContextCollectionInput,
    ContextCollectorRegistry,
    InstructionCollector,
    NotesCollector,
    RecentValidationCollector,
    TaskStateCollector,
)
from xcode.agent.messages import (
    CompactionSummaryMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.message_converter import convert_to_llm
from xcode.agent.protocols import AgentToolResult, ToolExecutionMode
from xcode.agent.types import TextContent
from xcode.ai.events import Message, TextDelta, ToolCall, ToolCallEvent
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.agent_skills import (
    SkillIndexCollector,
    SkillRegistry,
    build_skill_search_dirs,
)
import pytest

# ── 辅助 Provider ──


class CaptureProvider:
    """捕获发送给 provider 的消息。"""

    def __init__(self) -> None:
        self.captured_messages: list[list[Message]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> Any:
        self.captured_messages.append(messages)
        yield TextDelta(chunk="done")


class ToolCaptureProvider:
    """带工具调用的 capture provider。"""

    def __init__(self) -> None:
        self.captured_messages: list[list[Message]] = []
        self.call_count = 0

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ) -> Any:
        self.captured_messages.append(messages)
        self.call_count += 1
        if self.call_count == 1:
            yield ToolCallEvent(
                calls=[ToolCall(id="tc-1", name="echo", input={"text": "hi"})]
            )


# ── Data Model Tests ──


class TestContextBlock:
    """测试 ContextBlock 数据模型。"""

    def test_token_count_cached(self) -> None:
        """预计算的 token_count 应直接返回。"""
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.HIGH,
            content="some content",
            token_count=42,
        )
        assert block.get_token_count() == 42

    def test_token_count_estimated(self) -> None:
        """未预计算时自动估算。"""
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.LOW,
            content="hello world",
        )
        assert block.get_token_count() > 0

    def test_default_created_values(self) -> None:
        """created_turn 和 created_step 默认为 0。"""
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.MEDIUM,
            content="test",
        )
        assert block.created_turn == 0
        assert block.created_step == 0

    def test_created_values_custom(self) -> None:
        """created_turn 和 created_step 可自定义。"""
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.HIGH,
            content="test",
            created_turn=5,
            created_step=3,
        )
        assert block.created_turn == 5
        assert block.created_step == 3


class TestContextExpiry:
    """测试 ContextExpiry 过期策略。"""

    def test_never_by_default(self) -> None:
        """默认永不过期。"""
        expiry = ContextExpiry()
        assert expiry.never

    def test_not_never_when_set(self) -> None:
        """设置任何过期条件后 never 返回 False。"""
        assert not (ContextExpiry(max_turns=5).never)
        assert not (ContextExpiry(max_steps=10).never)

    def test_max_turns_zero_means_unlimited(self) -> None:
        """max_turns=0 表示不限。"""
        assert ContextExpiry(max_turns=0).never
        assert not (ContextExpiry(max_turns=1).never)

    def test_max_steps_zero_means_unlimited(self) -> None:
        """max_steps=0 表示不限。"""
        assert ContextExpiry(max_steps=0).never
        assert not (ContextExpiry(max_steps=1).never)


class TestContextPriority:
    """测试 ContextPriority 排序语义。"""

    def test_ordering(self) -> None:
        """CRITICAL < HIGH < MEDIUM < LOW < BACKGROUND。"""
        assert ContextPriority.CRITICAL < ContextPriority.HIGH
        assert ContextPriority.HIGH < ContextPriority.MEDIUM
        assert ContextPriority.MEDIUM < ContextPriority.LOW
        assert ContextPriority.LOW < ContextPriority.BACKGROUND


# ── trim_to_budget 纯函数测试 ──


class TestTrimToBudget:
    """测试 trim_to_budget 纯函数。"""

    def test_no_budget_returns_all_sorted(self) -> None:
        """budget <= 0 时所有块按优先级排序后返回。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.BACKGROUND,
                content="bg",
            ),
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.CRITICAL,
                content="critical",
            ),
        ]
        used, dropped = trim_to_budget(blocks, 0, 0)
        assert len(used) == 2
        assert len(dropped) == 0
        assert used[0].priority == ContextPriority.CRITICAL
        assert used[1].priority == ContextPriority.BACKGROUND

    def test_budget_exceeded_drops_lowest_priority(self) -> None:
        """超出预算时丢弃扫描到的块，剩余的是高优先级块。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.HIGH,
                content="high",
                token_count=50,
            ),
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.BACKGROUND,
                content="low",
                token_count=100,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=60, base_tokens=0)
        assert len(used) == 1
        assert len(dropped) == 1
        assert used[0].priority == ContextPriority.HIGH
        assert dropped[0].priority == ContextPriority.BACKGROUND

    def test_deterministic_with_equal_priority(self) -> None:
        """同优先级顺序稳定。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.MEDIUM,
                content="first",
                token_count=30,
            ),
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.MEDIUM,
                content="second",
                token_count=30,
            ),
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.MEDIUM,
                content="third",
                token_count=30,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=50, base_tokens=0)
        assert len(used) == 1
        assert len(dropped) == 2
        assert used[0].content == "first"

    def test_base_tokens_consumes_budget_critical_dropped(self) -> None:
        """base_tokens 占满预算时即使是 CRITICAL 块也被丢弃。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.CRITICAL,
                content="important",
                token_count=10,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=10, base_tokens=10)
        assert len(used) == 0
        assert len(dropped) == 1

    def test_greedy_policy_skips_large_high_priority(self) -> None:
        """高优先级块太大时跳过，继续尝试低优先级小块的 greedy 策略。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.HIGH,
                content="too large",
                token_count=100,
            ),
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.MEDIUM,
                content="small fits",
                token_count=3,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=10, base_tokens=0)
        assert len(used) == 1
        assert len(dropped) == 1
        assert used[0].priority == ContextPriority.MEDIUM
        assert dropped[0].priority == ContextPriority.HIGH

    def test_greedy_half_fill(self) -> None:
        """多个块填充到预算上限，剩余丢弃。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.CRITICAL,
                content="a",
                token_count=10,
            ),
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.HIGH,
                content="b",
                token_count=10,
            ),
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=ContextPriority.LOW,
                content="c",
                token_count=10,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=25, base_tokens=0)
        # a(10) + b(10) = 20 fits, c(10) exceeds remaining(5) → dropped
        assert len(used) == 2
        assert len(dropped) == 1
        assert dropped[0].content == "c"


# ── DefaultContextAssembler 单元测试 ──


class TestDefaultContextAssemblerNoOp:
    """未配置 context_blocks 时行为不变。"""

    def setup_method(self, method) -> None:
        self.assembler = DefaultContextAssembler()

    def test_messages_unchanged_no_blocks(self) -> None:
        """无 blocks 时消息原样返回。"""
        msgs: list[UserMessage] = [UserMessage(content="hello")]
        result = self.assembler.assemble(ContextAssemblyInput(messages=msgs))
        assert len(result.messages) == 1
        assert result.messages[0] is msgs[0]

    def test_result_metadata_zero_when_no_blocks(self) -> None:
        """无 blocks 时 metadata 为零值。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(messages=[UserMessage(content="hi")])
        )
        assert len(result.blocks_used) == 0
        assert len(result.blocks_dropped) == 0
        assert result.total_tokens == 1


class TestDefaultContextAssemblerPriority:
    """测试优先级排序。"""

    def setup_method(self, method) -> None:
        self.assembler = DefaultContextAssembler()

    def test_blocks_ordered_by_priority(self) -> None:
        """块按优先级排序后注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.BACKGROUND,
                        content="bg",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.CRITICAL,
                        content="critical",
                    ),
                ],
            )
        )
        assert len(result.blocks_used) == 2
        assert result.blocks_used[0].priority == ContextPriority.CRITICAL
        assert result.blocks_used[1].priority == ContextPriority.BACKGROUND

    def test_deterministic_order_within_same_priority(self) -> None:
        """同优先级保持原始插入顺序。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.MEDIUM,
                        content="first",
                        block_id="a",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.MEDIUM,
                        content="second",
                        block_id="b",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.MEDIUM,
                        content="third",
                        block_id="c",
                    ),
                ],
            )
        )
        ids = [b.block_id for b in result.blocks_used]
        assert ids == ["a", "b", "c"]


class TestDefaultContextAssemblerBudget:
    """测试预算裁剪。"""

    def setup_method(self, method) -> None:
        self.assembler = DefaultContextAssembler()

    def test_budget_trimming_drops_lowest_priority(self) -> None:
        """超出预算时丢弃低优先级。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hi")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.CRITICAL,
                        content="critical content",
                        token_count=10,
                    ),
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.BACKGROUND,
                        content="background noise",
                        token_count=100,
                    ),
                ],
                token_budget=15,
            )
        )
        assert len(result.blocks_used) == 1
        assert result.blocks_used[0].priority == ContextPriority.CRITICAL
        assert len(result.blocks_dropped) == 1
        assert result.blocks_dropped[0].priority == ContextPriority.BACKGROUND

    def test_critical_dropped_when_base_exceeds_budget(self) -> None:
        """base messages 已耗尽预算时即使是 CRITICAL 块也丢弃。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="x" * 2000)],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.CRITICAL,
                        content="vital",
                        token_count=10,
                    ),
                ],
                token_budget=5,
            )
        )
        assert len(result.blocks_used) == 0
        assert len(result.blocks_dropped) == 1
        assert result.blocks_dropped[0].priority == ContextPriority.CRITICAL

    def test_greedy_fill_over_budget(self) -> None:
        """高优先级块太大时跳过，小块的次高优先级块保留。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="base")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.CRITICAL,
                        content="big critical",
                        token_count=100,
                    ),
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="small high",
                        token_count=3,
                    ),
                ],
                token_budget=10,
            )
        )
        assert len(result.blocks_used) == 1
        assert result.blocks_used[0].priority == ContextPriority.HIGH
        assert len(result.blocks_dropped) == 1
        assert result.blocks_dropped[0].priority == ContextPriority.CRITICAL

    def test_budget_trimming_deterministic(self) -> None:
        """相同输入产生相同结果。"""
        input_blocks = [
            ContextBlock(
                source=ContextBlockSource.NOTES,
                priority=p,
                content=f"block-{p}",
                token_count=10,
            )
            for p in (
                ContextPriority.HIGH,
                ContextPriority.MEDIUM,
                ContextPriority.LOW,
            )
        ]
        inp = ContextAssemblyInput(
            messages=[UserMessage(content="base")],
            context_blocks=list(input_blocks),
            token_budget=15,
        )

        r1 = self.assembler.assemble(inp)
        r2 = self.assembler.assemble(inp)

        assert len(r1.blocks_used) == len(r2.blocks_used)
        assert [b.block_id for b in r1.blocks_used] == [
            b.block_id for b in r2.blocks_used
        ]


class TestDefaultContextAssemblerExpiry:
    """测试过期过滤（相对期限）。"""

    def setup_method(self, method) -> None:
        self.assembler = DefaultContextAssembler()

    def test_expired_by_step_relative(self) -> None:
        """created_step=1, max_steps=3, current_step=4 → 4-1=3 >= 3 → 过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="valid",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.CRITICAL,
                        content="expired by step",
                        expiry=ContextExpiry(max_steps=3),
                        created_step=1,
                    ),
                ],
                current_step=4,
            )
        )
        assert len(result.blocks_used) == 1
        assert result.blocks_used[0].content == "valid"
        assert len(result.blocks_dropped) == 1
        assert result.blocks_dropped[0].content == "expired by step"

    def test_not_expired_by_step_before_limit(self) -> None:
        """created_step=1, max_steps=3, current_step=3 → 3-1=2 < 3 → 未过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="still valid",
                        expiry=ContextExpiry(max_steps=3),
                        created_step=1,
                    ),
                ],
                current_step=3,
            )
        )
        assert len(result.blocks_used) == 1

    def test_expired_by_turn_relative(self) -> None:
        """created_turn=2, max_turns=3, current_turn=5 → 5-2=3 >= 3 → 过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.MEDIUM,
                        content="still valid",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="expired by turn",
                        expiry=ContextExpiry(max_turns=3),
                        created_turn=2,
                    ),
                ],
                current_turn=5,
            )
        )
        assert len(result.blocks_used) == 1
        assert result.blocks_used[0].content == "still valid"

    def test_not_expired_by_turn_before_limit(self) -> None:
        """created_turn=2, max_turns=3, current_turn=4 → 4-2=2 < 3 → 未过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="not expired yet",
                        expiry=ContextExpiry(max_turns=3),
                        created_turn=2,
                    ),
                ],
                current_turn=4,
            )
        )
        assert len(result.blocks_used) == 1

    def test_never_expiry_not_excluded(self) -> None:
        """never 过期策略的块永不被排除。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="never expires",
                        expiry=ContextExpiry(),
                    ),
                ],
                current_step=999,
                current_turn=999,
            )
        )
        assert len(result.blocks_used) == 1

    def test_none_expiry_not_excluded(self) -> None:
        """expiry 为 None 时永不过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="no expiry set",
                        expiry=None,
                    ),
                ],
                current_step=999,
            )
        )
        assert len(result.blocks_used) == 1

    def test_expiry_with_default_created(self) -> None:
        """created_turn/created_step 默认 0，行为等价于绝对截止。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="expires at step 3 from start",
                        expiry=ContextExpiry(max_steps=3),
                    ),
                ],
                current_step=3,
            )
        )
        # created_step=0, 3-0=3 >= 3 → expired
        assert len(result.blocks_used) == 0
        assert len(result.blocks_dropped) == 1


class TestDefaultContextAssemblerMessageInjection:
    """测试消息注入位置。"""

    def setup_method(self, method) -> None:
        self.assembler = DefaultContextAssembler()

    def test_blocks_injected_after_system(self) -> None:
        """块在系统消息后注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[
                    SystemMessage(content="system prompt"),
                    UserMessage(content="user query"),
                ],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="plan info",
                    ),
                ],
            )
        )
        assert len(result.messages) == 3
        assert getattr(result.messages[0], "role", "") == "system"
        assert getattr(result.messages[1], "role", "") == "user"
        assert "plan info" in str(result.messages[1].content)
        # 验证 synthetic 消息包含 [notes] 来源标记
        assert "[notes]" in str(result.messages[1].content)

    def test_blocks_injected_at_start_no_system(self) -> None:
        """无系统消息时块在开头注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="user query")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.MEDIUM,
                        content="note content",
                    ),
                ],
            )
        )
        assert len(result.messages) == 2
        assert "note content" in str(result.messages[0].content)
        assert "[notes]" in str(result.messages[0].content)

    def test_multiple_blocks_all_injected(self) -> None:
        """多个块全部注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hi")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.LOW,
                        content=f"block-{i}",
                    )
                    for i in range(3)
                ],
            )
        )
        assert len(result.messages) == 4
        assert len(result.blocks_used) == 3


# ── Agent 循环集成测试 ──


class TestContextAssemblerAgentLoopIntegration:
    """通过 agent 循环测试 context_assembler hook。"""

    def test_no_assembler_configured_no_change(self) -> None:
        """未配置 assembler 时行为不变。"""
        provider = CaptureProvider()
        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        assert len(provider.captured_messages) == 1
        msgs = provider.captured_messages[0]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) >= 1

    def test_transform_context_without_assembler(self) -> None:
        """transform_context 在无 assembler 时正常工作。"""
        provider = CaptureProvider()
        transform_called = False

        def _transform(msgs: list, signal: object = None) -> list:
            nonlocal transform_called
            transform_called = True
            return msgs

        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            transform_context=_transform,
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        assert transform_called

    def test_assembler_and_transform_context_coexist(self) -> None:
        """assembler 和 transform_context 同时配置时都执行。"""
        provider = CaptureProvider()
        assembler = DefaultContextAssembler()
        transform_log: list[str] = []

        def _transform(msgs: list, signal: object = None) -> list:
            transform_log.append("called")
            return msgs

        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_assembler=assembler,
            transform_context=_transform,
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        assert len(transform_log) == 1

    def test_assembler_with_blocks_integration(self) -> None:
        """assembler 配置 blocks 后消息包含块内容。"""
        provider = ToolCaptureProvider()
        assembler = DefaultContextAssembler()

        # 创建一个 echo 工具使 agent 能执行工具调用
        class EchoTool:
            name = "echo"
            label = "echo"
            description = "echo back input"
            parameters = {"text": {"type": "string"}}
            execution_mode: ToolExecutionMode = "sequential"
            examples = []

            async def execute(self, tool_call_id, params, signal=None, on_update=None):
                return AgentToolResult(
                    content=[TextContent(text=params.get("text", ""))]
                )

        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_assembler=assembler,
            max_steps=2,
        )
        context = AgentContext(
            messages=[],
            tools=[EchoTool()],
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="say hi")],
                context=context,
                config=config,
                emit=lambda _e: None,
            )
        )
        assert len(provider.captured_messages) > 0

    def test_assembler_output_sent_as_provider_messages(self) -> None:
        """组装后的消息正常发送给 provider。"""
        provider = CaptureProvider()
        assembler = DefaultContextAssembler()

        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_assembler=assembler,
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        # 验证消息被 provider 接收
        assert len(provider.captured_messages) == 1
        msgs = provider.captured_messages[0]
        assert len(msgs) > 0

    def test_current_step_passed_to_assembler(self) -> None:
        """current_step 被正确传递到 ContextAssemblyInput。"""
        captured_steps: list[int] = []

        class StepCaptureAssembler:
            def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult:
                captured_steps.append(input.current_step)
                return ContextAssemblyResult(messages=list(input.messages))

        provider = CaptureProvider()
        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_assembler=StepCaptureAssembler(),
            max_steps=3,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        # max_steps=3, assembler 在每个 provider 调用前执行
        assert len(captured_steps) >= 1
        # 第一个 step 应为 1
        assert captured_steps[0] == 1


class TestContextAssemblyInputConstruction:
    """测试 ContextAssemblyInput 的 provider 侧构造。"""

    def test_default_values(self) -> None:
        """默认值不应导致错误。"""
        inp = ContextAssemblyInput()
        assert inp.current_turn == 0
        assert inp.current_step == 0
        assert inp.token_budget == 0
        assert len(inp.state) == 0
        assert len(inp.messages) == 0
        assert len(inp.tools) == 0
        assert len(inp.context_blocks) == 0

    def test_system_prompt_preserved(self) -> None:
        """system_prompt 字段传递正确。"""
        inp = ContextAssemblyInput(system_prompt="you are a helpful agent")
        assert inp.system_prompt == "you are a helpful agent"


# ── ContextCollector 注册表测试 ──


class FakeCollector:
    """测试用 fake collector，返回预设的块列表。"""

    def __init__(self, blocks: list[ContextBlock], *, name: str = "") -> None:
        self._blocks = blocks
        self._name = name

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        return list(self._blocks)


class ErrorCollector:
    """测试用 fake collector，collect 时抛出异常。"""

    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        msg = "collector error"
        raise RuntimeError(msg)


class TestContextCollectorRegistry:
    """测试 ContextCollectorRegistry 基本行为。"""

    def test_empty_registry_returns_empty(self) -> None:
        """空注册表返回空列表。"""
        registry = ContextCollectorRegistry()
        result = registry.collect(ContextCollectionInput())
        assert len(result) == 0

    def test_empty_registry_is_falsey(self) -> None:
        """空注册表 __bool__ 为 False。"""
        registry = ContextCollectorRegistry()
        assert not (registry)

    def test_registry_with_collectors_is_truthy(self) -> None:
        """有 collector 时 __bool__ 为 True。"""
        registry = ContextCollectorRegistry()
        registry.register(FakeCollector([]))
        assert registry

    def test_single_collector_returns_blocks(self) -> None:
        """单个 collector 返回的块被正确收集。"""
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.HIGH,
            content="from collector",
        )
        registry = ContextCollectorRegistry()
        registry.register(FakeCollector([block]))
        result = registry.collect(ContextCollectionInput())
        assert len(result) == 1
        assert result[0].content == "from collector"

    def test_multiple_collectors_preserve_order(self) -> None:
        """多个 collector 按注册顺序合并结果。"""
        block_a = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.HIGH,
            content="first",
            block_id="a",
        )
        block_b = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.MEDIUM,
            content="second",
            block_id="b",
        )
        block_c = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.LOW,
            content="third",
            block_id="c",
        )
        registry = ContextCollectorRegistry()
        registry.register(FakeCollector([block_a, block_b]))
        registry.register(FakeCollector([block_c]))
        result = registry.collect(ContextCollectionInput())
        assert len(result) == 3
        assert [b.block_id for b in result] == ["a", "b", "c"]

    def test_error_collector_skipped_other_still_run(self) -> None:
        """异常 collector 被跳过（log + skip），其他 collector 仍正常执行。"""
        good_block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.HIGH,
            content="good",
        )
        registry = ContextCollectorRegistry()
        registry.register(ErrorCollector())
        registry.register(FakeCollector([good_block]))
        # 异常被吞噬，不会传播到调用方
        result = registry.collect(ContextCollectionInput())
        assert len(result) == 1
        assert result[0].content == "good"

    def test_error_collector_exception_never_propagates(self) -> None:
        """collector 异常绝不传播到 collect() 的调用方。"""
        registry = ContextCollectorRegistry()
        registry.register(ErrorCollector())
        # 不需要 try/except，因为 collect 内部已捕获
        result = registry.collect(ContextCollectionInput())
        assert len(result) == 0


class TestContextCollectorWithAssembler:
    """测试 collector → assembler 集成。"""

    def test_collector_blocks_reach_assembler(self) -> None:
        """collector 产出的块被传入 assembler 的 context_blocks。"""
        captured_blocks: list[ContextBlock] = []

        class CaptureAssembler:
            def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult:
                captured_blocks.extend(input.context_blocks)
                return ContextAssemblyResult(messages=list(input.messages))

        provider = CaptureProvider()
        registry = ContextCollectorRegistry()
        registry.register(
            FakeCollector(
                [
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.CRITICAL,
                        content="collector-block",
                        block_id="c1",
                    ),
                ]
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_collectors=registry,
            context_assembler=CaptureAssembler(),
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        assert len(captured_blocks) == 1
        assert captured_blocks[0].block_id == "c1"

    def test_no_collectors_assembler_receives_empty_blocks(self) -> None:
        """未配置 collector 时 assembler 收到空 blocks。"""
        captured_blocks: list[ContextBlock] = []

        class CaptureAssembler:
            def assemble(self, input: ContextAssemblyInput) -> ContextAssemblyResult:
                captured_blocks.extend(input.context_blocks)
                return ContextAssemblyResult(messages=list(input.messages))

        provider = CaptureProvider()
        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_assembler=CaptureAssembler(),
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        assert len(captured_blocks) == 0

    def test_collector_with_default_assembler_block_injected(self) -> None:
        """collector 产出的块经 DefaultContextAssembler 注入到消息中。"""
        provider = CaptureProvider()
        registry = ContextCollectorRegistry()
        registry.register(
            FakeCollector(
                [
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        priority=ContextPriority.HIGH,
                        content="plan step 1",
                    ),
                ]
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_collectors=registry,
            context_assembler=DefaultContextAssembler(),
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        # 验证块内容出现在发送给 provider 的消息中
        assert len(provider.captured_messages) == 1
        msgs = provider.captured_messages[0]
        combined = " ".join(str(m.get("content", "") or "") for m in msgs)
        assert "plan step 1" in combined
        assert "[notes]" in combined

    def test_collectors_skipped_when_no_assembler(self) -> None:
        """未配置 assembler 时 collector 不执行。"""
        call_count: list[int] = []

        class CountingCollector:
            def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
                call_count.append(1)
                return []

        provider = CaptureProvider()
        registry = ContextCollectorRegistry()
        registry.register(CountingCollector())

        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_collectors=registry,
            context_assembler=None,
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        assert len(call_count) == 0

    def test_transform_context_runs_after_collectors(self) -> None:
        """collector + assembler 配置下 transform_context 仍执行。"""
        transform_log: list[str] = []

        def _transform(msgs: list, signal: object = None) -> list:
            transform_log.append("called")
            return msgs

        provider = CaptureProvider()
        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_collectors=ContextCollectorRegistry(),
            context_assembler=DefaultContextAssembler(),
            transform_context=_transform,
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(messages=[]),
                config=config,
                emit=lambda _e: None,
            )
        )
        assert len(transform_log) == 1

    def test_active_diff_as_user_context_after_system(self) -> None:
        """ACTIVE_DIFF 块作为 UserMessage 注入，位于 SystemMessage 之后。"""
        provider = CaptureProvider()
        registry = ContextCollectorRegistry()
        registry.register(
            FakeCollector(
                [
                    ContextBlock(
                        source=ContextBlockSource.ACTIVE_DIFF,
                        target=ContextBlockTarget.USER_CONTEXT,
                        priority=ContextPriority.HIGH,
                        content="[unstaged]\n M src/main.py",
                    ),
                ]
            )
        )

        config = AgentLoopConfig(
            provider=provider,
            convert_to_llm=convert_to_llm,
            context_collectors=registry,
            context_assembler=DefaultContextAssembler(),
            max_steps=1,
        )
        import asyncio

        asyncio.run(
            run_agent_loop(
                prompts=[UserMessage(content="hello")],
                context=AgentContext(
                    messages=[
                        SystemMessage(content="identity prompt"),
                    ]
                ),
                config=config,
                emit=lambda _e: None,
            )
        )
        assert len(provider.captured_messages) == 1
        msgs = provider.captured_messages[0]
        roles = [m["role"] for m in msgs]
        # 顺序: system(identity) + system(manifest) + user(diff) + user(question)
        # 但这里没有 manifest collector，只有 diff collector
        assert roles == ["system", "user", "user"]
        # diff 内容中包含 source 标记
        combined = " ".join(str(m.get("content", "") or "") for m in msgs)
        assert "[active_diff]" in combined
        assert "M src/main.py" in combined


# ── ContextBlockTarget 测试 ──


class TestContextBlockTarget:
    """测试 ContextBlockTarget 枚举。"""

    def test_default_target_is_user_context(self) -> None:
        """ContextBlock 默认 target 为 USER_CONTEXT。"""
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.HIGH,
            content="test",
        )
        assert block.target == ContextBlockTarget.USER_CONTEXT

    def test_system_target_explicit(self) -> None:
        """SYSTEM target 可显式设置。"""
        block = ContextBlock(
            source=ContextBlockSource.INSTRUCTION,
            target=ContextBlockTarget.SYSTEM,
            priority=ContextPriority.CRITICAL,
            content="system instructions",
        )
        assert block.target == ContextBlockTarget.SYSTEM

    def test_target_enum_values(self) -> None:
        """枚举值正确。"""
        assert ContextBlockTarget.SYSTEM.value == "system"
        assert ContextBlockTarget.USER_CONTEXT.value == "user_context"


# ── SYSTEM 块组装测试 ──


class TestDefaultContextAssemblerSystemBlocks:
    """测试 SYSTEM 目标块的组装行为。"""

    def setup_method(self, method) -> None:
        self.assembler = DefaultContextAssembler()

    def test_system_block_becomes_system_message(self) -> None:
        """SYSTEM 块被注入为 SystemMessage。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="project rules",
                    ),
                ],
            )
        )
        assert len(result.messages) == 2
        assert getattr(result.messages[0], "role", "") == "system"
        assert result.messages[0].content == "project rules"
        assert result.blocks_used[0].content == "project rules"

    def test_system_block_injected_after_existing_system(self) -> None:
        """SYSTEM 块在已有 SystemMessage 之后注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[
                    SystemMessage(content="identity prompt"),
                    UserMessage(content="question"),
                ],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="project rules",
                    ),
                ],
            )
        )
        assert len(result.messages) == 3
        roles = [getattr(m, "role", "") for m in result.messages]
        assert roles == ["system", "system", "user"]
        assert result.messages[0].content == "identity prompt"
        assert result.messages[1].content == "project rules"

    def test_mixed_targets_separate_injection(self) -> None:
        """SYSTEM 和 USER_CONTEXT 块分别注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="question")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="project rules",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.NOTES,
                        target=ContextBlockTarget.USER_CONTEXT,
                        priority=ContextPriority.HIGH,
                        content="plan step",
                    ),
                ],
            )
        )
        assert len(result.messages) == 3
        roles = [getattr(m, "role", "") for m in result.messages]
        assert roles == ["system", "user", "user"]
        assert result.messages[0].content == "project rules"
        assert "plan step" in str(result.messages[1].content)
        assert "[notes]" in str(result.messages[1].content)

    def test_multiple_system_blocks_all_as_system_messages(self) -> None:
        """多个 SYSTEM 块全部作为 SystemMessage 注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="question")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="rules part 1",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="rules part 2",
                    ),
                ],
            )
        )
        assert len(result.messages) == 3
        roles = [getattr(m, "role", "") for m in result.messages]
        assert roles == ["system", "system", "user"]

    def test_compaction_summary_does_not_affect_injection_position(self) -> None:
        """CompactionSummaryMessage 不会中断 system block scanning。

        ProjectManifest SYSTEM 块应在 prompting system message 之后、
        压缩历史（compaction_summary）之前注入。
        """
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[
                    SystemMessage(content="identity prompt"),
                    CompactionSummaryMessage(summary="previous turns"),
                    UserMessage(content="current question"),
                ],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.INSTRUCTION,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="project rules",
                    ),
                ],
            )
        )
        roles = [getattr(m, "role", "") for m in result.messages]
        assert roles == ["system", "system", "compaction_summary", "user"]
        assert result.messages[0].content == "identity prompt"
        assert result.messages[1].content == "project rules"


# ── InstructionCollector 测试 ──


class TestInstructionCollector:
    """测试 InstructionCollector（空配置时回退到 AGENTS.md）。"""

    def test_no_files_returns_empty(self) -> None:
        """项目根目录无 AGENTS.md 时返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 0

    def test_empty_config_fallback_agents(self) -> None:
        """空配置时 AGENTS.md 回退。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Use tests.", encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].source == ContextBlockSource.INSTRUCTION
            assert blocks[0].target == ContextBlockTarget.SYSTEM
            assert blocks[0].priority == ContextPriority.CRITICAL
            assert blocks[0].content == "Use tests."

    def test_empty_config_ignores_unconfigured_instruction_file(self) -> None:
        """空配置时仅回退 AGENTS.md。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Real content.", encoding="utf-8")
            (root / "NOTES.md").write_text("Extra content.", encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].content == "Real content."

    def test_empty_config_without_agents_returns_empty(self) -> None:
        """没有 AGENTS.md 时不收集额外文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "NOTES.md").write_text("Standalone notes.", encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 0

    def test_agents_md_via_input_project_root(self) -> None:
        """project_root 可通过 ContextCollectionInput 传入。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
            collector = InstructionCollector()
            inp = ContextCollectionInput(project_root=root)
            blocks = collector.collect(inp)
            assert len(blocks) == 1
            assert "Project rules." in blocks[0].content

    def test_file_source(self) -> None:
        """配置的 file 源被正确收集。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CUSTOM.md").write_text("Custom instructions.", encoding="utf-8")
            sources = ({"type": "file", "path": "CUSTOM.md"},)
            collector = InstructionCollector(sources=sources, project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].source == ContextBlockSource.INSTRUCTION
            assert blocks[0].content == "Custom instructions."

    def test_file_source_priority(self) -> None:
        """配置的 file 源的 priority 被正确映射。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CUSTOM.md").write_text("Low priority.", encoding="utf-8")
            sources = ({"type": "file", "path": "CUSTOM.md", "priority": "low"},)
            collector = InstructionCollector(sources=sources, project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert blocks[0].priority == ContextPriority.LOW

    def test_inline_source(self) -> None:
        """配置的 inline 源被正确收集。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = ({"type": "inline", "content": "No deps without approval."},)
            collector = InstructionCollector(sources=sources, project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].source == ContextBlockSource.INSTRUCTION
            assert blocks[0].content == "No deps without approval."

    def test_inline_source_priority(self) -> None:
        """配置的 inline 源的 priority 被正确映射。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = (
                {"type": "inline", "content": "High priority.", "priority": "high"},
            )
            collector = InstructionCollector(sources=sources, project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert blocks[0].priority == ContextPriority.HIGH

    def test_configured_source_wins_dedup(self) -> None:
        """AGENTS.md 配置为 file 源时，回退 AGENTS.md 被跳过。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Fallback.", encoding="utf-8")
            sources = ({"type": "file", "path": "AGENTS.md", "priority": "medium"},)
            collector = InstructionCollector(sources=sources, project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].priority == ContextPriority.MEDIUM
            assert blocks[0].content == "Fallback."

    def test_duplicate_configured_file_source_first_wins(self) -> None:
        """同一文件出现在两个配置源时，首个源优先。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "RULES.md").write_text("First priority.", encoding="utf-8")
            sources = (
                {"type": "file", "path": "RULES.md", "priority": "critical"},
                {"type": "file", "path": "RULES.md", "priority": "low"},
            )
            collector = InstructionCollector(sources=sources, project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].priority == ContextPriority.CRITICAL
            assert blocks[0].content == "First priority."

    def test_configured_plus_fallback(self) -> None:
        """配置源 + AGENTS.md 回退均收集，忽略未配置文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Fallback AGENTS.", encoding="utf-8")
            (root / "NOTES.md").write_text("Fallback notes.", encoding="utf-8")
            sources = ({"type": "inline", "content": "Inline instruction."},)
            collector = InstructionCollector(sources=sources, project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 2
            contents = [b.content for b in blocks]
            assert "Inline instruction." in contents
            assert "Fallback AGENTS." in contents
            assert "Fallback notes." not in contents

    def test_inline_source_size_governed(self) -> None:
        """inline 内容 > 32KB 时被压缩。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "x" * 50000
            sources = ({"type": "inline", "content": "# Guide\n\n" + body},)
            collector = InstructionCollector(sources=sources, project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks[0].content.encode("utf-8")) <= 32 * 1024
            assert "<manifest-truncated>" in blocks[0].content


class TestInstructionCollectorSizeGovernance:
    """测试指令大小治理（与 InstructionCollector 配合）。"""

    def test_small_content_passes_unchanged(self) -> None:
        """小内容（≤24KB）保持原样。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "Small instruction file."
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].content == content

    def test_medium_content_passes_unchanged(self) -> None:
        """中等内容（size ≤ 32KB）保持原样。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "# Medium\n\n" + ("x" * (28 * 1024))
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].content == content

    def test_oversized_content_is_condensed(self) -> None:
        """超大内容（>32KB）被压缩到 32KB 以内。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "background\n" * 5000
            content = "# Opening\n\n" + body + "\n\n## Validation\n\nRun tests.\n"
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert len(blocks[0].content.encode("utf-8")) <= 32 * 1024

    def test_condensed_output_has_truncation_marker(self) -> None:
        """压缩后的输出包含 <manifest-truncated> 标记。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "background\n" * 5000
            content = "# Opening\n\n" + body
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert "<manifest-truncated>" in blocks[0].content

    def test_condensed_keeps_key_section(self) -> None:
        """压缩后保留匹配的 ## 关键节段。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "#" * 40000
            content = (
                "# Guide\n\n"
                + body
                + "\n\n## Validation\n\nRun targeted validation for modified files.\n"
            )
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert "Run targeted validation" in blocks[0].content

    def test_condensing_deterministic(self) -> None:
        """相同输入产生相同压缩结果。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "#" * 40000
            content = "# Guide\n\n" + body + "\n\n## Priority\n\nFirst.\n"
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            first = collector.collect(ContextCollectionInput())
            second = collector.collect(ContextCollectionInput())
            assert first[0].content == second[0].content

    def test_non_key_section_dropped_when_oversized(self) -> None:
        """超大内容时非关键节段被丢弃。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "#" * 40000
            content = "# Guide\n\n" + body + "\n\n## Random Notes\n\nNot important.\n"
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert "Random Notes" not in blocks[0].content

    def test_system_target_preserved(self) -> None:
        """指令块是 SYSTEM 目标。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "#" * 40000
            content = "# Guide\n\n" + body + "\n\n## Priority\n\nFirst.\n"
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].target == ContextBlockTarget.SYSTEM
            assert blocks[0].source == ContextBlockSource.INSTRUCTION

    def test_marker_fully_present_not_truncated(self) -> None:
        """压缩标记始终完整包含在输出中，不被截断。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "#" * 50000
            content = "# Guide\n\n" + body
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert "<manifest-truncated>" in blocks[0].content
            assert "</manifest-truncated>" in blocks[0].content
            assert blocks[0].content.strip().endswith("</manifest-truncated>")

    def test_output_strictly_bounded(self) -> None:
        """压缩后输出严格 ≤ MANIFEST_MAX_BYTES。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "#" * 100000
            content = (
                "# Guide\n\n" + body + "\n\n## Validation\n\nRun tests.\n"
                "\n\n## Priority\n\nFirst.\n"
                "\n\n## Git Safety\n\nNever.\n"
            )
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks[0].content.encode("utf-8")) <= 32 * 1024

    def test_marker_survives_when_content_overflows(self) -> None:
        """超阈值内容中标记仍然完整存在。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "x" * 50000
            content = body + "\n\n## Validation\n\nRun tests.\n"
            (root / "AGENTS.md").write_text(content, encoding="utf-8")
            collector = InstructionCollector(sources=(), project_root=root)
            blocks = collector.collect(ContextCollectionInput())
            output = blocks[0].content
            assert "<manifest-truncated>" in output
            assert "</manifest-truncated>" in output
            assert output.strip().endswith("</manifest-truncated>")
            assert len(output.encode("utf-8")) <= 32 * 1024


# ── 配置验证测试 ──


class TestInstructionSourceValidation:
    """测试 prompt.instructions 配置验证。"""

    def test_invalid_entry_raises(self) -> None:
        """非 dict 条目抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {"prompt": {"instructions": ["bad"]}}
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)

    def test_invalid_type_raises(self) -> None:
        """不支持的 type 抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {"prompt": {"instructions": [{"type": "xyz"}]}}
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)

    def test_absolute_path_posix_raises(self) -> None:
        """POSIX 绝对路径抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {"prompt": {"instructions": [{"type": "file", "path": "/etc/passwd"}]}}
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)

    def test_absolute_path_windows_raises(self) -> None:
        """Windows 绝对路径抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {"prompt": {"instructions": [{"type": "file", "path": "C:\\foo"}]}}
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)

    def test_home_relative_path_raises(self) -> None:
        """~ 开头的路径抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {"prompt": {"instructions": [{"type": "file", "path": "~/foo"}]}}
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)

    def test_traversal_path_raises(self) -> None:
        """../foo 抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {"prompt": {"instructions": [{"type": "file", "path": "../foo"}]}}
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)

    def test_traversal_path_segment_raises(self) -> None:
        """路径中包含 .. 段抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {"prompt": {"instructions": [{"type": "file", "path": "foo/../../bar"}]}}
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)

    def test_inline_empty_content_raises(self) -> None:
        """inline 内容为空抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {"prompt": {"instructions": [{"type": "inline", "content": ""}]}}
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)

    def test_invalid_priority_raises(self) -> None:
        """不支持的 priority 抛出 ValueError。"""
        from xcode.harness.config import _validate_instruction_sources

        raw = {
            "prompt": {
                "instructions": [
                    {"type": "inline", "content": "ok", "priority": "urgent"}
                ]
            }
        }
        with pytest.raises(ValueError) as exc_info:
            _validate_instruction_sources(raw)
        assert "prompt.instructions[0]" in str(exc_info.value)


# ── ActiveDiffCollector 测试 ──


class TestActiveDiffCollector:
    """测试 ActiveDiffCollector。"""

    def test_no_git_repo_returns_empty(self) -> None:
        """非 git 仓库返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 0

    def test_clean_repo_returns_empty(self) -> None:
        """干净的 git 仓库返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init(root)
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            _git(root, "commit", "-m", "initial")
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 0

    def test_modified_file_produces_block(self) -> None:
        """修改过的文件产生一个 ACTIVE_DIFF 块。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init(root)
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            _git(root, "commit", "-m", "initial")
            (root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].source == ContextBlockSource.ACTIVE_DIFF
            assert "a.txt" in blocks[0].content

    def test_block_target_is_user_context(self) -> None:
        """块目标为 USER_CONTEXT。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init(root)
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            _git(root, "commit", "-m", "initial")
            (root / "a.txt").write_text("changed\n", encoding="utf-8")
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].target == ContextBlockTarget.USER_CONTEXT

    def test_priority_is_high(self) -> None:
        """优先级为 HIGH。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init(root)
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            _git(root, "commit", "-m", "initial")
            (root / "a.txt").write_text("changed\n", encoding="utf-8")
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert blocks[0].priority == ContextPriority.HIGH

    def test_small_diff_has_no_truncation_marker(self) -> None:
        """小 diff 不包含截断标记。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init(root)
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            _git(root, "commit", "-m", "initial")
            (root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert "active-diff-truncated" not in blocks[0].content

    def test_oversized_diff_has_full_marker(self) -> None:
        """超大 diff 包含完整截断标记。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init(root)
            # 创建多个文件使统计信息和摘录都足够大
            for fname in ("a.txt", "b.txt", "c.txt"):
                (root / fname).write_text("line\n" * 10, encoding="utf-8")
            _git(root, "add", "-A")
            _git(root, "commit", "-m", "initial")
            long_line = "changed" + "X" * 2000 + "\n"
            for fname in ("a.txt", "b.txt", "c.txt"):
                (root / fname).write_text(long_line * 5000, encoding="utf-8")
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert "<active-diff-truncated>" in blocks[0].content
            assert "</active-diff-truncated>" in blocks[0].content
            assert len(blocks[0].content.encode("utf-8")) <= 8 * 1024

    def test_staged_only_change_produces_block(self) -> None:
        """仅 staged 的修改产生 ACTIVE_DIFF 块。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init(root)
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            _git(root, "add", "-A")
            _git(root, "commit", "-m", "initial")
            (root / "b.txt").write_text("new\n", encoding="utf-8")
            _git(root, "add", "b.txt")
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            # b.txt is staged, a.txt is not modified — diff shows [staged]
            assert "[staged]" in blocks[0].content
            assert "b.txt" in blocks[0].content

    def test_staged_and_unstaged_includes_both(self) -> None:
        """staged 和 unstaged 同时存在时都包含。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git_init(root)
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            (root / "b.txt").write_text("base\n", encoding="utf-8")
            _git(root, "add", "-A")
            _git(root, "commit", "-m", "initial")
            # staged change
            (root / "a.txt").write_text("staged\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            # unstaged change
            (root / "b.txt").write_text("unstaged\n", encoding="utf-8")
            collector = ActiveDiffCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert "[staged]" in blocks[0].content
            assert "[unstaged]" in blocks[0].content
            assert "a.txt" in blocks[0].content
            assert "b.txt" in blocks[0].content

    def test_git_failure_returns_empty(self) -> None:
        """git 失败时返回空列表，不抛出异常。"""
        collector = ActiveDiffCollector(Path("/nonexistent"))
        blocks = collector.collect(ContextCollectionInput())
        assert len(blocks) == 0


# ── RecentValidationCollector 测试 ──


class TestRecentValidationCollector:
    """测试 RecentValidationCollector。"""

    def test_no_messages_returns_empty(self) -> None:
        """无消息时返回空列表。"""
        collector = RecentValidationCollector()
        blocks = collector.collect(ContextCollectionInput(messages=[]))
        assert len(blocks) == 0

    def test_no_error_messages_returns_empty(self) -> None:
        """无错误消息时返回空列表。"""
        msg = ToolResultMessage(tool_name="bash", content="success", is_error=False)
        collector = RecentValidationCollector()
        blocks = collector.collect(ContextCollectionInput(messages=[msg]))
        assert len(blocks) == 0

    def test_non_validation_error_ignored(self) -> None:
        """非 bash/shell 工具的错误被忽略。"""
        msg = ToolResultMessage(
            tool_name="read_file", content="not found", is_error=True
        )
        collector = RecentValidationCollector()
        blocks = collector.collect(ContextCollectionInput(messages=[msg]))
        assert len(blocks) == 0

    def test_bash_error_emits_block(self) -> None:
        """bash 工具错误发出验证失败块。"""
        msg = ToolResultMessage(
            tool_name="bash",
            content="Error: command not found",
            is_error=True,
        )
        collector = RecentValidationCollector()
        blocks = collector.collect(ContextCollectionInput(messages=[msg]))
        assert len(blocks) == 1
        assert blocks[0].source == ContextBlockSource.RECENT_VALIDATION
        assert blocks[0].target == ContextBlockTarget.USER_CONTEXT
        assert blocks[0].priority == ContextPriority.HIGH
        assert "Command: bash" in blocks[0].content
        assert "Error: command not found" in blocks[0].content

    def test_last_error_is_used(self) -> None:
        """只使用最近的错误。"""
        msgs = [
            ToolResultMessage(tool_name="bash", content="old error", is_error=True),
            ToolResultMessage(tool_name="bash", content="", is_error=False),
            ToolResultMessage(tool_name="bash", content="latest error", is_error=True),
        ]
        collector = RecentValidationCollector()
        blocks = collector.collect(ContextCollectionInput(messages=msgs))
        assert len(blocks) == 1
        assert "latest error" in blocks[0].content

    def test_successful_validation_returns_empty(self) -> None:
        """成功执行且无错误的验证不产生块。"""
        msgs = [
            ToolResultMessage(
                tool_name="bash", content="All tests passed", is_error=False
            ),
            ToolResultMessage(
                tool_name="bash", content="ruff check passed", is_error=False
            ),
        ]
        collector = RecentValidationCollector()
        blocks = collector.collect(ContextCollectionInput(messages=msgs))
        assert len(blocks) == 0

    def test_oversized_output_bounded(self) -> None:
        """超大输出被截断并包含完整标记。"""
        from xcode.agent.context_collector import RECENT_VALIDATION_MAX_BYTES

        large_content = "error line\n" * 10000
        msg = ToolResultMessage(tool_name="bash", content=large_content, is_error=True)
        collector = RecentValidationCollector()
        blocks = collector.collect(ContextCollectionInput(messages=[msg]))
        assert len(blocks) == 1
        assert "<validation-truncated>" in blocks[0].content
        assert "</validation-truncated>" in blocks[0].content
        assert (
            len(blocks[0].content.encode("utf-8")) <= RECENT_VALIDATION_MAX_BYTES + 200
        )  # +200 for "Command: bash\n" prefix


# ── TaskStateCollector 测试 ──


class TestTaskStateCollector:
    """测试 TaskStateCollector。"""

    def test_no_provider_returns_empty(self) -> None:
        """无 provider 时返回空列表。"""
        collector = TaskStateCollector(provider=None)
        blocks = collector.collect(ContextCollectionInput())
        assert len(blocks) == 0

    def test_empty_state_returns_empty(self) -> None:
        """provider 返回空字符串时返回空列表。"""
        collector = TaskStateCollector(provider=lambda: "")
        blocks = collector.collect(ContextCollectionInput())
        assert len(blocks) == 0

    def test_task_state_emits_block(self) -> None:
        """任务状态发出 USER_CONTEXT 块。"""
        collector = TaskStateCollector(provider=lambda: "- #1 [pending] Implement X")
        blocks = collector.collect(ContextCollectionInput())
        assert len(blocks) == 1
        assert blocks[0].source == ContextBlockSource.TASK_STATE
        assert blocks[0].target == ContextBlockTarget.USER_CONTEXT
        assert blocks[0].priority == ContextPriority.HIGH
        assert "Implement X" in blocks[0].content

    def test_oversized_state_bounded(self) -> None:
        """超大状态被截断并包含完整标记。"""
        large_state = "- #1 [pending] " + "x" * 10000
        collector = TaskStateCollector(provider=lambda: large_state)
        blocks = collector.collect(ContextCollectionInput())
        assert len(blocks) == 1
        content = blocks[0].content
        assert "<task-state-truncated>" in content
        assert "</task-state-truncated>" in content
        from xcode.agent.context_collector import TASK_STATE_MAX_BYTES

        assert len(content.encode("utf-8")) <= TASK_STATE_MAX_BYTES


# ── NotesCollector 测试 ──


class TestNotesCollector:
    """测试 NotesCollector。"""

    def test_missing_notes_dir_returns_empty(self) -> None:
        """缺少 .local/notes/ 目录时返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector = NotesCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 0

    def test_notes_files_emit_block(self) -> None:
        """笔记文件发出 NOTES 块。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_dir = root / ".local" / "notes"
            notes_dir.mkdir(parents=True)
            (notes_dir / "a.md").write_text("Note A content", encoding="utf-8")
            collector = NotesCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert blocks[0].source == ContextBlockSource.NOTES
            assert blocks[0].target == ContextBlockTarget.USER_CONTEXT
            assert blocks[0].priority == ContextPriority.MEDIUM
            assert "Note A content" in blocks[0].content

    def test_deterministic_ordering(self) -> None:
        """笔记文件按路径名字母序排列。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_dir = root / ".local" / "notes"
            notes_dir.mkdir(parents=True)
            (notes_dir / "z.md").write_text("Z content", encoding="utf-8")
            (notes_dir / "a.md").write_text("A content", encoding="utf-8")
            (notes_dir / "m.md").write_text("M content", encoding="utf-8")
            collector = NotesCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            a_pos = blocks[0].content.index("A content")
            m_pos = blocks[0].content.index("M content")
            z_pos = blocks[0].content.index("Z content")
            assert a_pos < m_pos
            assert m_pos < z_pos

    def test_ignores_non_text_files(self) -> None:
        """忽略非 .md/.txt 后缀的文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_dir = root / ".local" / "notes"
            notes_dir.mkdir(parents=True)
            (notes_dir / "note.md").write_text("valid", encoding="utf-8")
            (notes_dir / "data.bin").write_text("binary", encoding="utf-8")
            (notes_dir / "notes.py").write_text("print('hi')", encoding="utf-8")
            collector = NotesCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert "valid" in blocks[0].content
            assert "binary" not in blocks[0].content

    def test_bounded_output(self) -> None:
        """总输出受 NOTES_MAX_BYTES 限制。"""
        from xcode.agent.context_collector import NOTES_MAX_BYTES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_dir = root / ".local" / "notes"
            notes_dir.mkdir(parents=True)
            for name in ("big1.md", "big2.md", "big3.md"):
                (notes_dir / name).write_text("line\n" * 5000, encoding="utf-8")
            collector = NotesCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert len(blocks[0].content.encode("utf-8")) <= NOTES_MAX_BYTES

    def test_bounded_output_has_full_marker(self) -> None:
        """超出预算时包含完整截断标记。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_dir = root / ".local" / "notes"
            notes_dir.mkdir(parents=True)
            (notes_dir / "large.md").write_text("big\n" * 10000, encoding="utf-8")
            collector = NotesCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert "<notes-truncated>" in blocks[0].content
            assert "</notes-truncated>" in blocks[0].content

    def test_oversized_file_skipped(self) -> None:
        """超大文件被跳过。"""
        from xcode.agent.context_collector import NOTES_MAX_FILE_BYTES

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes_dir = root / ".local" / "notes"
            notes_dir.mkdir(parents=True)
            (notes_dir / "small.md").write_text("small note", encoding="utf-8")
            (notes_dir / "huge.md").write_text(
                "x" * (NOTES_MAX_FILE_BYTES + 1), encoding="utf-8"
            )
            collector = NotesCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            assert "small note" in blocks[0].content
            assert "huge" not in blocks[0].content


class TestSkillIndexCollector:
    """Test SkillIndexCollector summary injection."""

    def _make_skill(self, base: Path, *parts: str, content: str) -> Path:
        skill_dir = base.joinpath(*parts)
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        path.write_text(content, encoding="utf-8")
        return path

    def _collect_text(self, collector: SkillIndexCollector) -> str:
        blocks = collector.collect(object())
        if not blocks:
            return ""
        return blocks[0].content

    def test_missing_all_skill_dirs_returns_empty(self) -> None:
        registry = SkillRegistry()
        collector = SkillIndexCollector(registry)
        blocks = collector.collect(object())
        assert len(blocks) == 0

    def test_summary_block_contains_names_and_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_skill(
                root,
                ".xcode",
                "skills",
                "review",
                content=(
                    "---\nname: code-review\ndescription: Review code changes.\n---\n\nFull body."
                ),
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            collector = SkillIndexCollector(registry)
            text = self._collect_text(collector)
            assert "code-review" in text
            assert "Review code changes." in text
            assert "Full body" not in text

    def test_available_skills_xml_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_skill(
                root,
                ".xcode",
                "skills",
                "test",
                content=("---\nname: test-skill\ndescription: Test.\n---\n\nBody."),
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            collector = SkillIndexCollector(registry)
            text = self._collect_text(collector)
            assert "<available-skills>" in text
            assert "</available-skills>" in text

    def test_disable_model_invocation_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_skill(
                root,
                ".xcode",
                "skills",
                "visible",
                content=(
                    "---\nname: visible-skill\ndescription: Visible.\n---\n\nBody."
                ),
            )
            self._make_skill(
                root,
                ".xcode",
                "skills",
                "hidden",
                content=(
                    "---\nname: hidden-skill\ndescription: Hidden.\n"
                    "disable-model-invocation: true\n---\n\nBody."
                ),
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            collector = SkillIndexCollector(registry)
            text = self._collect_text(collector)
            assert "visible-skill" in text
            assert "hidden-skill" not in text

    def test_block_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_skill(
                root,
                ".xcode",
                "skills",
                "test",
                content=("---\nname: test-skill\ndescription: Test.\n---\n\nBody."),
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            collector = SkillIndexCollector(registry)
            blocks = collector.collect(object())
            assert len(blocks) == 1
            assert blocks[0].source == ContextBlockSource.SKILL
            assert blocks[0].target == ContextBlockTarget.USER_CONTEXT
            assert blocks[0].priority == ContextPriority.MEDIUM

    def test_existing_collectors_still_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_skill(
                root,
                ".xcode",
                "skills",
                "my-skill",
                content=("---\nname: my-skill\ndescription: My skill.\n---\n\nBody."),
            )
            notes_dir = root / ".local" / "notes"
            notes_dir.mkdir(parents=True)
            (notes_dir / "my-note.md").write_text("My Note", encoding="utf-8")

            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            collector_registry = ContextCollectorRegistry()
            collector_registry.register(SkillIndexCollector(registry))
            collector_registry.register(NotesCollector(root))
            input_ = ContextCollectionInput(project_root=root)
            blocks = collector_registry.collect(input_)
            assert len(blocks) == 2
            sources = {b.source for b in blocks}
            assert ContextBlockSource.SKILL in sources
            assert ContextBlockSource.NOTES in sources


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=root,
        capture_output=True,
        check=True,
    )


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
