"""上下文组装模块测试。

测试 ContextBlock、DefaultContextAssembler 以及 context_assembler hook
在 agent 循环中的行为。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
import unittest

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
    ContextCollectionInput,
    ContextCollectorRegistry,
    ProjectManifestCollector,
)
from xcode.agent.messages import (
    SystemMessage,
    UserMessage,
)
from xcode.agent.message_converter import convert_to_llm
from xcode.agent.protocols import AgentToolResult, ToolExecutionMode
from xcode.agent.types import TextContent
from xcode.ai.events import Message, TextDelta, ToolCall, ToolCallEvent
from xcode.ai.types import StreamOptions, ToolDefinition


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


class TestContextBlock(unittest.TestCase):
    """测试 ContextBlock 数据模型。"""

    def test_token_count_cached(self) -> None:
        """预计算的 token_count 应直接返回。"""
        block = ContextBlock(
            source=ContextBlockSource.PLAN,
            priority=ContextPriority.HIGH,
            content="some content",
            token_count=42,
        )
        self.assertEqual(block.get_token_count(), 42)

    def test_token_count_estimated(self) -> None:
        """未预计算时自动估算。"""
        block = ContextBlock(
            source=ContextBlockSource.NOTES,
            priority=ContextPriority.LOW,
            content="hello world",
        )
        self.assertGreater(block.get_token_count(), 0)

    def test_default_created_values(self) -> None:
        """created_turn 和 created_step 默认为 0。"""
        block = ContextBlock(
            source=ContextBlockSource.CUSTOM,
            priority=ContextPriority.MEDIUM,
            content="test",
        )
        self.assertEqual(block.created_turn, 0)
        self.assertEqual(block.created_step, 0)

    def test_created_values_custom(self) -> None:
        """created_turn 和 created_step 可自定义。"""
        block = ContextBlock(
            source=ContextBlockSource.CUSTOM,
            priority=ContextPriority.HIGH,
            content="test",
            created_turn=5,
            created_step=3,
        )
        self.assertEqual(block.created_turn, 5)
        self.assertEqual(block.created_step, 3)


class TestContextExpiry(unittest.TestCase):
    """测试 ContextExpiry 过期策略。"""

    def test_never_by_default(self) -> None:
        """默认永不过期。"""
        expiry = ContextExpiry()
        self.assertTrue(expiry.never)

    def test_not_never_when_set(self) -> None:
        """设置任何过期条件后 never 返回 False。"""
        self.assertFalse(ContextExpiry(max_turns=5).never)
        self.assertFalse(ContextExpiry(max_steps=10).never)

    def test_max_turns_zero_means_unlimited(self) -> None:
        """max_turns=0 表示不限。"""
        self.assertTrue(ContextExpiry(max_turns=0).never)
        self.assertFalse(ContextExpiry(max_turns=1).never)

    def test_max_steps_zero_means_unlimited(self) -> None:
        """max_steps=0 表示不限。"""
        self.assertTrue(ContextExpiry(max_steps=0).never)
        self.assertFalse(ContextExpiry(max_steps=1).never)


class TestContextPriority(unittest.TestCase):
    """测试 ContextPriority 排序语义。"""

    def test_ordering(self) -> None:
        """CRITICAL < HIGH < MEDIUM < LOW < BACKGROUND。"""
        self.assertLess(ContextPriority.CRITICAL, ContextPriority.HIGH)
        self.assertLess(ContextPriority.HIGH, ContextPriority.MEDIUM)
        self.assertLess(ContextPriority.MEDIUM, ContextPriority.LOW)
        self.assertLess(ContextPriority.LOW, ContextPriority.BACKGROUND)


# ── trim_to_budget 纯函数测试 ──


class TestTrimToBudget(unittest.TestCase):
    """测试 trim_to_budget 纯函数。"""

    def test_no_budget_returns_all_sorted(self) -> None:
        """budget <= 0 时所有块按优先级排序后返回。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.BACKGROUND,
                content="bg",
            ),
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.CRITICAL,
                content="critical",
            ),
        ]
        used, dropped = trim_to_budget(blocks, 0, 0)
        self.assertEqual(len(used), 2)
        self.assertEqual(len(dropped), 0)
        self.assertEqual(used[0].priority, ContextPriority.CRITICAL)
        self.assertEqual(used[1].priority, ContextPriority.BACKGROUND)

    def test_budget_exceeded_drops_lowest_priority(self) -> None:
        """超出预算时丢弃扫描到的块，剩余的是高优先级块。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.HIGH,
                content="high",
                token_count=50,
            ),
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.BACKGROUND,
                content="low",
                token_count=100,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=60, base_tokens=0)
        self.assertEqual(len(used), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(used[0].priority, ContextPriority.HIGH)
        self.assertEqual(dropped[0].priority, ContextPriority.BACKGROUND)

    def test_deterministic_with_equal_priority(self) -> None:
        """同优先级顺序稳定。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.MEDIUM,
                content="first",
                token_count=30,
            ),
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.MEDIUM,
                content="second",
                token_count=30,
            ),
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.MEDIUM,
                content="third",
                token_count=30,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=50, base_tokens=0)
        self.assertEqual(len(used), 1)
        self.assertEqual(len(dropped), 2)
        self.assertEqual(used[0].content, "first")

    def test_base_tokens_consumes_budget_critical_dropped(self) -> None:
        """base_tokens 占满预算时即使是 CRITICAL 块也被丢弃。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.CRITICAL,
                content="important",
                token_count=10,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=10, base_tokens=10)
        self.assertEqual(len(used), 0)
        self.assertEqual(len(dropped), 1)

    def test_greedy_policy_skips_large_high_priority(self) -> None:
        """高优先级块太大时跳过，继续尝试低优先级小块的 greedy 策略。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.HIGH,
                content="too large",
                token_count=100,
            ),
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.MEDIUM,
                content="small fits",
                token_count=3,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=10, base_tokens=0)
        self.assertEqual(len(used), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(used[0].priority, ContextPriority.MEDIUM)
        self.assertEqual(dropped[0].priority, ContextPriority.HIGH)

    def test_greedy_half_fill(self) -> None:
        """多个块填充到预算上限，剩余丢弃。"""
        blocks = [
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.CRITICAL,
                content="a",
                token_count=10,
            ),
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.HIGH,
                content="b",
                token_count=10,
            ),
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
                priority=ContextPriority.LOW,
                content="c",
                token_count=10,
            ),
        ]
        used, dropped = trim_to_budget(blocks, budget=25, base_tokens=0)
        # a(10) + b(10) = 20 fits, c(10) exceeds remaining(5) → dropped
        self.assertEqual(len(used), 2)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].content, "c")


# ── DefaultContextAssembler 单元测试 ──


class TestDefaultContextAssemblerNoOp(unittest.TestCase):
    """未配置 context_blocks 时行为不变。"""

    def setUp(self) -> None:
        self.assembler = DefaultContextAssembler()

    def test_messages_unchanged_no_blocks(self) -> None:
        """无 blocks 时消息原样返回。"""
        msgs: list[UserMessage] = [UserMessage(content="hello")]
        result = self.assembler.assemble(ContextAssemblyInput(messages=msgs))
        self.assertEqual(len(result.messages), 1)
        self.assertIs(result.messages[0], msgs[0])

    def test_result_metadata_zero_when_no_blocks(self) -> None:
        """无 blocks 时 metadata 为零值。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(messages=[UserMessage(content="hi")])
        )
        self.assertEqual(len(result.blocks_used), 0)
        self.assertEqual(len(result.blocks_dropped), 0)
        self.assertEqual(result.total_tokens, 1)


class TestDefaultContextAssemblerPriority(unittest.TestCase):
    """测试优先级排序。"""

    def setUp(self) -> None:
        self.assembler = DefaultContextAssembler()

    def test_blocks_ordered_by_priority(self) -> None:
        """块按优先级排序后注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.BACKGROUND,
                        content="bg",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.CRITICAL,
                        content="critical",
                    ),
                ],
            )
        )
        self.assertEqual(len(result.blocks_used), 2)
        self.assertEqual(result.blocks_used[0].priority, ContextPriority.CRITICAL)
        self.assertEqual(result.blocks_used[1].priority, ContextPriority.BACKGROUND)

    def test_deterministic_order_within_same_priority(self) -> None:
        """同优先级保持原始插入顺序。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.MEDIUM,
                        content="first",
                        block_id="a",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.MEDIUM,
                        content="second",
                        block_id="b",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.MEDIUM,
                        content="third",
                        block_id="c",
                    ),
                ],
            )
        )
        ids = [b.block_id for b in result.blocks_used]
        self.assertEqual(ids, ["a", "b", "c"])


class TestDefaultContextAssemblerBudget(unittest.TestCase):
    """测试预算裁剪。"""

    def setUp(self) -> None:
        self.assembler = DefaultContextAssembler()

    def test_budget_trimming_drops_lowest_priority(self) -> None:
        """超出预算时丢弃低优先级。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hi")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.CRITICAL,
                        content="critical content",
                        token_count=10,
                    ),
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.BACKGROUND,
                        content="background noise",
                        token_count=100,
                    ),
                ],
                token_budget=15,
            )
        )
        self.assertEqual(len(result.blocks_used), 1)
        self.assertEqual(result.blocks_used[0].priority, ContextPriority.CRITICAL)
        self.assertEqual(len(result.blocks_dropped), 1)
        self.assertEqual(result.blocks_dropped[0].priority, ContextPriority.BACKGROUND)

    def test_critical_dropped_when_base_exceeds_budget(self) -> None:
        """base messages 已耗尽预算时即使是 CRITICAL 块也丢弃。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="x" * 2000)],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.CRITICAL,
                        content="vital",
                        token_count=10,
                    ),
                ],
                token_budget=5,
            )
        )
        self.assertEqual(len(result.blocks_used), 0)
        self.assertEqual(len(result.blocks_dropped), 1)
        self.assertEqual(result.blocks_dropped[0].priority, ContextPriority.CRITICAL)

    def test_greedy_fill_over_budget(self) -> None:
        """高优先级块太大时跳过，小块的次高优先级块保留。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="base")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.CRITICAL,
                        content="big critical",
                        token_count=100,
                    ),
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.HIGH,
                        content="small high",
                        token_count=3,
                    ),
                ],
                token_budget=10,
            )
        )
        self.assertEqual(len(result.blocks_used), 1)
        self.assertEqual(result.blocks_used[0].priority, ContextPriority.HIGH)
        self.assertEqual(len(result.blocks_dropped), 1)
        self.assertEqual(result.blocks_dropped[0].priority, ContextPriority.CRITICAL)

    def test_budget_trimming_deterministic(self) -> None:
        """相同输入产生相同结果。"""
        input_blocks = [
            ContextBlock(
                source=ContextBlockSource.CUSTOM,
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

        self.assertEqual(len(r1.blocks_used), len(r2.blocks_used))
        self.assertEqual(
            [b.block_id for b in r1.blocks_used],
            [b.block_id for b in r2.blocks_used],
        )


class TestDefaultContextAssemblerExpiry(unittest.TestCase):
    """测试过期过滤（相对期限）。"""

    def setUp(self) -> None:
        self.assembler = DefaultContextAssembler()

    def test_expired_by_step_relative(self) -> None:
        """created_step=1, max_steps=3, current_step=4 → 4-1=3 >= 3 → 过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.HIGH,
                        content="valid",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.CRITICAL,
                        content="expired by step",
                        expiry=ContextExpiry(max_steps=3),
                        created_step=1,
                    ),
                ],
                current_step=4,
            )
        )
        self.assertEqual(len(result.blocks_used), 1)
        self.assertEqual(result.blocks_used[0].content, "valid")
        self.assertEqual(len(result.blocks_dropped), 1)
        self.assertEqual(result.blocks_dropped[0].content, "expired by step")

    def test_not_expired_by_step_before_limit(self) -> None:
        """created_step=1, max_steps=3, current_step=3 → 3-1=2 < 3 → 未过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.HIGH,
                        content="still valid",
                        expiry=ContextExpiry(max_steps=3),
                        created_step=1,
                    ),
                ],
                current_step=3,
            )
        )
        self.assertEqual(len(result.blocks_used), 1)

    def test_expired_by_turn_relative(self) -> None:
        """created_turn=2, max_turns=3, current_turn=5 → 5-2=3 >= 3 → 过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.MEDIUM,
                        content="still valid",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.HIGH,
                        content="expired by turn",
                        expiry=ContextExpiry(max_turns=3),
                        created_turn=2,
                    ),
                ],
                current_turn=5,
            )
        )
        self.assertEqual(len(result.blocks_used), 1)
        self.assertEqual(result.blocks_used[0].content, "still valid")

    def test_not_expired_by_turn_before_limit(self) -> None:
        """created_turn=2, max_turns=3, current_turn=4 → 4-2=2 < 3 → 未过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.HIGH,
                        content="not expired yet",
                        expiry=ContextExpiry(max_turns=3),
                        created_turn=2,
                    ),
                ],
                current_turn=4,
            )
        )
        self.assertEqual(len(result.blocks_used), 1)

    def test_never_expiry_not_excluded(self) -> None:
        """never 过期策略的块永不被排除。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.HIGH,
                        content="never expires",
                        expiry=ContextExpiry(),
                    ),
                ],
                current_step=999,
                current_turn=999,
            )
        )
        self.assertEqual(len(result.blocks_used), 1)

    def test_none_expiry_not_excluded(self) -> None:
        """expiry 为 None 时永不过期。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.HIGH,
                        content="no expiry set",
                        expiry=None,
                    ),
                ],
                current_step=999,
            )
        )
        self.assertEqual(len(result.blocks_used), 1)

    def test_expiry_with_default_created(self) -> None:
        """created_turn/created_step 默认 0，行为等价于绝对截止。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.HIGH,
                        content="expires at step 3 from start",
                        expiry=ContextExpiry(max_steps=3),
                    ),
                ],
                current_step=3,
            )
        )
        # created_step=0, 3-0=3 >= 3 → expired
        self.assertEqual(len(result.blocks_used), 0)
        self.assertEqual(len(result.blocks_dropped), 1)


class TestDefaultContextAssemblerMessageInjection(unittest.TestCase):
    """测试消息注入位置。"""

    def setUp(self) -> None:
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
                        source=ContextBlockSource.PLAN,
                        priority=ContextPriority.HIGH,
                        content="plan info",
                    ),
                ],
            )
        )
        self.assertEqual(len(result.messages), 3)
        self.assertEqual(getattr(result.messages[0], "role", ""), "system")
        self.assertEqual(getattr(result.messages[1], "role", ""), "user")
        self.assertIn("plan info", str(result.messages[1].content))
        # 验证 synthetic 消息包含 [plan] 来源标记
        self.assertIn("[plan]", str(result.messages[1].content))

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
        self.assertEqual(len(result.messages), 2)
        self.assertIn("note content", str(result.messages[0].content))
        self.assertIn("[notes]", str(result.messages[0].content))

    def test_multiple_blocks_all_injected(self) -> None:
        """多个块全部注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hi")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.CUSTOM,
                        priority=ContextPriority.LOW,
                        content=f"block-{i}",
                    )
                    for i in range(3)
                ],
            )
        )
        self.assertEqual(len(result.messages), 4)
        self.assertEqual(len(result.blocks_used), 3)


# ── Agent 循环集成测试 ──


class TestContextAssemblerAgentLoopIntegration(unittest.TestCase):
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
        self.assertEqual(len(provider.captured_messages), 1)
        msgs = provider.captured_messages[0]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        self.assertGreaterEqual(len(user_msgs), 1)

    def test_legacy_transform_context_still_works(self) -> None:
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
        self.assertTrue(transform_called)

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
        self.assertEqual(len(transform_log), 1)

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
        self.assertGreater(len(provider.captured_messages), 0)

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
        self.assertEqual(len(provider.captured_messages), 1)
        msgs = provider.captured_messages[0]
        self.assertGreater(len(msgs), 0)

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
        self.assertGreaterEqual(len(captured_steps), 1)
        # 第一个 step 应为 1
        self.assertEqual(captured_steps[0], 1)


class TestContextAssemblyInputConstruction(unittest.TestCase):
    """测试 ContextAssemblyInput 的 provider 侧构造。"""

    def test_default_values(self) -> None:
        """默认值不应导致错误。"""
        inp = ContextAssemblyInput()
        self.assertEqual(inp.current_turn, 0)
        self.assertEqual(inp.current_step, 0)
        self.assertEqual(inp.token_budget, 0)
        self.assertEqual(len(inp.state), 0)
        self.assertEqual(len(inp.messages), 0)
        self.assertEqual(len(inp.tools), 0)
        self.assertEqual(len(inp.context_blocks), 0)

    def test_system_prompt_preserved(self) -> None:
        """system_prompt 字段传递正确。"""
        inp = ContextAssemblyInput(system_prompt="you are a helpful agent")
        self.assertEqual(inp.system_prompt, "you are a helpful agent")


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


class TestContextCollectorRegistry(unittest.TestCase):
    """测试 ContextCollectorRegistry 基本行为。"""

    def test_empty_registry_returns_empty(self) -> None:
        """空注册表返回空列表。"""
        registry = ContextCollectorRegistry()
        result = registry.collect(ContextCollectionInput())
        self.assertEqual(len(result), 0)

    def test_empty_registry_is_falsey(self) -> None:
        """空注册表 __bool__ 为 False。"""
        registry = ContextCollectorRegistry()
        self.assertFalse(registry)

    def test_registry_with_collectors_is_truthy(self) -> None:
        """有 collector 时 __bool__ 为 True。"""
        registry = ContextCollectorRegistry()
        registry.register(FakeCollector([]))
        self.assertTrue(registry)

    def test_single_collector_returns_blocks(self) -> None:
        """单个 collector 返回的块被正确收集。"""
        block = ContextBlock(
            source=ContextBlockSource.CUSTOM,
            priority=ContextPriority.HIGH,
            content="from collector",
        )
        registry = ContextCollectorRegistry()
        registry.register(FakeCollector([block]))
        result = registry.collect(ContextCollectionInput())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "from collector")

    def test_multiple_collectors_preserve_order(self) -> None:
        """多个 collector 按注册顺序合并结果。"""
        block_a = ContextBlock(
            source=ContextBlockSource.CUSTOM,
            priority=ContextPriority.HIGH,
            content="first",
            block_id="a",
        )
        block_b = ContextBlock(
            source=ContextBlockSource.CUSTOM,
            priority=ContextPriority.MEDIUM,
            content="second",
            block_id="b",
        )
        block_c = ContextBlock(
            source=ContextBlockSource.CUSTOM,
            priority=ContextPriority.LOW,
            content="third",
            block_id="c",
        )
        registry = ContextCollectorRegistry()
        registry.register(FakeCollector([block_a, block_b]))
        registry.register(FakeCollector([block_c]))
        result = registry.collect(ContextCollectionInput())
        self.assertEqual(len(result), 3)
        self.assertEqual([b.block_id for b in result], ["a", "b", "c"])

    def test_error_collector_skipped_other_still_run(self) -> None:
        """异常 collector 被跳过（log + skip），其他 collector 仍正常执行。"""
        good_block = ContextBlock(
            source=ContextBlockSource.CUSTOM,
            priority=ContextPriority.HIGH,
            content="good",
        )
        registry = ContextCollectorRegistry()
        registry.register(ErrorCollector())
        registry.register(FakeCollector([good_block]))
        # 异常被吞噬，不会传播到调用方
        result = registry.collect(ContextCollectionInput())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "good")

    def test_error_collector_exception_never_propagates(self) -> None:
        """collector 异常绝不传播到 collect() 的调用方。"""
        registry = ContextCollectorRegistry()
        registry.register(ErrorCollector())
        # 不需要 try/except，因为 collect 内部已捕获
        result = registry.collect(ContextCollectionInput())
        self.assertEqual(len(result), 0)


class TestContextCollectorWithAssembler(unittest.TestCase):
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
                        source=ContextBlockSource.CUSTOM,
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
        self.assertEqual(len(captured_blocks), 1)
        self.assertEqual(captured_blocks[0].block_id, "c1")

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
        self.assertEqual(len(captured_blocks), 0)

    def test_collector_with_default_assembler_block_injected(self) -> None:
        """collector 产出的块经 DefaultContextAssembler 注入到消息中。"""
        provider = CaptureProvider()
        registry = ContextCollectorRegistry()
        registry.register(
            FakeCollector(
                [
                    ContextBlock(
                        source=ContextBlockSource.PLAN,
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
        self.assertEqual(len(provider.captured_messages), 1)
        msgs = provider.captured_messages[0]
        combined = " ".join(str(m.get("content", "") or "") for m in msgs)
        self.assertIn("plan step 1", combined)
        self.assertIn("[plan]", combined)

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
        self.assertEqual(len(call_count), 0)

    def test_legacy_transform_context_still_runs_after_collectors(self) -> None:
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
        self.assertEqual(len(transform_log), 1)


# ── ContextBlockTarget 测试 ──


class TestContextBlockTarget(unittest.TestCase):
    """测试 ContextBlockTarget 枚举。"""

    def test_default_target_is_user_context(self) -> None:
        """ContextBlock 默认 target 为 USER_CONTEXT。"""
        block = ContextBlock(
            source=ContextBlockSource.CUSTOM,
            priority=ContextPriority.HIGH,
            content="test",
        )
        self.assertEqual(block.target, ContextBlockTarget.USER_CONTEXT)

    def test_system_target_explicit(self) -> None:
        """SYSTEM target 可显式设置。"""
        block = ContextBlock(
            source=ContextBlockSource.PROJECT_MANIFEST,
            target=ContextBlockTarget.SYSTEM,
            priority=ContextPriority.CRITICAL,
            content="system instructions",
        )
        self.assertEqual(block.target, ContextBlockTarget.SYSTEM)

    def test_target_enum_values(self) -> None:
        """枚举值正确。"""
        self.assertEqual(ContextBlockTarget.SYSTEM.value, "system")
        self.assertEqual(ContextBlockTarget.USER_CONTEXT.value, "user_context")


# ── SYSTEM 块组装测试 ──


class TestDefaultContextAssemblerSystemBlocks(unittest.TestCase):
    """测试 SYSTEM 目标块的组装行为。"""

    def setUp(self) -> None:
        self.assembler = DefaultContextAssembler()

    def test_system_block_becomes_system_message(self) -> None:
        """SYSTEM 块被注入为 SystemMessage。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="hello")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.PROJECT_MANIFEST,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="project rules",
                    ),
                ],
            )
        )
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(getattr(result.messages[0], "role", ""), "system")
        self.assertEqual(result.messages[0].content, "project rules")
        self.assertEqual(result.blocks_used[0].content, "project rules")

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
                        source=ContextBlockSource.PROJECT_MANIFEST,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="project rules",
                    ),
                ],
            )
        )
        self.assertEqual(len(result.messages), 3)
        roles = [getattr(m, "role", "") for m in result.messages]
        self.assertEqual(roles, ["system", "system", "user"])
        self.assertEqual(result.messages[0].content, "identity prompt")
        self.assertEqual(result.messages[1].content, "project rules")

    def test_mixed_targets_separate_injection(self) -> None:
        """SYSTEM 和 USER_CONTEXT 块分别注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="question")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.PROJECT_MANIFEST,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="project rules",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.PLAN,
                        target=ContextBlockTarget.USER_CONTEXT,
                        priority=ContextPriority.HIGH,
                        content="plan step",
                    ),
                ],
            )
        )
        self.assertEqual(len(result.messages), 3)
        roles = [getattr(m, "role", "") for m in result.messages]
        self.assertEqual(roles, ["system", "user", "user"])
        self.assertEqual(result.messages[0].content, "project rules")
        self.assertIn("plan step", str(result.messages[1].content))
        self.assertIn("[plan]", str(result.messages[1].content))

    def test_multiple_system_blocks_all_as_system_messages(self) -> None:
        """多个 SYSTEM 块全部作为 SystemMessage 注入。"""
        result = self.assembler.assemble(
            ContextAssemblyInput(
                messages=[UserMessage(content="question")],
                context_blocks=[
                    ContextBlock(
                        source=ContextBlockSource.PROJECT_MANIFEST,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="rules part 1",
                    ),
                    ContextBlock(
                        source=ContextBlockSource.PROJECT_MANIFEST,
                        target=ContextBlockTarget.SYSTEM,
                        priority=ContextPriority.CRITICAL,
                        content="rules part 2",
                    ),
                ],
            )
        )
        self.assertEqual(len(result.messages), 3)
        roles = [getattr(m, "role", "") for m in result.messages]
        self.assertEqual(roles, ["system", "system", "user"])


# ── ProjectManifestCollector 测试 ──


class TestProjectManifestCollector(unittest.TestCase):
    """测试 ProjectManifestCollector。"""

    def test_no_files_returns_empty(self) -> None:
        """项目根目录无 AGENTS.md / CLAUDE.md 时返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector = ProjectManifestCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            self.assertEqual(len(blocks), 0)

    def test_agents_md_content_collected(self) -> None:
        """AGENTS.md 内容被收集为 PROJECT_MANIFEST + SYSTEM + CRITICAL。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Use tests.", encoding="utf-8")
            collector = ProjectManifestCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0].source, ContextBlockSource.PROJECT_MANIFEST)
            self.assertEqual(blocks[0].target, ContextBlockTarget.SYSTEM)
            self.assertEqual(blocks[0].priority, ContextPriority.CRITICAL)
            self.assertEqual(blocks[0].content, "Use tests.")

    def test_agents_md_via_input_project_root(self) -> None:
        """project_root 可通过 ContextCollectionInput 传入。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
            collector = ProjectManifestCollector()
            inp = ContextCollectionInput(project_root=root)
            blocks = collector.collect(inp)
            self.assertEqual(len(blocks), 1)
            self.assertIn("Project rules.", blocks[0].content)

    def test_claude_md_agents_reference_skipped(self) -> None:
        """CLAUDE.md 仅包含 @AGENTS.md 引用时不重复发出块。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Real content.", encoding="utf-8")
            (root / "CLAUDE.md").write_text("@AGENTS.md", encoding="utf-8")
            collector = ProjectManifestCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0].content, "Real content.")

    def test_claude_md_own_content_not_skipped(self) -> None:
        """CLAUDE.md 有自有内容时作为独立块发出。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("AGENTS content.", encoding="utf-8")
            (root / "CLAUDE.md").write_text("CLAUDE specific rules.", encoding="utf-8")
            collector = ProjectManifestCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            self.assertEqual(len(blocks), 2)
            contents = [b.content for b in blocks]
            self.assertIn("AGENTS content.", contents)
            self.assertIn("CLAUDE specific rules.", contents)

    def test_claude_md_reference_with_whitespace(self) -> None:
        """带空格的 @AGENTS.md 引用也被识别。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Content.", encoding="utf-8")
            (root / "CLAUDE.md").write_text("  @AGENTS.md  ", encoding="utf-8")
            collector = ProjectManifestCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            self.assertEqual(len(blocks), 1)

    def test_only_claude_md_without_agents(self) -> None:
        """仅有 CLAUDE.md 且其内容有效时作为独立块发出。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("CLAUDE rules.", encoding="utf-8")
            collector = ProjectManifestCollector(root)
            blocks = collector.collect(ContextCollectionInput())
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0].content, "CLAUDE rules.")
