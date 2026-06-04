from __future__ import annotations

from typing import Any
import unittest

from xcode.agent.agent_loop import run_agent_loop
from xcode.agent.messages import convert_to_llm
from xcode.agent.provider_response import provider_events_to_response
from xcode.agent.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopResult,
    AgentToolResult,
    AssistantMessage,
    TextContent,
    ToolExecutionMode,
    ToolResultMessage,
    UserMessage,
)
from xcode.ai.events import (
    FinalMessage,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from xcode.ai.types import StreamOptions, ToolDefinition


class TextProvider:
    def __init__(self) -> None:
        self.messages: list[Message] | None = None
        self.tools: list[ToolDefinition] | None = None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ):
        self.messages = messages
        self.tools = tools
        yield TextDelta("ok")


class ToolProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[Message]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            yield ToolCallEvent([ToolCall("call-1", "echo", {"text": "hello"})])
            return
        yield TextDelta("done")


class EchoTool:
    name = "echo"
    label = "Echo"
    description = "Echo text."
    parameters = {"type": "object"}
    execution_mode: ToolExecutionMode | None = "sequential"

    def __init__(self) -> None:
        self.seen_signal: Any | None = None

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update=None,
    ) -> AgentToolResult[None]:
        self.seen_signal = signal
        return AgentToolResult(content=[TextContent(text=str(params["text"]))])


class AgentLoopContractTests(unittest.IsolatedAsyncioTestCase):
    def test_provider_events_to_response_keeps_core_stream_semantics(self) -> None:
        response = provider_events_to_response(
            [
                ReasoningDelta("why"),
                TextDelta("hel"),
                TextDelta("lo"),
                ToolCallEvent([ToolCall("call-1", "echo", {"text": "hello"})]),
                FinalMessage("", stop_reason="tool_use"),
            ]
        )

        self.assertEqual(response.reasoning_content, "why")
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(response.deltas[0].kind, "reasoning")
        self.assertEqual(response.deltas[1].chunk, "hel")
        self.assertEqual(response.content[0], TextContent(text="hello"))

    async def test_streams_text_from_injected_provider(self) -> None:
        provider = TextProvider()
        events: list[AgentEvent] = []

        result = await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=AgentContext(),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
            ),
            emit=events.append,
        )

        self.assertEqual(provider.messages, [{"role": "user", "content": "hello"}])
        self.assertEqual(provider.tools, [])
        final = result.messages[-1]
        self.assertIsInstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        self.assertEqual(final.content, [TextContent(text="ok")])
        self.assertEqual(events[-1].type, "agent_end")

    async def test_executes_tools_without_harness_context(self) -> None:
        provider = ToolProvider()
        tool = EchoTool()
        events: list[AgentEvent] = []

        result = await run_agent_loop(
            prompts=[UserMessage(content="use tool")],
            context=AgentContext(tools=[tool]),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
            ),
            emit=events.append,
        )

        self.assertEqual(provider.calls, 2)
        self.assertIsNone(tool.seen_signal)
        self.assertTrue(
            any(isinstance(msg, ToolResultMessage) for msg in result.messages)
        )
        self.assertEqual(provider.messages[-1][-1]["role"], "tool")
        final = result.messages[-1]
        self.assertIsInstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        self.assertEqual(final.content, [TextContent(text="done")])


# ── 新增功能测试 ──


class StepLimitProvider:
    """总是返回工具调用，用于测试步数限制。"""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools, options: StreamOptions | None = None, **kwargs: Any):
        self.calls += 1
        yield ToolCallEvent([ToolCall(f"call-{self.calls}", "noop", {})])


class NoopTool:
    name = "noop"
    label = "Noop"
    description = "Does nothing."
    parameters = {"type": "object"}
    execution_mode: ToolExecutionMode | None = None

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])


class ErrorProvider:
    """前 N 次调用抛异常，之后返回正常文本。"""

    def __init__(self, fail_count: int = 1, error: Exception | None = None) -> None:
        self.fail_count = fail_count
        self.calls = 0
        self.error = error or RuntimeError("transient error: rate limit")

    async def stream(self, messages, tools, options: StreamOptions | None = None, **kwargs: Any):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise self.error
        yield TextDelta("recovered")


class MaxTokensProvider:
    """前 N 次返回 max_tokens，之后返回正常文本。"""

    def __init__(self, max_tokens_count: int = 1) -> None:
        self.max_tokens_count = max_tokens_count
        self.calls = 0

    async def stream(self, messages, tools, options: StreamOptions | None = None, **kwargs: Any):
        self.calls += 1
        if self.calls <= self.max_tokens_count:
            yield TextDelta("partial" * 100)
            yield FinalMessage("", stop_reason="max_tokens")
        else:
            yield TextDelta("final")


class RepeatedToolProvider:
    """始终返回相同的工具调用，用于测试重复工具看门狗。"""

    async def stream(self, messages, tools, options: StreamOptions | None = None, **kwargs: Any):
        yield ToolCallEvent([ToolCall("same-call", "echo", {"text": "hi"})])


class AgentLoopFeatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_limit_enforced(self) -> None:
        provider = StepLimitProvider()
        tool = NoopTool()

        result = await run_agent_loop(
            prompts=[UserMessage(content="go")],
            context=AgentContext(tools=[tool]),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
                max_steps=3,
            ),
            emit=lambda e: None,
        )

        self.assertIsInstance(result, AgentLoopResult)
        self.assertTrue(result.stopped_by_limit)
        self.assertEqual(result.steps, 3)

    async def test_error_retry_recovers(self) -> None:
        provider = ErrorProvider(fail_count=1)

        result = await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=AgentContext(),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
                max_step_retries=3,
                retry_backoff_base=0.01,
            ),
            emit=lambda e: None,
        )

        self.assertEqual(provider.calls, 2)
        final = result.messages[-1]
        self.assertIsInstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        self.assertEqual(final.content, [TextContent(text="recovered")])

    async def test_error_retry_exhausted(self) -> None:
        provider = ErrorProvider(fail_count=10)

        await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=AgentContext(),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
                max_step_retries=2,
                retry_backoff_base=0.01,
            ),
            emit=lambda e: None,
        )

        # 应该重试 max_step_retries 次后放弃
        self.assertGreater(provider.calls, 1)

    async def test_max_tokens_continuation(self) -> None:
        provider = MaxTokensProvider(max_tokens_count=1)

        result = await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=AgentContext(),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
                max_tokens_continuation=True,
                max_consecutive_continuations=3,
                min_continuation_tokens=10,
            ),
            emit=lambda e: None,
        )

        # 应该续写一次后完成
        self.assertEqual(provider.calls, 2)
        final = result.messages[-1]
        self.assertIsInstance(final, AssistantMessage)

    async def test_repeated_tool_watchdog(self) -> None:
        provider = RepeatedToolProvider()
        tool = EchoTool()

        result = await run_agent_loop(
            prompts=[UserMessage(content="loop")],
            context=AgentContext(tools=[tool]),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
                watchdog_repeated_tool_limit=3,
                max_steps=100,
            ),
            emit=lambda e: None,
        )

        self.assertTrue(result.stopped_by_watchdog)
        self.assertIsNotNone(result.watchdog_reason)
        assert result.watchdog_reason is not None
        self.assertIn("repeated", result.watchdog_reason.lower())

    async def test_idle_step_watchdog(self) -> None:
        """测试空闲步骤看门狗：工具总是抛出异常。"""

        class AlwaysFailTool:
            name = "fail"
            label = "Fail"
            description = "Always fails."
            parameters = {"type": "object"}
            execution_mode: ToolExecutionMode | None = None

            async def execute(self, tool_call_id, params, signal=None, on_update=None):
                raise RuntimeError("tool failed")

        class FailToolProvider:
            def __init__(self):
                self.call_count = 0

            async def stream(self, messages, tools, options: StreamOptions | None = None, **kwargs: Any):
                self.call_count += 1
                yield ToolCallEvent([ToolCall(f"call-{self.call_count}", "fail", {})])

        provider = FailToolProvider()
        tool = AlwaysFailTool()

        result = await run_agent_loop(
            prompts=[UserMessage(content="fail")],
            context=AgentContext(tools=[tool]),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
                max_consecutive_idle_steps=2,
                watchdog_repeated_tool_limit=0,
                max_steps=100,
            ),
            emit=lambda e: None,
        )

        self.assertTrue(result.stopped_by_watchdog)
        self.assertIsNotNone(result.watchdog_reason)
        assert result.watchdog_reason is not None
        self.assertIn("consecutive steps", result.watchdog_reason.lower())

    async def test_compaction_hook_called(self) -> None:
        compact_called = False

        def should_compact(messages):
            # 始终返回 True，确保压缩被触发
            return True

        def compact(messages):
            nonlocal compact_called
            compact_called = True
            return messages[-1:]

        provider = TextProvider()

        await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=AgentContext(),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
                should_compact=should_compact,
                compact=compact,
            ),
            emit=lambda e: None,
        )

        self.assertTrue(compact_called)

    async def test_metrics_collected(self) -> None:
        provider = ToolProvider()
        tool = EchoTool()

        result = await run_agent_loop(
            prompts=[UserMessage(content="use tool")],
            context=AgentContext(tools=[tool]),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
            ),
            emit=lambda e: None,
        )

        self.assertIsNotNone(result.metrics)
        assert result.metrics is not None
        self.assertGreaterEqual(result.metrics.steps, 1)
        self.assertGreater(result.metrics.llm_calls, 0)
        self.assertGreater(result.metrics.tool_calls, 0)


if __name__ == "__main__":
    unittest.main()
