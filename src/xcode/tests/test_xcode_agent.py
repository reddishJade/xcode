from __future__ import annotations

from typing import Any
import unittest

from xcode.agent.agent_loop import run_agent_loop
from xcode.agent.messages import convert_to_llm
from xcode.agent.config import AgentContext, AgentLoopConfig, AgentLoopResult
from xcode.agent.events import AgentEvent
from xcode.agent.messages import AssistantMessage, ToolResultMessage, UserMessage
from xcode.agent.protocols import AgentToolResult, ToolExecutionMode
from xcode.ai.events import (
    FinalMessage,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.agent.types import (
    TextContent,
    ThinkingContent,
    ToolCallContent,
)


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
        yield TextDelta(chunk="ok")


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
            yield ToolCallEvent(calls=[ToolCall(id="call-1", name="echo", input={"text": "hello"})])
            return
        yield TextDelta(chunk="done")


class EchoTool:
    name = "echo"
    label = "Echo"
    description = "Echo text."
    parameters = {"type": "object"}
    execution_mode: ToolExecutionMode | None = "sequential"
    examples: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.seen_signal: Any | None = None

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update=None,
    ) -> AgentToolResult:
        self.seen_signal = signal
        return AgentToolResult(content=[TextContent(text=str(params["text"]))])


class BuiltinShellTool:
    """模拟带 Responses builtin shell 元数据的 AgentTool。"""

    name = "shell"
    label = "Shell"
    description = "Run shell commands."
    parameters = {"type": "object"}
    execution_mode: ToolExecutionMode | None = "sequential"
    examples: list[dict[str, Any]] = []
    builtin: dict[str, Any] = {"type": "shell", "environment": {"type": "local"}}

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update=None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="ok")])


class AgentLoopContractTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_builtin_tool_metadata_reaches_provider(self) -> None:
        """AgentTool builtin 元数据会传递到 provider 工具定义。"""
        provider = TextProvider()
        tool = BuiltinShellTool()

        await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=AgentContext(tools=[tool]),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
            ),
            emit=lambda _event: None,
        )

        assert provider.tools is not None
        self.assertEqual(provider.tools[0].name, "shell")
        self.assertEqual(provider.tools[0].builtin, tool.builtin)

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

    def test_tool_result_converts_to_plain_tool_message(self) -> None:
        """工具结果在 provider 边界使用 OpenAI 兼容消息。"""
        converted = convert_to_llm(
            [
                ToolResultMessage(
                    tool_call_id="call-1",
                    content="output",
                    is_error=False,
                )
            ]
        )

        self.assertEqual(
            converted,
            [
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "output",
                }
            ],
        )

    def test_assistant_thinking_content_becomes_reasoning_content(self) -> None:
        """思考块在 provider 边界保留为 reasoning_content。"""
        converted = convert_to_llm(
            [
                AssistantMessage(
                    content=[
                        ThinkingContent(thinking="think"),
                        TextContent(text="answer"),
                    ]
                )
            ]
        )

        self.assertEqual(converted[0]["content"], "answer")
        self.assertEqual(converted[0]["reasoning_content"], "think")

    def test_tool_call_arguments_convert_to_json_string(self) -> None:
        """工具调用参数在 provider 边界转为 JSON 字符串。"""
        converted = convert_to_llm(
            [
                AssistantMessage(
                    content=[
                        ToolCallContent(
                            id="call-1",
                            name="echo",
                            arguments={"text": "hello"},
                        )
                    ]
                )
            ]
        )

        arguments = converted[0]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(arguments, str)
        self.assertEqual(arguments, '{"text":"hello"}')


# ── 新增功能测试 ──


class StepLimitProvider:
    """总是返回工具调用，用于测试步数限制。"""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self, messages, tools, options: StreamOptions | None = None, **kwargs: Any
    ):
        self.calls += 1
        yield ToolCallEvent(calls=[ToolCall(f"call-{self.calls}", "noop", {})])


class NoopTool:
    name = "noop"
    label = "Noop"
    description = "Does nothing."
    parameters = {"type": "object"}
    execution_mode: ToolExecutionMode | None = None
    examples: list[dict[str, Any]] = []

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])


class ErrorProvider:
    """前 N 次调用抛异常，之后返回正常文本。"""

    def __init__(self, fail_count: int = 1, error: Exception | None = None) -> None:
        self.fail_count = fail_count
        self.calls = 0
        self.error = error or RuntimeError("transient error: rate limit")

    async def stream(
        self, messages, tools, options: StreamOptions | None = None, **kwargs: Any
    ):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise self.error
        yield TextDelta(chunk="recovered")


class MaxTokensProvider:
    """前 N 次返回 max_tokens，之后返回正常文本。"""

    def __init__(self, max_tokens_count: int = 1) -> None:
        self.max_tokens_count = max_tokens_count
        self.calls = 0

    async def stream(
        self, messages, tools, options: StreamOptions | None = None, **kwargs: Any
    ):
        self.calls += 1
        if self.calls <= self.max_tokens_count:
            yield TextDelta(chunk="partial" * 100)
            yield FinalMessage("", stop_reason="max_tokens")
        else:
            yield TextDelta(chunk="final")


class RepeatedToolProvider:
    """始终返回相同的工具调用，用于测试重复工具看门狗。"""

    async def stream(
        self, messages, tools, options: StreamOptions | None = None, **kwargs: Any
    ):
        yield ToolCallEvent(calls=[ToolCall(id="same-call", name="echo", input={"text": "hi"})])


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
            examples: list[dict[str, Any]] = []

            async def execute(self, tool_call_id, params, signal=None, on_update=None):
                raise RuntimeError("tool failed")

        class FailToolProvider:
            def __init__(self):
                self.call_count = 0

            async def stream(
                self,
                messages,
                tools,
                options: StreamOptions | None = None,
                **kwargs: Any,
            ):
                self.call_count += 1
                yield ToolCallEvent(calls=[ToolCall(f"call-{self.call_count}", "fail", {})])

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
