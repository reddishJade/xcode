from __future__ import annotations

from typing import Any
from xcode.agent.agent_loop import run_agent_loop
from xcode.agent.message_converter import convert_to_llm
from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.results import AgentLoopResult, TerminationReason
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
import pytest
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

class CancelledSignal:
    """始终处于取消状态的测试信号。"""

    reason = "user cancelled"

    def is_cancelled(self) -> bool:
        """返回固定取消状态。"""
        return True

class ErrorTextProvider:
    def __init__(self) -> None:
        self.messages: list[Message] | None = None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: Any,
    ):
        self.messages = messages
        yield FinalMessage(
            content="Provider error: boom",
            stop_reason="error",
        )

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
            yield ToolCallEvent(
                calls=[ToolCall(id="call-1", name="echo", input={"text": "hello"})]
            )
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

class AgentLoopContractTests:
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

        assert provider.messages == [{"role": "user", "content": "hello"}]
        assert provider.tools == []
        final = result.messages[-1]
        assert isinstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        assert final.content == [TextContent(text="ok")]
        assert events[-1].type == "agent_end"

    async def test_provider_error_text_is_preserved(self) -> None:
        provider = ErrorTextProvider()

        result = await run_agent_loop(
            prompts=[UserMessage(content="hello")],
            context=AgentContext(),
            config=AgentLoopConfig(
                provider=provider,
                convert_to_llm=convert_to_llm,
                max_step_retries=0,
            ),
            emit=lambda _event: None,
        )

        final = result.messages[-1]
        assert isinstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        assert final.content == [TextContent(text="Provider error: boom")]
        assert final.error_message == "Provider error: boom"

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
        assert provider.tools[0].name == "shell"
        assert provider.tools[0].builtin == tool.builtin

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

        assert provider.calls == 2
        assert tool.seen_signal is None
        assert any(isinstance(msg, ToolResultMessage) for msg in result.messages)
        assert provider.messages[-1][-1]["role"] == "tool"
        final = result.messages[-1]
        assert isinstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        assert final.content == [TextContent(text="done")]

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

        assert converted == [
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "output",
                }
            ]

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

        assert converted[0]["content"] == "answer"
        assert converted[0]["reasoning_content"] == "think"

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
        assert isinstance(arguments, str)
        assert arguments == '{"text":"hello"}'

# ── 新增功能测试 ──

class StepLimitProvider:
    """总是返回工具调用，用于测试步数限制。"""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self, messages, tools, options: StreamOptions | None = None, **kwargs: Any
    ):
        self.calls += 1
        yield ToolCallEvent(
            calls=[ToolCall(id=f"call-{self.calls}", name="noop", input={})]
        )

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
            yield FinalMessage(content="", stop_reason="max_tokens")
        else:
            yield TextDelta(chunk="final")

class RepeatedToolProvider:
    """始终返回相同的工具调用，用于测试重复工具看门狗。"""

    async def stream(
        self, messages, tools, options: StreamOptions | None = None, **kwargs: Any
    ):
        yield ToolCallEvent(
            calls=[ToolCall(id="same-call", name="echo", input={"text": "hi"})]
        )

class AgentLoopFeatureTests:
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

        assert isinstance(result, AgentLoopResult)
        assert result.stopped_by_limit
        assert result.termination_reason == TerminationReason.STEP_LIMIT
        assert result.steps == 3

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

        assert provider.calls == 2
        final = result.messages[-1]
        assert isinstance(final, AssistantMessage)
        assert isinstance(final, AssistantMessage)
        assert final.content == [TextContent(text="recovered")]

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
        assert provider.calls > 1

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
        assert provider.calls == 2
        final = result.messages[-1]
        assert isinstance(final, AssistantMessage)

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

        assert result.stopped_by_watchdog
        assert result.termination_reason == TerminationReason.WATCHDOG
        assert result.watchdog_reason is not None
        assert result.watchdog_reason is not None
        assert "repeated" in result.watchdog_reason.lower()

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
                yield ToolCallEvent(
                    calls=[
                        ToolCall(id=f"call-{self.call_count}", name="fail", input={})
                    ]
                )

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

        assert result.stopped_by_watchdog
        assert result.termination_reason == TerminationReason.WATCHDOG
        assert result.watchdog_reason is not None
        assert result.watchdog_reason is not None
        assert "consecutive steps" in result.watchdog_reason.lower()

    async def test_cancelled_loop_has_structured_termination_reason(self) -> None:
        result = await run_agent_loop(
            [UserMessage(content="stop")],
            AgentContext(),
            AgentLoopConfig(provider=TextProvider()),
            lambda _event: None,
            signal=CancelledSignal(),
        )
        assert result.termination_reason == TerminationReason.CANCELLED
        assert result.error_detail == "user cancelled"

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

        assert compact_called

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

        assert result.metrics is not None
        assert result.metrics is not None
        assert result.metrics.steps >= 1
        assert result.metrics.llm_calls > 0
        assert result.metrics.tool_calls > 0

if __name__ == "__main__":
    pytest.main()
