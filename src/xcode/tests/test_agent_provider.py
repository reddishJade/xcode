"""Provider 交互逻辑单元测试：纯函数部分。"""

from __future__ import annotations

from xcode.ai.events import (
    FinalMessage,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    UsageUpdate,
)
from xcode.ai.types import ToolDefinition
from xcode.agent._provider import (
    _tools_to_definitions,
    _provider_events_to_response,
    _tool_call_content_blocks,
)
from xcode.agent.results import AgentLoopMetrics
from xcode.agent.types import AgentTool, ToolCallContent


class _MockAgentTool(AgentTool):
    def __init__(self, name: str, desc: str = "") -> None:
        self._name = name
        self._desc = desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._desc

    @property
    def parameters(self) -> dict[str, object]:
        return {"type": "object", "properties": {}}

    @property
    def execution_mode(self) -> str | None:
        return None

    @property
    def examples(self) -> list[dict[str, object]]:
        return []

    async def execute(self, *args, **kwargs) -> object:
        from xcode.agent.types import AgentToolResult, TextContent
        return AgentToolResult(content=[TextContent(text="ok")])


class TestToolsToDefinitions:
    def test_none_tools(self) -> None:
        assert _tools_to_definitions(None) == []

    def test_empty_tools(self) -> None:
        assert _tools_to_definitions([]) == []

    def test_single_tool(self) -> None:
        tool = _MockAgentTool("my_tool", "Does something")
        result = _tools_to_definitions([tool])
        assert len(result) == 1
        assert isinstance(result[0], ToolDefinition)
        assert result[0].name == "my_tool"

    def test_tool_with_examples_appends_to_description(self) -> None:
        class ToolWithExamples(_MockAgentTool):
            @property
            def examples(self) -> list[dict[str, object]]:
                return [{"name": "my_tool", "input": {"x": 1}, "output": "ok"}]

        tool = ToolWithExamples("my_tool", "Does something")
        result = _tools_to_definitions([tool])
        assert "Examples:" in result[0].description


class TestProviderEventsToResponse:
    def test_text_only(self) -> None:
        events = [TextDelta(chunk="Hello"), TextDelta(chunk=" World")]
        metrics = AgentLoopMetrics()
        response = _provider_events_to_response(events, metrics, lambda _e: None)
        assert "Hello World" in str(response.message.content[0])

    def test_tool_call(self) -> None:
        events = [
            ToolCallEvent(
                calls=[ToolCall(id="c1", name="search", input={"q": "x"})]
            ),
            FinalMessage(content="", stop_reason="tool_use"),
        ]
        metrics = AgentLoopMetrics()
        response = _provider_events_to_response(events, metrics, lambda _e: None)
        assert response.stop_reason == "tool_use"
        assert any(
            isinstance(b, ToolCallContent) and b.name == "search"
            for b in response.message.content
        )

    def test_usage_update(self) -> None:
        events = [
            TextDelta(chunk="Hi"),
            UsageUpdate(input_tokens=10, output_tokens=20),
        ]
        metrics = AgentLoopMetrics()
        response = _provider_events_to_response(events, metrics, lambda _e: None)
        assert response.message.usage is not None
        assert response.message.usage["prompt_tokens"] == 10

    def test_reasoning_content(self) -> None:
        events = [ReasoningDelta(chunk="thinking..."), TextDelta(chunk="Answer")]
        metrics = AgentLoopMetrics()
        response = _provider_events_to_response(events, metrics, lambda _e: None)
        assert response.message.reasoning_content == "thinking..."

    def test_final_content_when_no_text_deltas(self) -> None:
        events = [FinalMessage(content="Final text", stop_reason="end_turn")]
        metrics = AgentLoopMetrics()
        response = _provider_events_to_response(events, metrics, lambda _e: None)
        assert "Final text" in str(response.message.content[0])

    def test_empty_events(self) -> None:
        metrics = AgentLoopMetrics()
        response = _provider_events_to_response([], metrics, lambda _e: None)
        assert response.stop_reason == "end_turn"


class TestToolCallContentBlocks:
    def test_converts_tool_call_event(self) -> None:
        event = ToolCallEvent(
            calls=[ToolCall(id="c1", name="search", input={"q": "hello"})]
        )
        blocks = _tool_call_content_blocks(event)
        assert len(blocks) == 1
        assert blocks[0].name == "search"
        assert blocks[0].arguments == {"q": "hello"}
