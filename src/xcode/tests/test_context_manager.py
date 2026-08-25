"""会话级 ContextManager 单元测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from xcode.agent.agent import Agent
from xcode.agent.config import AgentLoopConfig
from xcode.agent.context_manager import ContextManager
from xcode.agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.request import RequestAssembly
from xcode.agent.types import TextContent, ToolCallContent
from xcode.ai.events import FinalMessage, ProviderEvent, TextDelta, UsageUpdate
from xcode.ai.types import StreamOptions, ToolDefinition


class _Provider:
    @property
    def model(self) -> str:
        return "test-model"

    async def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **_kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        del messages, tools, options
        yield TextDelta("answer")
        yield UsageUpdate(input_tokens=11, output_tokens=3)
        yield FinalMessage(content="", stop_reason="end_turn")


def test_replace_history_repairs_tool_pairs_and_tracks_version() -> None:
    manager = ContextManager()
    assistant = AssistantMessage(
        content=[
            TextContent(text="before"),
            ToolCallContent(id="keep", name="read", arguments={}),
            ToolCallContent(id="drop", name="read", arguments={}),
        ]
    )
    messages = [
        UserMessage(content="question"),
        assistant,
        ToolResultMessage(tool_call_id="keep", tool_name="read", content="ok"),
        ToolResultMessage(tool_call_id="orphan", tool_name="read", content="no"),
    ]

    result = manager.replace_history(messages)

    assert manager.history_version == 1
    assert len(result) == 3
    assert isinstance(result[1], AssistantMessage)
    assert [
        block.id for block in result[1].content if isinstance(block, ToolCallContent)
    ] == ["keep"]
    assert isinstance(result[2], ToolResultMessage)
    assert result[2].tool_call_id == "keep"
    assert manager.token_usage.estimated_prompt_tokens > 0


def test_record_request_and_provider_usage_updates_session_metadata() -> None:
    manager = ContextManager()
    assembly = RequestAssembly(
        messages=(SystemMessage(content="system"), UserMessage(content="question")),
        wire_messages=(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ),
        tools=(ToolDefinition(name="read", description="Read", parameters={}),),
        context_trace=(),
        current_step=1,
        hygiene_applied=True,
        estimated_tokens=17,
        token_budget=100,
        budget_remaining=83,
    )

    manager.record_request(assembly)
    manager.record_request(assembly)
    manager.record_provider_usage(
        {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "cached_tokens": 2,
            "ignored": "not numeric",
        }
    )
    manager.record_provider_usage(
        {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
    )

    assert manager.token_usage.estimated_prompt_tokens == 17
    assert manager.token_usage.context_budget == 100
    assert manager.token_usage.budget_remaining == 83
    assert manager.token_usage.prompt_tokens == 12
    assert manager.token_usage.completion_tokens == 7
    assert manager.token_usage.total_tokens == 19
    assert manager.token_usage.last_prompt_tokens == 7
    assert manager.provider_usage["cached_tokens"] == 2
    assert manager.prompt_cache.request_count == 2
    assert manager.prompt_cache.prompt_sha256
    assert manager.prompt_cache.request_sha256


def test_compaction_replaces_history_and_resets_context_baseline() -> None:
    manager = ContextManager()
    manager.replace_history([UserMessage(content="old")])
    manager.record_provider_usage({"prompt_tokens": 9})
    manager.context_state.persistent_messages.append(
        SystemMessage(content="dynamic context")
    )

    replacement = manager.complete_compaction(
        [UserMessage(content="summary")],
        reason="manual",
        before_messages=9,
    )

    assert replacement == [UserMessage(content="summary")]
    assert manager.history == replacement
    assert manager.context_state.persistent_messages == []
    assert manager.compaction.context_window_id == 1
    assert manager.compaction.compaction_count == 1
    assert manager.compaction.last_reason == "manual"
    assert manager.compaction.last_messages_before == 9
    assert manager.compaction.last_messages_after == 1
    assert manager.token_usage.last_prompt_tokens is None


def test_normalize_messages_returns_a_request_safe_projection() -> None:
    manager = ContextManager()
    messages = [
        AssistantMessage(
            content=[ToolCallContent(id="missing", name="read", arguments={})]
        ),
        UserMessage(content="continue"),
    ]

    normalized = manager.normalize_messages(messages)

    assert normalized == [UserMessage(content="continue")]
    assert messages[0].content


@pytest.mark.asyncio
async def test_agent_persists_result_and_usage_in_context_manager() -> None:
    manager = ContextManager()
    provider = _Provider()
    agent = Agent(tools=[], model=provider)

    result = await agent.run(
        [UserMessage(content="question")],
        AgentLoopConfig(provider=provider, max_steps=1),
        context_manager=manager,
    )

    assert manager.history == result.surface
    assert manager.token_usage.last_prompt_tokens == 11
    assert manager.provider_usage["prompt_tokens"] == 11
    assert manager.provider_usage["completion_tokens"] == 3
