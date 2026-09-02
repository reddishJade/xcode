"""Token 估算与上下文换窗触发单元测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from xcode.agent._context_window import (
    estimate_message_tokens,
    estimate_tokens,
    extract_prompt_tokens_from_usage,
    should_rollover_token_aware,
)
from xcode.agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from xcode.agent.types import TextContent, ThinkingContent, ToolCallContent
from xcode.harness.agent_runtime.config import _rollover_decision
from xcode.harness.config import AgentConfig


def test_estimate_tokens_handles_empty_and_non_empty_text() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("hello world") > 0


def test_estimate_message_tokens_covers_all_message_kinds() -> None:
    messages = [
        SystemMessage(content="system"),
        UserMessage(content="hello"),
        AssistantMessage(
            content=[
                TextContent(text="world"),
                ThinkingContent(thinking="reasoning"),
                ToolCallContent(id="c1", name="get", arguments={"key": "value"}),
            ],
            stop_reason="tool_use",
        ),
        ToolResultMessage(tool_call_id="c1", tool_name="get", content="result"),
    ]

    assert estimate_message_tokens(messages) > 0


def test_provider_usage_is_preferred_for_rollover_decision() -> None:
    assert should_rollover_token_aware([], last_prompt_tokens=32_000)
    assert not should_rollover_token_aware([], last_prompt_tokens=100)


def test_static_rollover_fallbacks() -> None:
    messages = [UserMessage(content="x") for _ in range(10)]

    assert should_rollover_token_aware(messages, message_threshold=5)
    assert should_rollover_token_aware(messages, token_threshold=1)
    assert not should_rollover_token_aware([])


def test_extract_prompt_tokens_from_usage() -> None:
    assert extract_prompt_tokens_from_usage(None) is None
    assert extract_prompt_tokens_from_usage({}) is None
    assert extract_prompt_tokens_from_usage({"prompt_tokens": 150}) == 150
    assert extract_prompt_tokens_from_usage({"prompt_tokens": "abc"}) is None


def test_runtime_rollover_uses_provider_context_window_override() -> None:
    composition = SimpleNamespace(
        config=AgentConfig(reserve_tokens=100, rollover_trigger_ratio=0.95)
    )
    provider = SimpleNamespace(model="gpt-5.5", context_window=1_000)

    def rollover(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages

    before = _rollover_decision(
        [], cast(Any, rollover), None, 899, cast(Any, composition), cast(Any, provider)
    )
    at_limit = _rollover_decision(
        [], cast(Any, rollover), None, 900, cast(Any, composition), cast(Any, provider)
    )

    assert before is None
    assert at_limit == "token_limit"


def test_runtime_rollover_estimates_tokens_when_usage_is_missing() -> None:
    composition = SimpleNamespace(
        config=AgentConfig(reserve_tokens=0, rollover_trigger_ratio=0.95)
    )
    provider = SimpleNamespace(model="gpt-5.5", context_window=10)

    result = _rollover_decision(
        [UserMessage(content="enough text to cross a tiny configured window")],
        cast(Any, lambda messages: messages),
        None,
        None,
        cast(Any, composition),
        cast(Any, provider),
    )

    assert result == "token_limit"
