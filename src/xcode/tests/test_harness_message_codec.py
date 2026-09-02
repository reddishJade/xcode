"""Provider dict ↔ AgentMessage 编解码单元测试。"""

from __future__ import annotations

from xcode.agent.messages import SystemMessage, UserMessage
from xcode.harness.agent_runtime.message_codec import (
    _content_to_text,
    _tool_call_from_provider,
    messages_from_provider_dicts,
)


class TestMessagesFromProviderDicts:
    def test_system_message(self) -> None:
        result = messages_from_provider_dicts(
            [{"role": "system", "content": "Be helpful"}]
        )
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)

    def test_user_message(self) -> None:
        result = messages_from_provider_dicts([{"role": "user", "content": "hello"}])
        assert len(result) == 1
        assert isinstance(result[0], UserMessage)

    def test_assistant_with_tool_calls(self) -> None:
        result = messages_from_provider_dicts(
            [
                {
                    "role": "assistant",
                    "content": "I'll search",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "search", "arguments": '{"q": "x"}'},
                        }
                    ],
                }
            ]
        )
        assert len(result) == 1
        msg = result[0]
        assert len(msg.content) == 2  # text + tool call

    def test_unknown_role_returns_none(self) -> None:
        result = messages_from_provider_dicts([{"role": "unknown", "content": "x"}])
        assert len(result) == 0


class TestToolCallFromProvider:
    def test_valid(self) -> None:
        item = {"id": "c1", "function": {"name": "search", "arguments": '{"q": "x"}'}}
        result = _tool_call_from_provider(item)
        assert result is not None
        assert result.name == "search"
        assert result.arguments == {"q": "x"}

    def test_string_args_decoded(self) -> None:
        item = {"id": "c1", "function": {"name": "search", "arguments": '{"q": "x"}'}}
        result = _tool_call_from_provider(item)
        assert result is not None
        assert isinstance(result.arguments, dict)

    def test_dict_args_passthrough(self) -> None:
        item = {"id": "c1", "function": {"name": "search", "arguments": {"q": "x"}}}
        result = _tool_call_from_provider(item)
        assert result is not None
        assert result.arguments == {"q": "x"}

    def test_missing_id_returns_none(self) -> None:
        item = {"function": {"name": "search"}}
        assert _tool_call_from_provider(item) is None

    def test_missing_name_returns_none(self) -> None:
        item = {"id": "c1", "function": {"arguments": "{}"}}
        assert _tool_call_from_provider(item) is None

    def test_not_dict_returns_none(self) -> None:
        assert _tool_call_from_provider("string") is None


class TestContentToText:
    def test_none(self) -> None:
        assert _content_to_text(None) == ""

    def test_string(self) -> None:
        assert _content_to_text("hello") == "hello"

    def test_list_with_text_blocks(self) -> None:
        content = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
        assert _content_to_text(content) == "HelloWorld"

    def test_list_with_tool_result(self) -> None:
        content = [{"type": "tool_result", "content": "output"}]
        assert _content_to_text(content) == "output"
