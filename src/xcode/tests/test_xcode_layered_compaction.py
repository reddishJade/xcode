from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
from xcode.harness.agent_runtime.compaction import (
    CompactController,
    LayeredCompactor,
    build_compact_tool,
    estimate_message_tokens,
    micro_compact_tool_results,
    summarize_inactive_branches,
)
from xcode.harness.config import AgentConfig, RequestHygieneConfig
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.harness.agent_runtime.agent_helpers import budget_messages_for_provider

from xcode.tests.fixtures import FakeProvider
from xcode.ai.events import (
    TextDelta,
    FinalMessage,
    ToolCallEvent,
    ToolCall,
    UsageUpdate,
)
from xcode.agent.messages import AssistantMessage, ToolResultMessage
from xcode.agent.types import ToolCallContent
import pytest
class XcodeLayeredCompactionTests:
    def test_old_tool_results_are_micro_compacted(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "start"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "a" * 200}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "b" * 200}],
            },
        ]

        compacted = micro_compact_tool_results(
            messages, keep_recent=1, max_content_chars=10
        )

        assert "compacted" in compacted[1]["content"][0]["content"]
        assert compacted[2]["content"][0]["content"] == "b" * 200

    def test_layered_compactor_saves_transcript_and_keeps_recent_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            messages = [{"role": "user", "content": "root"}]
            for index in range(6):
                messages.append({"role": "assistant", "content": f"old {index}"})
            compactor = LayeredCompactor(Path(tmp), max_recent_messages=2)

            compacted = compactor(messages)

            assert list(Path(tmp).glob("transcript_*.jsonl"))
            assert "[Compressed]" in compacted[1]["content"]
            assert compacted[-1]["content"] == "old 5"

    def test_compaction_preserves_activated_skill_tool_pair(self) -> None:
        """压缩保留已激活技能的 tool_use 与完整结果。"""
        activation = (
            '<skill name="review" root="C:/skills/review" activated="true">\n'
            '<skill-activation-state>{"name": "review"}</skill-activation-state>\n'
            "FULL_SKILL_BODY\n"
            "</skill>"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "root"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "skill-1",
                        "name": "load_skill",
                        "input": {"name": "review"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "skill-1",
                        "content": activation,
                    }
                ],
            },
            {"role": "user", "content": "later one"},
            {"role": "assistant", "content": "later two"},
            {"role": "user", "content": "latest"},
        ]
        compactor = LayeredCompactor(
            max_recent_messages=2,
            keep_recent_tool_results=0,
            max_tool_result_chars=10,
        )

        compacted = compactor(messages)
        rendered = str(compacted)

        assert "load_skill" in rendered
        assert "skill-1" in rendered
        assert "FULL_SKILL_BODY" in rendered

    def test_summarize_inactive_branches_replaces_branch_run(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "root"},
            {
                "role": "user",
                "content": "inactive one",
                "metadata": {"branch_id": "branch-a"},
            },
            {
                "role": "assistant",
                "content": "inactive two",
                "metadata": {"branch_id": "branch-a"},
            },
            {
                "role": "user",
                "content": "active",
                "metadata": {"branch_id": "branch-b", "active_branch": True},
            },
        ]

        compacted = summarize_inactive_branches(
            messages,
            active_branch_id="branch-b",
            compact_token_threshold=1,
            budget_trigger_token_ratio=0,
            summarize_fn=lambda branch_messages: "branch-a summary",
        )

        assert len(compacted) == 3
        summary = compacted[1]
        assert summary["metadata"]["type"] == "branch_summary"
        assert summary["metadata"]["branch_id"] == "branch-a"
        assert "branch-a summary" in summary["content"][0]["text"]
        assert compacted[2]["content"] == "active"

    def test_summarize_inactive_branches_waits_for_token_pressure(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "root"},
            {
                "role": "user",
                "content": "inactive",
                "metadata": {"branch_id": "branch-a"},
            },
        ]

        compacted = summarize_inactive_branches(
            messages,
            compact_token_threshold=100_000,
            budget_trigger_token_ratio=1,
            summarize_fn=lambda _branch_messages: "unused",
        )

        assert compacted == messages

    def test_manual_compact_tool_triggers_structured_agent_compaction(self) -> None:
        controller = CompactController()
        compact_tool = build_compact_tool(controller)
        responses = iter(
            [
                [
                    ToolCallEvent(calls=[ToolCall(id="c", name="compact", input={})]),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        seen_lengths: list[int] = []

        def factory(messages, _tools):
            seen_lengths.append(len(messages))
            return next(responses)

        provider = FakeProvider(factory)
        agent = StructuredAgent(
            provider=provider,
            registry=(compact_tool,),
            config=AgentConfig(max_steps=3),
            runtime=AgentRuntimeConfig(
                compactor=LayeredCompactor(max_recent_messages=1),
                compact_controller=controller,
            ),
        )

        result = agent.run("work")

        assert result.answer == "done"
        assert seen_lengths[-1] <= 3

    def test_structured_agent_compacts_on_token_threshold(self) -> None:
        responses = iter(
            [
                [
                    ToolCallEvent(calls=[ToolCall(id="c", name="compact", input={})]),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        seen_tokens: list[int] = []

        def factory(messages, _tools):
            seen_tokens.append(estimate_message_tokens(messages))
            return next(responses)

        provider = FakeProvider(factory)
        agent = StructuredAgent(
            provider=provider,
            registry=(build_compact_tool(CompactController()),),
            config=AgentConfig(max_steps=3, compact_token_threshold=1),
            runtime=AgentRuntimeConfig(
                compactor=LayeredCompactor(max_recent_messages=1),
            ),
        )

        result = agent.run("work " + ("long " * 40))

        assert result.answer == "done"
        assert seen_tokens[-1] < seen_tokens[0] + 100

    def test_structured_agent_sends_compacted_messages_to_provider(self) -> None:
        seen_messages: list[list[Any]] = []

        def compact(_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{"role": "user", "content": "[Compressed]\nkept facts"}]

        def factory(messages: list[Any], _tools: list[Any]) -> list[Any]:
            seen_messages.append(messages)
            return [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        provider = FakeProvider(factory)
        agent = StructuredAgent(
            provider=provider,
            registry=(),
            config=AgentConfig(max_steps=1, compact_token_threshold=1),
            runtime=AgentRuntimeConfig(compactor=compact),
        )

        result = agent.run("work " + ("long " * 40))

        assert result.answer == "done"
        assert len(seen_messages) == 1
        assert seen_messages[0] == [{"role": "user", "content": "[Compressed]\nkept facts"}]

    def test_structured_agent_applies_request_hygiene_only_to_provider_request(
        self,
    ) -> None:
        large_content = "\n".join(f"line {index}" for index in range(200))
        seen_messages: list[list[Any]] = []

        def factory(messages: list[Any], _tools: list[Any]) -> list[Any]:
            seen_messages.append(messages)
            return [
                TextDelta(chunk="done"),
                FinalMessage(content="", stop_reason="end_turn"),
            ]

        provider = FakeProvider(factory)
        agent = StructuredAgent(
            provider=provider,
            registry=(),
            config=AgentConfig(max_steps=1),
            runtime=AgentRuntimeConfig(
                request_hygiene=RequestHygieneConfig(
                    keep_head_lines=5,
                    keep_tail_lines=5,
                ),
            ),
        )
        agent.load_history(
            [
                AssistantMessage(
                    content=[
                        ToolCallContent(
                            id="call_1",
                            name="bash",
                            arguments={},
                        )
                    ]
                ),
                ToolResultMessage(
                    tool_call_id="call_1",
                    tool_name="bash",
                    content=large_content,
                ),
            ]
        )

        result = agent.run("continue")

        assert result.answer == "done"
        tool_message = next(
            message for message in seen_messages[0] if message.get("role") == "tool"
        )
        assert "omitted" in tool_message["content"]
        assert len(tool_message["content"].splitlines()) < 200

        history = agent.history_messages()
        history_result = history[1]
        assert isinstance(history_result, ToolResultMessage)
        assert history_result.content == large_content

    def test_structured_agent_compacts_after_high_provider_prompt_tokens(self) -> None:
        seen_messages: list[list[Any]] = []
        responses = iter(
            [
                [
                    UsageUpdate(input_tokens=40_000, output_tokens=10),
                    TextDelta(chunk="first"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="second"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )

        def compact(_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{"role": "user", "content": "[Compressed]\nfrom real tokens"}]

        def factory(messages: list[Any], _tools: list[Any]) -> list[Any]:
            seen_messages.append(messages)
            return next(responses)

        provider = FakeProvider(factory)
        agent = StructuredAgent(
            provider=provider,
            registry=(),
            config=AgentConfig(max_steps=1),
            runtime=AgentRuntimeConfig(compactor=compact),
        )

        first = agent.run("first")
        second = agent.run("second")

        assert first.answer == "first"
        assert second.answer == "second"
        assert seen_messages[1] == [{"role": "user", "content": "[Compressed]\nfrom real tokens"}]

    def test_estimate_message_tokens(self) -> None:
        messages = [
            {"role": "system", "content": "hello world"},
            {"role": "user", "content": "foo bar"},
        ]
        tokens = estimate_message_tokens(messages)
        assert tokens > 0

        # Test estimating tokens for empty lists or simple formats
        assert estimate_message_tokens([]) == 0

    def test_estimate_text_tokens(self) -> None:
        from xcode.harness.agent_runtime.compaction import estimate_text_tokens

        # If tiktoken is installed, empty string produces 0 tokens; fallback produces 1.
        assert estimate_text_tokens("") in (0, 1)
        assert estimate_text_tokens("hello world") > 0

    def test_provider_budget_truncates_large_non_read_tool_results(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "grep_1",
                        "name": "grep_search",
                        "input": {"pattern": "skill"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "grep_1",
                        "content": "a" * 20_000,
                    }
                ],
            },
        ]

        budgeted = budget_messages_for_provider(messages)

        content = budgeted[2]["content"][0]["content"]
        assert len(content) < 10_000
        assert "truncated" in content

    def test_edit_file_works_after_compaction(self) -> None:
        from xcode.coding_agent.tools.file import build_file_tools
        from xcode.harness.agent_runtime.compaction import LayeredCompactor
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "test_compact.txt"
            file_path.write_text("original content for compact", encoding="utf-8")

            tools = build_file_tools(root)
            edit_tool = next(t for t in tools if t.name == "edit_file")

            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "start"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_2",
                            "name": "read_file",
                            "input": {"path": "test_compact.txt"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_2",
                            "content": "original content for compact",
                            "status": "ok",
                        }
                    ],
                },
            ]

            compactor = LayeredCompactor(Path(tmp), max_recent_messages=1)
            compactor(messages)

            # edit_file 不再依赖 read_versions 缓存，压缩后直接基于 old_text 匹配即可工作
            edit_res = edit_tool.handler(
                {
                    "path": "test_compact.txt",
                    "old_text": "original content",
                    "new_text": "new",
                }
            )
            assert "replacements=1" in edit_res
            assert file_path.read_text(encoding="utf-8") == "new for compact"

if __name__ == "__main__":
    pytest.main()
