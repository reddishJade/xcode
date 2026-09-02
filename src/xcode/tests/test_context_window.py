"""上下文窗口切换与活动工作集单元测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from xcode.harness.agent_runtime.context_window import (
    ContextWindowController,
    ContextWindowRollover,
    _compute_recent_count_from_tokens,
    _find_turn_boundary,
    budget_large_tool_outputs,
    build_new_context_tool,
    latest_read_file_tool_result_ids,
    trim_old_tool_results,
)


def _message(role: str, content: object = "text") -> dict[str, object]:
    return {"role": role, "content": content}


def test_rollover_uses_latest_user_turn_without_summary() -> None:
    messages = [
        _message("system", "system"),
        _message("user", "old goal"),
        _message("assistant", "old result"),
        _message("user", "current action"),
        _message("assistant", "current work"),
    ]

    window = ContextWindowRollover()(messages)
    rendered = "\n".join(str(message["content"]) for message in window)

    assert "old goal" not in rendered
    assert "old result" not in rendered
    assert "current action" in rendered
    assert "current work" in rendered
    assert "without a summary" in rendered
    assert "[Compressed]" not in rendered


def test_rollover_does_not_mutate_source_messages() -> None:
    rollover = ContextWindowRollover(keep_recent_tool_results=0)
    messages = [
        _message("user", "current"),
        _message(
            "tool",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "x" * 500,
                }
            ],
        ),
    ]
    original = deepcopy(messages)

    rollover(messages)

    assert messages == original


def test_rollover_replaces_previous_reset_notice() -> None:
    rollover = ContextWindowRollover()
    first = rollover([_message("system", "system"), _message("user", "first")])
    second = rollover([*first, _message("user", "second")])
    notices = [
        str(message["content"])
        for message in second
        if "<context-window-reset" in str(message["content"])
    ]

    assert len(notices) == 1
    assert "second" in str(second[-1]["content"])


def test_manual_idle_rollover_starts_without_previous_turn() -> None:
    window = ContextWindowRollover()(
        [
            _message("system", "system"),
            _message("user", "finished task"),
            _message("assistant", "finished answer"),
        ],
        preserve_active_turn=False,
    )

    rendered = "\n".join(str(message["content"]) for message in window)
    assert "system" in rendered
    assert "finished task" not in rendered
    assert "finished answer" not in rendered


def test_controller_consumes_reason_once() -> None:
    controller = ContextWindowController()
    controller.request("model")
    assert controller.consume() == "model"
    assert controller.consume() is None


def test_new_context_tool_requires_explicit_working_note(tmp_path: Path) -> None:
    controller = ContextWindowController()
    (tool,) = build_new_context_tool(controller, tmp_path)

    result = tool.handler({"reason": "window is noisy"})

    assert "Write NOTE.md" in result
    assert controller.consume() is None


def test_new_context_tool_schedules_model_rollover(tmp_path: Path) -> None:
    (tmp_path / "NOTE.md").write_text("Next: run tests.\n", encoding="utf-8")
    controller = ContextWindowController()
    (tool,) = build_new_context_tool(controller, tmp_path)

    result = tool.handler({"reason": "window is noisy"})

    assert "No summary" in result
    assert controller.consume() == "model"


def test_compute_recent_count_is_safe_for_empty_input() -> None:
    assert _compute_recent_count_from_tokens([], 100) == 1


def test_find_turn_boundary_moves_tool_result_to_assistant() -> None:
    messages = [
        _message("system"),
        _message("user"),
        _message("assistant"),
        _message("tool"),
    ]
    assert _find_turn_boundary(messages, 3) == 2


def test_openai_tool_calls_are_tracked_and_latest_read_is_preserved() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"src/main.py"}',
                    },
                }
            ],
        },
        _message(
            "tool",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "source",
                }
            ],
        ),
    ]

    assert latest_read_file_tool_result_ids(messages) == {"call-1"}


def test_openai_skill_activation_becomes_explicit_startup_context() -> None:
    messages = [
        _message("system", "system"),
        _message(
            "tool",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "skill-1",
                    "content": (
                        '<skill-activation-state>{"name":"review"}'
                        "</skill-activation-state>\nSkill instructions"
                    ),
                }
            ],
        ),
        _message("user", "continue"),
    ]

    window = ContextWindowRollover()(messages)

    activation_contexts = [
        message
        for message in window
        if message.get("role") == "user"
        and "skill-activation-state" in str(message.get("content", ""))
    ]
    assert len(activation_contexts) == 1
    assert not any(message.get("role") == "tool" for message in window)


def test_tool_output_budget_retrieval_marker() -> None:
    messages = [
        _message(
            "tool",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "a" * 100,
                }
            ],
        )
    ]

    window = budget_large_tool_outputs(
        messages,
        large_tool_output_chars=20,
        large_tool_output_head_chars=5,
        large_tool_output_tail_chars=5,
        active_window_token_threshold=1,
        tool_trim_trigger_ratio=0,
    )

    assert "retrieve exact output from history" in str(window[0]["content"])


def test_old_tool_result_trim_only_changes_copy() -> None:
    messages = [
        _message(
            "tool",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "a" * 100,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call-2",
                    "content": "latest",
                },
            ],
        )
    ]

    window = trim_old_tool_results(messages, keep_recent=1, max_content_chars=10)

    assert "available through history" in str(window[0]["content"])
    assert "a" * 100 in str(messages[0]["content"])
