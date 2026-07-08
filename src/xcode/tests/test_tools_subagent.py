"""subagent 工具输入解析单元测试。"""

from __future__ import annotations

from xcode.coding_agent.tools.subagent import _max_concurrent, _parse_tasks


def test_parse_single_prompt() -> None:
    parsed = _parse_tasks({"description": "scan auth", "prompt": "Inspect auth"})
    assert not isinstance(parsed, str)
    assert parsed == [
        {
            "description": "scan auth",
            "prompt": "Inspect auth",
            "subagent_type": "coding",
        }
    ]


def test_parse_parallel_tasks() -> None:
    parsed = _parse_tasks(
        {
            "subagent_type": "research",
            "tasks": [
                {"description": "auth", "prompt": "Inspect auth"},
                {
                    "description": "db",
                    "prompt": "Inspect db",
                    "subagent_type": "coding",
                },
            ],
        }
    )
    assert not isinstance(parsed, str)
    assert [task["subagent_type"] for task in parsed] == ["research", "coding"]


def test_parse_tasks_requires_prompt() -> None:
    assert (
        _parse_tasks({"tasks": [{"description": "missing"}]})
        == "Error: task 1 prompt is required"
    )


def test_max_concurrent_clamps() -> None:
    assert _max_concurrent(0) == 1
    assert _max_concurrent(99) == 16
    assert _max_concurrent("nope") == 4


def test_bounded_prompt_adds_summary_constraint() -> None:
    from xcode.coding_agent.tools.subagent import _bounded_prompt

    prompt = _bounded_prompt("Inspect auth")
    assert "Inspect auth" in prompt
    assert "return a concise summary" in prompt


def test_subagent_batch_preview() -> None:
    from xcode.cli.repl_tools import brief_input

    preview = brief_input(
        "subagent",
        {
            "tasks": [
                {"description": "tools"},
                {"description": "runtime"},
                {"description": "cli"},
            ]
        },
    )
    assert preview == "subagent tasks (3)"


def test_subagent_updates_keep_numbered_slots() -> None:
    import io

    from rich.console import Console

    from xcode.cli.commands import ReplState
    from xcode.cli.repl_turn_handler import ToolCallHandler
    from xcode.harness.agent_runtime.events import ToolUpdateData

    output = io.StringIO()
    handler = ToolCallHandler(
        ReplState(), Console(file=output, force_terminal=False, width=120)
    )

    handler.handle_tool_update(
        ToolUpdateData(
            tool_call_id="sub-1",
            tool_name="subagent",
            partial_result="[1] → tools\n[2] → runtime\n[3] → cli",
        )
    )
    handler.handle_tool_update(
        ToolUpdateData(
            tool_call_id="sub-1",
            tool_name="subagent",
            partial_result="[2]   → read src/xcode/coding_agent/tools/subagent.py",
        )
    )
    assert (
        handler._subagent_slots[2]["tool"]
        == "→ read src/xcode/coding_agent/tools/subagent.py"
    )
    handler.handle_tool_update(
        ToolUpdateData(
            tool_call_id="sub-1",
            tool_name="subagent",
            partial_result="[2] ✓ runtime",
        )
    )
    assert handler._subagent_slots == {
        1: {"task": "→ tools", "tool": ""},
        2: {"task": "✓ runtime", "tool": ""},
        3: {"task": "→ cli", "tool": ""},
    }


def test_subagent_tool_call_text_is_multiline_list() -> None:
    from xcode.cli.repl_tools import tool_call_text

    rendered = tool_call_text(
        "subagent",
        "subagent tasks (2)",
        {
            "tasks": [
                {"description": "tools", "subagent_type": "coding"},
                {"description": "runtime", "subagent_type": "research"},
            ]
        },
    )

    assert rendered.plain.splitlines() == [
        "  → Subagent tasks (2)",
        "    [1] tools [coding]",
        "    [2] runtime [research]",
    ]
