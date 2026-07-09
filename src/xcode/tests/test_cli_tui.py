"""TUI 入口和状态渲染测试。"""

from __future__ import annotations

import threading

from typing import Any

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import DummyOutput

from xcode.agent.types import ToolSpec
from xcode.ai.events import ToolCall
from xcode.cli.tui import (
    _HitlRequest,
    _TuiState,
    _XcodeTui,
    _rendered_markdown_lines,
    _visible_lines,
)
from xcode.harness.agent_runtime.events import (
    FinalStructuredEvent,
    TextDeltaStructuredEvent,
    ToolResultBlock,
    ToolResultStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.harness.agent_runtime.result import CodingAgentHarnessResult
from xcode.main import parse_args


def test_parse_tui_command() -> None:
    args = parse_args(["tui"])
    assert args.command == "tui"


def test_tui_state_renders_fixed_subagent_slots() -> None:
    state = _TuiState()
    state.add_user("inspect files")

    assert state.record_subagent_update("[2] → runtime")
    assert state.record_subagent_update("[1] → tools")
    assert state.record_subagent_update("[3] → cli")
    assert state.record_subagent_update(
        "[2]   → read src/xcode/coding_agent/tools/subagent.py"
    )

    rendered = state.render()
    assert rendered.index("[1] → tools") < rendered.index("[2] → runtime")
    assert rendered.index("[2] → runtime") < rendered.index("[3] → cli")
    assert "read src/xcode/coding_agent/tools/subagent.py" in rendered


def test_tui_state_uses_firstcoder_like_message_blocks() -> None:
    state = _TuiState()
    state.add_user("hello")
    rendered = state.render()
    assert "Xcode TUI" not in rendered
    assert rendered.startswith("│ you\n│   > hello")


def test_tui_state_emits_styled_fragments() -> None:
    state = _TuiState()
    state.add_user("hello")
    state.current_answer = "hi"

    fragments = state.fragments()
    assert ("class:user", "│ you") in fragments
    assert ("class:assistant", "│ xcode") in fragments


def test_visible_lines_follow_tail_and_scrollback() -> None:
    lines = [f"line {i}" for i in range(8)]

    assert _visible_lines(lines, 3, 0) == ["line 5", "line 6", "line 7"]
    assert _visible_lines(lines, 3, 2) == ["line 3", "line 4", "line 5"]
    assert _visible_lines(lines, 20, 0) == lines


def test_markdown_lines_render_markdown_structure() -> None:
    rendered = "\n".join(_rendered_markdown_lines("# Title\n\n- item"))

    assert "Title" in rendered
    assert "item" in rendered


def test_tui_state_collapses_thinking_and_tools() -> None:
    state = _TuiState()
    state.thinking = "private reasoning"
    state.tool_events.append("│ tool read_file running\n│   read file")

    state.toggle_thinking()
    state.toggle_tools()
    rendered = state.render()

    assert "│ thinking" in rendered
    assert "│ tools collapsed" in rendered
    assert "private reasoning" not in rendered
    assert "read file" not in rendered


def test_tui_state_keeps_tool_summary_after_final() -> None:
    state = _TuiState()
    state.add_user("inspect tui")
    state.handle_event(
        ToolUseStructuredEvent(
            "tool_use",
            1,
            ToolCall("tool-1", "read_file", {"path": "src/xcode/cli/tui.py"}),
        )
    )
    state.handle_event(
        ToolResultStructuredEvent(
            "tool_result",
            1,
            ToolResultBlock("tool-1", "ok"),
        )
    )
    state.handle_event(TextDeltaStructuredEvent("text_delta", 1, "done"))
    state.handle_event(
        FinalStructuredEvent(
            "final",
            1,
            CodingAgentHarnessResult(
                answer="done",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )
    )

    rendered = state.render()
    assert "│ tool read_file running" in rendered
    assert "read src/xcode/.../tui.py" in rendered
    assert "│ tool read_file success" in rendered
    assert "│ xcode\n│   done" in rendered


def test_tui_constructs_with_input_focus(tmp_path) -> None:
    class _Token:
        def reset(self) -> None:
            pass

        def cancel(self, _reason: str) -> None:
            pass

    class _Agent:
        cancellation_token = _Token()

    class _App:
        agent = _Agent()

        def ask_stream(self, _text: str, mode: object = None) -> list[Any]:
            return []

    with create_pipe_input() as pipe_input:
        tui = _XcodeTui(_App(), tmp_path, input=pipe_input, output=DummyOutput())

    assert tui._application.layout.current_control == tui._input.control
    assert isinstance(tui._output_control, FormattedTextControl)
    bound_keys = {binding.keys[0] for binding in tui._bindings().bindings}
    assert Keys.ScrollUp in bound_keys
    assert Keys.ScrollDown in bound_keys
    assert "c-t" in bound_keys
    assert "c-o" in bound_keys
    assert "1" not in bound_keys
    assert "2" not in bound_keys
    assert "3" not in bound_keys
    assert "4" not in bound_keys


def test_tui_reuses_cli_command_registry(tmp_path) -> None:
    class _Token:
        def reset(self) -> None:
            pass

        def cancel(self, _reason: str) -> None:
            pass

    class _Agent:
        cancellation_token = _Token()
        permission_policy = None
        restricted_dirs: tuple[str, ...] = ()

    class _App:
        agent = _Agent()

        def ask_stream(self, _text: str, mode: object = None) -> list[Any]:
            return []

    with create_pipe_input() as pipe_input:
        tui = _XcodeTui(_App(), tmp_path, input=pipe_input, output=DummyOutput())

    tui._submit("/build")

    assert tui._repl_state.mode == "build"
    assert tui._state.mode == "build"
    assert "Build Mode enabled" in tui._state.render()


def test_tui_wires_hitl_approval_callback(tmp_path) -> None:
    class _Token:
        def reset(self) -> None:
            pass

        def cancel(self, _reason: str) -> None:
            pass

    class _Agent:
        cancellation_token = _Token()
        approval_callback = None

    class _App:
        agent = _Agent()

        def ask_stream(self, _text: str, mode: object = None) -> list[Any]:
            return []

    with create_pipe_input() as pipe_input:
        tui = _XcodeTui(_App(), tmp_path, input=pipe_input, output=DummyOutput())

    assert _App.agent.approval_callback == tui._approval_callback


def test_tui_hitl_callback_returns_selected_result(tmp_path) -> None:
    class _Token:
        def reset(self) -> None:
            pass

        def cancel(self, _reason: str) -> None:
            pass

    class _Agent:
        cancellation_token = _Token()
        approval_callback = None

    class _App:
        agent = _Agent()

        def ask_stream(self, _text: str, mode: object = None) -> list[Any]:
            return []

    with create_pipe_input() as pipe_input:
        tui = _XcodeTui(_App(), tmp_path, input=pipe_input, output=DummyOutput())

    result_box: list[object] = []
    tool = ToolSpec("bash", "", "", lambda _data, _update: "")

    def run_callback() -> None:
        result_box.append(tui._approval_callback(tool, {"command": "echo hi"}))

    thread = threading.Thread(target=run_callback)
    thread.start()
    assert tui._state.pending_hitl is not None
    rendered = tui._state.render()
    assert "Authorization" not in rendered
    assert "authorization request" in rendered
    assert "Command: echo hi" in rendered
    assert "Allow this session" in rendered
    assert "Always allow" in rendered
    tui._answer_hitl_text("Allow (once)")
    thread.join(timeout=1)

    assert result_box
    result = result_box[0]
    assert result.decision == "allow"
    assert result.scope == "once"
    assert tui._state.pending_hitl is None


def test_tui_hitl_session_and_permanent_choices(tmp_path) -> None:
    class _Token:
        def reset(self) -> None:
            pass

        def cancel(self, _reason: str) -> None:
            pass

    class _Agent:
        cancellation_token = _Token()
        approval_callback = None

    class _App:
        agent = _Agent()

        def ask_stream(self, _text: str, mode: object = None) -> list[Any]:
            return []

    with create_pipe_input() as pipe_input:
        tui = _XcodeTui(_App(), tmp_path, input=pipe_input, output=DummyOutput())

    request = _HitlRequest("bash", ["Tool: bash"], threading.Event())
    tui._state.pending_hitl = request
    tui._answer_hitl_text("Allow this session")
    assert request.result is not None
    assert request.result.scope == "session"
    assert tui._state.pending_hitl is None

    request = _HitlRequest("bash", ["Tool: bash"], threading.Event())
    tui._state.pending_hitl = request
    tui._answer_hitl_text("Always allow")
    assert request.result is not None
    assert request.result.scope == "permanent"
