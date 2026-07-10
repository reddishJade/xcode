"""TUI 入口和状态渲染测试。"""

from __future__ import annotations

from typing import Any

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import DummyOutput

from xcode.cli.tui.app import _XcodeTui
from xcode.cli.tui.rendering import rendered_markdown_lines, visible_lines
from xcode.cli.tui.state import _LogEntry, _TuiState
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
    assert rendered.endswith("> hello\n")


def test_tui_state_emits_styled_fragments() -> None:
    state = _TuiState()
    state.add_user("hello")
    state.log.append(_LogEntry("xcode", "hi", markdown=True))

    fragments = state.fragments()
    assert ("class:user", "> hello") in fragments
    assert any("hi" in fragment[1] for fragment in fragments)


def test_visible_lines_follow_tail_and_scrollback() -> None:
    lines = [f"line {i}" for i in range(8)]

    assert visible_lines(lines, 3, 0) == ["line 5", "line 6", "line 7"]
    assert visible_lines(lines, 3, 2) == ["line 3", "line 4", "line 5"]
    assert visible_lines(lines, 20, 0) == lines


def test_markdown_lines_render_markdown_structure() -> None:
    rendered = "\n".join(rendered_markdown_lines("# Title\n\n- item"))

    assert "Title" in rendered
    assert "item" in rendered


def test_tui_state_collapses_thinking() -> None:
    state = _TuiState()
    state.thinking = "private reasoning"

    state.toggle_thinking()
    rendered = state.render()

    assert "Thinking" in rendered
    assert "private reasoning" not in rendered


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
