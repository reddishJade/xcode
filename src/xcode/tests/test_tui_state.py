from pathlib import Path

from xcode.cli.tui.state import _TuiState


def test_streaming_answer_keeps_chunks_without_copying_previous_text() -> None:
    state = _TuiState(project_root=Path("/project"))

    state._append_or_update_answer("one")
    state._append_or_update_answer(" two")
    state._append_or_update_answer(" three")

    entry = state.streaming_answer
    assert entry is not None
    assert entry.content() == "one two three"
    assert entry.text == ""
    assert entry.text_parts == ["one", " two", " three"]


def test_streaming_answer_renders_from_chunks() -> None:
    state = _TuiState(project_root=Path("/project"))
    state._append_or_update_answer("**hello**")
    state._append_or_update_answer(" world")

    assert any("hello" in line for line in state.lines())
    assert any("world" in line for line in state.lines())


def test_plain_entries_count_visual_lines_at_output_width() -> None:
    state = _TuiState(project_root=Path("/project"))
    state.add_user("abcdefghij")

    assert state.lines(width=6) == ["> abcd", "efghij"]
    assert state.line_count(width=6) == 2


def test_plain_entries_wrap_wide_characters_by_display_width() -> None:
    state = _TuiState(project_root=Path("/project"))
    state.add_user("你好世界")

    assert state.lines(width=6) == ["> 你好", "世界"]


def test_markdown_reflows_when_output_width_changes() -> None:
    state = _TuiState(project_root=Path("/project"))
    state._append_or_update_answer("alpha beta gamma delta epsilon zeta")

    narrow_lines = state.ansi_lines(width=12)
    wide_lines = state.ansi_lines(width=80)

    assert len(narrow_lines) > len(wide_lines)
    assert state.line_count(width=12) == len(narrow_lines)
    assert state.line_count(width=80) == len(wide_lines)
