from pathlib import Path

from xcode.cli.tui.state import _TuiState


def test_streaming_answer_keeps_chunks_without_copying_previous_text() -> None:
    state = _TuiState(project_root=Path("/project"))

    state._append_or_update_answer("one")
    state._append_or_update_answer(" two")
    state._append_or_update_answer(" three")

    entry = state.log[-1]
    assert entry.content() == "one two three"
    assert entry.text == ""
    assert entry.text_parts == ["one", " two", " three"]


def test_streaming_answer_renders_from_chunks() -> None:
    state = _TuiState(project_root=Path("/project"))
    state._append_or_update_answer("**hello**")
    state._append_or_update_answer(" world")

    assert any("hello" in line for line in state.lines())
    assert any("world" in line for line in state.lines())
