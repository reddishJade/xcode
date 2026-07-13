from prompt_toolkit.widgets import TextArea

from xcode.cli.tui.app import _XcodeTui
from xcode.cli.tui.widgets import TuiInputLexer, tui_input_prompt


def test_input_lexer_does_not_highlight_shell_prefix() -> None:
    assert TuiInputLexer._highlight("!pwd") == [("", "!pwd")]


def test_input_prompt_uses_default_style_for_normal_input() -> None:
    assert tui_input_prompt(False, False) == [("", "> ")]


def test_input_prompt_highlights_marker_for_shell_command() -> None:
    assert tui_input_prompt(False, True) == [("class:prompt-marker", "> ")]


def test_input_height_tracks_text_lines_up_to_five() -> None:
    tui = object.__new__(_XcodeTui)
    tui._input = TextArea(text="one\ntwo\nthree")

    assert tui._input_height() == 3

    tui._input.text = "\n" * 5
    assert tui._input_height() == 5
