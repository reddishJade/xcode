from string_utils import count_words, reverse_string


def test_reverse_string() -> None:
    assert reverse_string("hello") == "olleh"


def test_count_words() -> None:
    assert count_words("hello world") == 2
