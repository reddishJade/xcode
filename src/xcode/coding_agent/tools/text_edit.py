from __future__ import annotations


def detect_line_ending(content: str) -> str:
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


def strip_bom(content: str) -> tuple[str, str]:
    if content.startswith("\ufeff"):
        return ("\ufeff", content[1:])
    return ("", content)


def apply_text_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    if old_string == new_string:
        raise ValueError("No changes to apply: oldString and newString are identical.")
    if not old_string:
        raise ValueError(
            "old_string cannot be empty when editing an existing file. "
            "Use write for an intentional full-file replacement."
        )

    idx = content.find(old_string)
    if idx == -1:
        raise ValueError(
            "Could not find old_string in the file. "
            "It must match exactly, including all whitespace and newlines."
        )
    if replace_all:
        return content.replace(old_string, new_string)
    last_idx = content.rfind(old_string)
    if idx == last_idx:
        return content[:idx] + new_string + content[idx + len(old_string) :]
    raise ValueError(
        "Found multiple matches for old_string. "
        "Provide more surrounding context to make the match unique."
    )
