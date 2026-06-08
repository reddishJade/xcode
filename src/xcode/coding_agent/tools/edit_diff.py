from __future__ import annotations

import unicodedata
from difflib import unified_diff
from typing import Any


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


def normalize_for_fuzzy_match(text: str) -> str:
    """\u5f52\u4e00\u5316\u6587\u672c\u4ee5\u5bb9\u5fcd LLM \u8f93\u51fa\u7684 Unicode \u53d8\u4f53\u3002

    \u8bbe\u8ba1\u539f\u56e0\uff1a
    LLM \u751f\u6210\u7684\u4ee3\u7801\u7247\u6bb5\u5e38\u5305\u542b\u667a\u80fd\u5f15\u53f7\u3001\u5168\u89d2\u7a7a\u683c\u7b49\u6392\u7248\u5b57\u7b26\uff0c
    \u5bfc\u81f4\u5b8c\u5168\u5339\u914d\u5931\u8d25\u3002\u5f52\u4e00\u5316\u5c06\u8fd9\u4e9b\u53d8\u4f53\u6620\u5c04\u5230 ASCII \u7b49\u4ef7\u5b57\u7b26\uff1a
    - NFKC: \u517c\u5bb9\u6027\u5206\u89e3\uff08\u5168\u89d2\u2192\u534a\u89d2\uff0c\u8fde\u5b57\u2192\u5355\u5b57\u7b26\uff09
    - \u667a\u80fd\u5f15\u53f7 \u2192 ASCII \u5f15\u53f7\uff08'"/\uff09
    - \u5404\u7c7b\u8fde\u5b57\u7b26/\u51cf\u53f7 \u2192 ASCII \u8fde\u5b57\u7b26\uff08-\uff09
    - \u5404\u7c7b\u7a7a\u683c \u2192 ASCII \u7a7a\u683c\uff08U+0020\uff09

    \u8fd9\u907f\u514d\u4e86\u56e0\u6392\u7248\u5dee\u5f02\u5bfc\u81f4\u7684 Edit \u5de5\u5177\u8c03\u7528\u5931\u8d25\u3002
    """
    result = unicodedata.normalize("NFKC", text)
    result = "\n".join(line.rstrip() for line in result.split("\n"))
    # \u667a\u80fd\u5355\u5f15\u53f7 \u2192 ASCII \u5355\u5f15\u53f7
    for src in ["\u2018", "\u2019", "\u201a", "\u201b"]:
        result = result.replace(src, "'")
    # \u667a\u80fd\u53cc\u5f15\u53f7 \u2192 ASCII \u53cc\u5f15\u53f7
    for src in ["\u201c", "\u201d", "\u201e", "\u201f"]:
        result = result.replace(src, '"')
    # \u5404\u7c7b\u8fde\u5b57\u7b26/\u51cf\u53f7 \u2192 ASCII \u8fde\u5b57\u7b26
    for src in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"]:
        result = result.replace(src, "-")
    # \u5404\u7c7b\u7a7a\u683c \u2192 ASCII \u7a7a\u683c
    for src in [
        "\u00a0",
        "\u2002",
        "\u2003",
        "\u2004",
        "\u2005",
        "\u2006",
        "\u2007",
        "\u2008",
        "\u2009",
        "\u200a",
        "\u202f",
        "\u205f",
        "\u3000",
    ]:
        result = result.replace(src, " ")
    return result


def fuzzy_find_text(content: str, old_text: str) -> dict[str, Any]:
    """查找文本位置，优先精确匹配，失败后尝试模糊匹配。

    返回字典包含：
    - found: 是否找到
    - index: 匹配位置（字符索引）
    - match_length: 匹配长度
    - used_fuzzy: 是否使用了模糊匹配
    """
    idx = content.find(old_text)
    if idx != -1:
        return {
            "found": True,
            "index": idx,
            "match_length": len(old_text),
            "used_fuzzy": False,
        }

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    idx = fuzzy_content.find(fuzzy_old)
    if idx != -1:
        return {
            "found": True,
            "index": idx,
            "match_length": len(fuzzy_old),
            "used_fuzzy": True,
        }

    return {"found": False, "index": -1, "match_length": 0, "used_fuzzy": False}


def _count_fuzzy_occurrences(content: str, old_text: str) -> int:
    """统计模糊匹配的出现次数，用于检测重复。"""
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    return fuzzy_content.count(fuzzy_old)


def apply_edits_fuzzy(
    content: str,
    edits: list[dict[str, str]],
    path: str,
) -> tuple[str, int]:
    """应用编辑序列到文件内容，支持模糊匹配。

    确保每个 old_text 在文件中唯一匹配，否则报错。
    返回：(修改后的内容, 修改次数)
    """
    normalized_edits = [
        {
            "old_text": normalize_to_lf(e["old_text"]),
            "new_text": normalize_to_lf(e["new_text"]),
        }
        for e in edits
    ]

    for i, edit in enumerate(normalized_edits):
        if not edit["old_text"]:
            _raise_field_error(
                "old_text must not be empty", path, i, len(normalized_edits)
            )

    matches = []
    needs_fuzzy = False
    for i, edit in enumerate(normalized_edits):
        match = fuzzy_find_text(content, edit["old_text"])
        if not match["found"]:
            _raise_not_found(path, i, len(normalized_edits))
        if match["used_fuzzy"]:
            needs_fuzzy = True
        matches.append(match)

    base = content
    if needs_fuzzy:
        base = normalize_for_fuzzy_match(content)
        matches = []
        for i, edit in enumerate(normalized_edits):
            match = fuzzy_find_text(base, edit["old_text"])
            if not match["found"]:
                _raise_not_found(path, i, len(normalized_edits))
            matches.append(match)

    for i, edit in enumerate(normalized_edits):
        occurrences = _count_fuzzy_occurrences(base, edit["old_text"])
        if occurrences > 1:
            _raise_duplicate(path, i, len(normalized_edits), occurrences)

    matched_edits: list[dict[str, Any]] = []
    for i in range(len(normalized_edits)):
        matched_edits.append(
            {
                "edit_index": i,
                "match_index": matches[i]["index"],
                "match_length": matches[i]["match_length"],
                "new_text": normalized_edits[i]["new_text"],
            }
        )

    matched_edits.sort(key=lambda m: m["match_index"])
    for a, b in zip(matched_edits, matched_edits[1:]):
        if a["match_index"] + a["match_length"] > b["match_index"]:
            raise ValueError(
                f"edits[{a['edit_index']}] and edits[{b['edit_index']}] "
                f"overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    new_content = base
    for m in reversed(matched_edits):
        new_content = (
            new_content[: m["match_index"]]
            + m["new_text"]
            + new_content[m["match_index"] + m["match_length"] :]
        )

    if base == new_content:
        _raise_no_change(path, len(normalized_edits))

    return new_content, len(matched_edits)


def generate_diff_string(old_content: str, new_content: str, path: str = "") -> str:
    if not path:
        return "".join(
            unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
            )
        )
    return "".join(
        unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _raise_field_error(msg: str, path: str, edit_index: int, total: int) -> None:
    if total == 1:
        raise ValueError(f"{msg} in {path}.")
    raise ValueError(f"edits[{edit_index}].{msg} in {path}.")


def _raise_not_found(path: str, edit_index: int, total: int) -> None:
    if total == 1:
        raise ValueError(
            f"Could not find the exact text in {path}. "
            "The old text must match exactly including all whitespace and newlines."
        )
    raise ValueError(
        f"Could not find edits[{edit_index}] in {path}. "
        "The old text must match exactly including all whitespace and newlines."
    )


def _raise_duplicate(path: str, edit_index: int, total: int, count: int) -> None:
    if total == 1:
        raise ValueError(
            f"Found {count} occurrences of the text in {path}. "
            "The text must be unique. Please provide more context to make it unique."
        )
    raise ValueError(
        f"Found {count} occurrences of edits[{edit_index}] in {path}. "
        "Each old_text must be unique. Please provide more context to make it unique."
    )


def _raise_no_change(path: str, total: int) -> None:
    if total == 1:
        raise ValueError(
            f"No changes made to {path}. The replacement produced identical content."
        )
    raise ValueError(
        f"No changes made to {path}. The replacements produced identical content."
    )
