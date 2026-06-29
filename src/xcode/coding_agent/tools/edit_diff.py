from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import re
import unicodedata
from difflib import unified_diff


@dataclass(frozen=True)
class NormalizedEdit:
    old_text: str
    new_text: str


@dataclass(frozen=True)
class TextMatch:
    found: bool
    index: int
    match_length: int
    used_fuzzy: bool = False


@dataclass(frozen=True)
class MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


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
    result = unicodedata.normalize("NFKC", text)
    result = "\n".join(line.rstrip() for line in result.split("\n"))
    table: dict[str, str] = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
    }
    for src, dst in table.items():
        result = result.replace(src, dst)
    return result


def fuzzy_find_text(content: str, old_text: str) -> TextMatch:
    """查找文本位置，优先精确匹配，失败后尝试模糊匹配。

    返回 TextMatch，供后续编辑排序和重叠检测使用。
    """
    idx = content.find(old_text)
    if idx != -1:
        return TextMatch(found=True, index=idx, match_length=len(old_text))

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    idx = fuzzy_content.find(fuzzy_old)
    if idx != -1:
        return TextMatch(
            found=True,
            index=idx,
            match_length=len(fuzzy_old),
            used_fuzzy=True,
        )

    return TextMatch(found=False, index=-1, match_length=0)


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
    normalized_edits = _normalize_edits(edits)
    _validate_edit_targets(normalized_edits, path)
    base, matched_edits = _plan_fuzzy_edits(content, normalized_edits, path)
    new_content = _apply_matched_edits(base, matched_edits)

    if base == new_content:
        _raise_no_change(path, len(normalized_edits))

    return new_content, len(matched_edits)


def _normalize_edits(edits: list[dict[str, str]]) -> list[NormalizedEdit]:
    return [
        NormalizedEdit(
            old_text=normalize_to_lf(edit["old_text"]),
            new_text=normalize_to_lf(edit["new_text"]),
        )
        for edit in edits
    ]


def _validate_edit_targets(edits: list[NormalizedEdit], path: str) -> None:
    for i, edit in enumerate(edits):
        if not edit.old_text:
            _raise_field_error("old_text must not be empty", path, i, len(edits))


def _plan_fuzzy_edits(
    content: str,
    edits: list[NormalizedEdit],
    path: str,
) -> tuple[str, list[MatchedEdit]]:
    matches, needs_fuzzy = _locate_edit_targets(content, edits, path)
    base = content
    if needs_fuzzy:
        base = normalize_for_fuzzy_match(content)
        matches, _ = _locate_edit_targets(base, edits, path)

    _ensure_unique_targets(base, edits, path)
    matched_edits = _build_matched_edits(edits, matches)
    _ensure_disjoint_edits(matched_edits, path)
    return base, matched_edits


def _locate_edit_targets(
    content: str,
    edits: list[NormalizedEdit],
    path: str,
) -> tuple[list[TextMatch], bool]:
    matches: list[TextMatch] = []
    needs_fuzzy = False
    for i, edit in enumerate(edits):
        match = fuzzy_find_text(content, edit.old_text)
        if not match.found:
            _raise_not_found(path, i, len(edits))
        if match.used_fuzzy:
            needs_fuzzy = True
        matches.append(match)
    return matches, needs_fuzzy


def _ensure_unique_targets(
    content: str,
    edits: list[NormalizedEdit],
    path: str,
) -> None:
    for i, edit in enumerate(edits):
        occurrences = _count_fuzzy_occurrences(content, edit.old_text)
        if occurrences > 1:
            _raise_duplicate(path, i, len(edits), occurrences)


def _build_matched_edits(
    edits: list[NormalizedEdit],
    matches: list[TextMatch],
) -> list[MatchedEdit]:
    matched_edits: list[MatchedEdit] = []
    for i, edit in enumerate(edits):
        matched_edits.append(
            MatchedEdit(
                edit_index=i,
                match_index=matches[i].index,
                match_length=matches[i].match_length,
                new_text=edit.new_text,
            )
        )
    return sorted(matched_edits, key=lambda m: m.match_index)


def _ensure_disjoint_edits(matched_edits: list[MatchedEdit], path: str) -> None:
    for a, b in zip(matched_edits, matched_edits[1:]):
        if a.match_index + a.match_length > b.match_index:
            raise ValueError(
                f"edits[{a.edit_index}] and edits[{b.edit_index}] "
                f"overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )


def _apply_matched_edits(content: str, matched_edits: list[MatchedEdit]) -> str:
    new_content = content
    for m in reversed(matched_edits):
        new_content = (
            new_content[: m.match_index]
            + m.new_text
            + new_content[m.match_index + m.match_length :]
        )
    return new_content


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


REPLACER = Callable[[str, str], Iterator[str]]


def _levenshtein(a: str, b: str) -> int:
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a):
        curr[0] = i + 1
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev, curr = curr, prev
    return prev[len(b)]


def _simple_replacer(content: str, find: str) -> Iterator[str]:
    yield find


def _line_trimmed_replacer(content: str, find: str) -> Iterator[str]:
    original_lines = content.split("\n")
    search_lines = find.split("\n")
    if search_lines and search_lines[-1] == "":
        search_lines.pop()
    for i in range(len(original_lines) - len(search_lines) + 1):
        if all(
            original_lines[i + j].strip() == search_lines[j].strip()
            for j in range(len(search_lines))
        ):
            start = sum(len(original_lines[k]) + 1 for k in range(i))
            end = start + sum(
                len(original_lines[i + k]) + (1 if k < len(search_lines) - 1 else 0)
                for k in range(len(search_lines))
            )
            yield content[start:end]


def _block_anchor_replacer(content: str, find: str) -> Iterator[str]:
    original_lines = content.split("\n")
    search_lines = find.split("\n")
    if search_lines and search_lines[-1] == "":
        search_lines.pop()
    if len(search_lines) < 3:
        return
    first_search = search_lines[0].strip()
    last_search = search_lines[-1].strip()
    search_block_size = len(search_lines)
    max_line_delta = max(1, search_block_size // 4)

    candidates: list[tuple[int, int]] = []
    for i in range(len(original_lines)):
        if original_lines[i].strip() != first_search:
            continue
        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last_search:
                actual_size = j - i + 1
                if abs(actual_size - search_block_size) <= max_line_delta:
                    candidates.append((i, j))
                break

    if not candidates:
        return

    if len(candidates) == 1:
        start_line, end_line = candidates[0]
        similarity = _calc_anchor_similarity(
            original_lines, search_lines, start_line, end_line
        )
        if similarity >= 0.65:
            yield _extract_block(content, original_lines, start_line, end_line)
        return

    best_similarity = -1.0
    best_candidate: tuple[int, int] | None = None
    for start_line, end_line in candidates:
        similarity = _calc_anchor_similarity(
            original_lines, search_lines, start_line, end_line
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_candidate = (start_line, end_line)
    if best_similarity >= 0.65 and best_candidate:
        yield _extract_block(content, original_lines, *best_candidate)


def _calc_anchor_similarity(
    original_lines: list[str],
    search_lines: list[str],
    start_line: int,
    end_line: int,
) -> float:
    inner = min(len(search_lines) - 2, end_line - start_line - 1)
    if inner <= 0:
        return 1.0
    total = 0.0
    for k in range(1, min(len(search_lines) - 1, end_line - start_line)):
        ol = original_lines[start_line + k].strip()
        sl = search_lines[k].strip()
        max_len = max(len(ol), len(sl))
        if max_len > 0:
            total += 1.0 - _levenshtein(ol, sl) / max_len
    return total / inner


def _extract_block(content: str, lines: list[str], start: int, end: int) -> str:
    start_idx = sum(len(lines[k]) + 1 for k in range(start))
    end_idx = start_idx + sum(
        len(lines[k]) + (1 if k < end else 0) for k in range(start, end + 1)
    )
    return content[start_idx:end_idx]


def _whitespace_normalized_replacer(content: str, find: str) -> Iterator[str]:
    def normalize_ws(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    norm_find = normalize_ws(find)
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if normalize_ws(line) == norm_find:
            yield line
        elif norm_find in normalize_ws(line):
            words = find.strip().split()
            if words:
                pattern = r"\s+".join(re.escape(w) for w in words)
                m = re.search(pattern, line)
                if m:
                    yield m.group(0)

    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i : i + len(find_lines)])
            if normalize_ws(block) == norm_find:
                yield block


def _indentation_flexible_replacer(content: str, find: str) -> Iterator[str]:
    def remove_indent(text: str) -> str:
        t_lines = text.split("\n")
        non_empty = [line for line in t_lines if line.strip()]
        if not non_empty:
            return text
        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
        return "\n".join(
            line[min_indent:] if line.strip() else line for line in t_lines
        )

    norm_find = remove_indent(find)
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i : i + len(find_lines)])
        if remove_indent(block) == norm_find:
            yield block


def _escape_normalized_replacer(content: str, find: str) -> Iterator[str]:
    def unescape(s: str) -> str:
        return re.sub(
            r"\\([nrt'\"`\\$])",
            lambda m: {
                "n": "\n",
                "t": "\t",
                "r": "\r",
                "'": "'",
                '"': '"',
                "`": "`",
                "\\": "\\",
                "$": "$",
            }.get(m.group(1), m.group(0)),
            s,
        )

    unescaped = unescape(find)
    if unescaped in content:
        yield unescaped
    lines = content.split("\n")
    find_lines = unescaped.split("\n")
    for i in range(len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if unescape(block) == unescaped:
            yield block


def _trimmed_boundary_replacer(content: str, find: str) -> Iterator[str]:
    trimmed = find.strip()
    if trimmed == find:
        return
    if trimmed in content:
        yield trimmed
    lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if block.strip() == trimmed:
            yield block


def _context_aware_replacer(content: str, find: str) -> Iterator[str]:
    find_lines = find.split("\n")
    if find_lines and find_lines[-1] == "":
        find_lines.pop()
    if len(find_lines) < 3:
        return
    first_line = find_lines[0].strip()
    last_line = find_lines[-1].strip()
    content_lines = content.split("\n")

    for i in range(len(content_lines)):
        if content_lines[i].strip() != first_line:
            continue
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last_line:
                block_lines = content_lines[i : j + 1]
                if len(block_lines) == len(find_lines):
                    matching = 0
                    total = 0
                    for k in range(1, len(block_lines) - 1):
                        bl = block_lines[k].strip()
                        fl = find_lines[k].strip()
                        if bl or fl:
                            total += 1
                            if bl == fl:
                                matching += 1
                    if total == 0 or matching / total >= 0.5:
                        yield "\n".join(block_lines)
                break


def _multi_occurrence_replacer(content: str, find: str) -> Iterator[str]:
    start = 0
    while True:
        idx = content.find(find, start)
        if idx == -1:
            break
        yield find
        start = idx + len(find)


_REPLACERS: list[REPLACER] = [
    _simple_replacer,
    _line_trimmed_replacer,
    _block_anchor_replacer,
    _whitespace_normalized_replacer,
    _indentation_flexible_replacer,
    _escape_normalized_replacer,
    _trimmed_boundary_replacer,
    _context_aware_replacer,
    _multi_occurrence_replacer,
]


def _is_disproportionate_match(search: str, old_string: str) -> bool:
    old_lines = old_string.count("\n") + 1
    search_lines = search.count("\n") + 1
    if search_lines >= max(old_lines + 3, old_lines * 2):
        return True
    if old_lines == 1:
        return False
    return len(search.strip()) > max(
        len(old_string.strip()) + 500, len(old_string.strip()) * 4
    )


def apply_fuzzy_replace(
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

    not_found = True
    for replacer in _REPLACERS:
        for search in replacer(content, old_string):
            idx = content.find(search)
            if idx == -1:
                continue
            not_found = False
            if _is_disproportionate_match(search, old_string):
                raise ValueError(
                    "Refusing replacement because the matched span is much larger "
                    "than oldString. Re-read the file and provide the full exact "
                    "oldString for the intended replacement."
                )
            if replace_all:
                return content.replace(search, new_string)
            last_idx = content.rfind(search)
            if idx != last_idx:
                continue
            return content[:idx] + new_string + content[idx + len(search) :]

    if not_found:
        raise ValueError(
            "Could not find old_string in the file. "
            "It must match exactly, including all whitespace and newlines."
        )
    raise ValueError(
        "Found multiple matches for old_string. "
        "Provide more surrounding context to make the match unique."
    )
