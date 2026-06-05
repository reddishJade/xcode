from __future__ import annotations

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
GREP_MAX_LINE_LENGTH = 500


def format_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_}B"
    if bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f}KB"
    return f"{bytes_ / (1024 * 1024):.1f}MB"


def _split_lines(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def truncate_head(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str:
    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return content

    if not lines:
        return ""

    first_line_bytes = len(lines[0].encode("utf-8"))
    if first_line_bytes > max_bytes:
        return ""

    output: list[str] = []
    output_bytes = 0
    for line in lines:
        if len(output) >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if output else 0)
        if output_bytes + line_bytes > max_bytes:
            break
        output.append(line)
        output_bytes += line_bytes

    result = "\n".join(output)
    return result


def truncate_tail(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> str:
    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return content

    output: list[str] = []
    output_bytes = 0
    for line in reversed(lines):
        if len(output) >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if output else 0)
        if output_bytes + line_bytes > max_bytes:
            break
        output.append(line)
        output_bytes += line_bytes

    output.reverse()
    return "\n".join(output)


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    if len(line) <= max_chars:
        return (line, False)
    return (line[:max_chars] + "... [truncated]", True)
