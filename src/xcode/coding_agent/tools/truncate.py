from __future__ import annotations

from dataclasses import dataclass

# 输出限制：平衡 LLM 上下文窗口利用率与响应速度
DEFAULT_MAX_LINES = 2000           # 默认行数限制：约 100K tokens
DEFAULT_MAX_BYTES = 50 * 1024      # 默认字节限制：50KB
GREP_MAX_LINE_LENGTH = 500         # grep 单行限制：避免二进制文件污染输出


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


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
) -> TruncationResult:
    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    if not lines:
        return TruncationResult(
            content="",
            truncated=False,
            truncated_by=None,
            total_lines=0,
            total_bytes=0,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line_bytes = len(lines[0].encode("utf-8"))
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output: list[str] = []
    output_bytes = 0
    truncated_by: str | None = None
    for line in lines:
        if len(output) >= max_lines:
            truncated_by = "lines"
            break
        line_bytes = len(line.encode("utf-8")) + (1 if output else 0)
        if output_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output.append(line)
        output_bytes += line_bytes

    if truncated_by is None:
        truncated_by = "lines" if len(output) < total_lines else None
    result = "\n".join(output)
    result_bytes = len(result.encode("utf-8"))
    result_lines = len(output)
    return TruncationResult(
        content=result,
        truncated=truncated_by is not None,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=result_lines,
        output_bytes=result_bytes,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    total_bytes = len(content.encode("utf-8"))
    lines = _split_lines(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output: list[str] = []
    output_bytes = 0
    last_line_partial = False
    truncated_by: str | None = None
    for line in reversed(lines):
        if len(output) >= max_lines:
            truncated_by = "lines"
            break
        line_bytes = len(line.encode("utf-8")) + (1 if output else 0)
        if output_bytes + line_bytes > max_bytes:
            if len(output) == 0:
                buf = line.encode("utf-8")
                if len(buf) > max_bytes:
                    start = len(buf) - max_bytes
                    while start < len(buf) and (buf[start] & 0xC0) == 0x80:
                        start += 1
                    truncated_line = buf[start:].decode("utf-8")
                    output.insert(0, truncated_line)
                    output_bytes = len(output[0].encode("utf-8"))
                    last_line_partial = True
            truncated_by = "bytes"
            break
        output.insert(0, line)
        output_bytes += line_bytes

    if truncated_by is None:
        truncated_by = "lines" if len(output) < total_lines else None
    result = "\n".join(output)
    result_bytes = len(result.encode("utf-8"))
    result_lines = len(output)
    return TruncationResult(
        content=result,
        truncated=truncated_by is not None,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=result_lines,
        output_bytes=result_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    if len(line) <= max_chars:
        return (line, False)
    return (line[:max_chars] + "... [truncated]", True)
