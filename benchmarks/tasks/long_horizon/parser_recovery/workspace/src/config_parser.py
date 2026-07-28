"""简单 KEY=VALUE 配置解析器。"""

from __future__ import annotations


class ParseError(ValueError):
    """输入行不符合配置格式。"""


def parse_config(lines: list[str]) -> dict[str, str]:
    """解析配置行，并让后出现的重复键覆盖先前值。"""
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("=")
        if len(parts) != 2:
            raise ParseError(f"line {line_number}: expected KEY=VALUE")
        key, value = parts
        if not key:
            raise ParseError(f"line {line_number}: key must not be empty")
        result[key] = value
    return result
