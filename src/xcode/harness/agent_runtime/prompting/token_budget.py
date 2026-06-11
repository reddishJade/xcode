from __future__ import annotations

MAX_CWD_ENTRIES = 12
INSTRUCTION_WARNING_BYTES = 24 * 1024
MAX_INSTRUCTION_BYTES = 32 * 1024
INSTRUCTION_OPENING_BYTES = 6 * 1024
SECTION_BUDGET_BYTES = 4 * 1024
KEY_INSTRUCTION_SECTIONS = frozenset(
    {
        "priority",
        "conversation style",
        "python coding principles",
        "checklist",
        "project rules",
        "comments and docstrings",
        "dependencies",
        "temporary scripts",
        "experimental features",
        "git safety",
        "commit rules",
        "validation",
        "working rules",
    }
)


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _utf8_prefix(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="ignore")
