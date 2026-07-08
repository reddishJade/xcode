"""Git preflight 纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.git_preflight import _truncate, MAX_SECTION_CHARS


class TestTruncate:
    def test_short_text(self) -> None:
        assert _truncate("hello") == "hello"

    def test_long_text_truncated(self) -> None:
        text = "x" * (MAX_SECTION_CHARS + 200)
        result = _truncate(text)
        assert len(result) <= MAX_SECTION_CHARS
        assert "truncated" in result
