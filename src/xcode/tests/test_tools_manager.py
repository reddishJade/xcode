"""工具管理器纯函数单元测试。"""

from __future__ import annotations

from xcode.coding_agent.tools.tools_manager import (
    _resolve_tool_path,
    ExternalToolDefinition,
)


def _mock_which(found: str | None) -> str | None:
    return found


class TestResolveToolPath:
    def test_found(self) -> None:
        def resolver(name: str) -> str | None:
            return "/usr/bin/rg" if name == "rg" else None

        result = _resolve_tool_path(
            ExternalToolDefinition(display_name="rg", candidate_names=("rg",)),
            resolver,
        )
        assert result == "/usr/bin/rg"

    def test_not_found(self) -> None:
        result = _resolve_tool_path(
            ExternalToolDefinition(display_name="rg", candidate_names=("rg",)),
            lambda _: None,
        )
        assert result is None

    def test_falls_back_through_candidates(self) -> None:
        def resolver(name: str) -> str | None:
            return "/usr/bin/foo" if name == "foo" else None

        result = _resolve_tool_path(
            ExternalToolDefinition(
                display_name="rg",
                candidate_names=("rg", "foo"),
            ),
            resolver,
        )
        assert result == "/usr/bin/foo"
