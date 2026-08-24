"""工具提示构建纯函数单元测试。"""

from __future__ import annotations

from xcode.agent.types import ToolSpec
from xcode.harness.agent_runtime.prompting.tools import (
    build_tool_guidelines,
    build_tool_prompt,
    compact_tool_params,
)


def _make_tool(
    name: str,
    description: str = "",
    snippet: str = "",
    schema: dict | None = None,
    guidelines: tuple[str, ...] = (),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_hint="",
        handler=lambda d, _: "",
        prompt_snippet=snippet or description,
        prompt_guidelines=guidelines,
        schema=schema,
    )


class TestBuildToolPrompt:
    def test_empty_registry(self) -> None:
        assert build_tool_prompt(()) == "(none)"

    def test_single_tool(self) -> None:
        tool = _make_tool("read_file", snippet="Read a file")
        result = build_tool_prompt((tool,))
        assert "read_file: Read a file" in result

    def test_falls_back_to_description(self) -> None:
        tool = _make_tool("bash", description="Run a command")
        result = build_tool_prompt((tool,))
        assert "bash: Run a command" in result


class TestCompactToolParams:
    def test_no_schema(self) -> None:
        tool = _make_tool("t")
        assert compact_tool_params(tool) == []

    def test_required_params_marked(self) -> None:
        tool = _make_tool(
            "t",
            schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        params = compact_tool_params(tool)
        assert any("path" in p and "string" in p for p in params)

    def test_optional_params_marked(self) -> None:
        tool = _make_tool(
            "t",
            schema={
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
        )
        params = compact_tool_params(tool)
        assert any("?" in p and "limit" in p for p in params)


class TestBuildToolGuidelines:
    def test_empty(self) -> None:
        assert build_tool_guidelines(()) == ""

    def test_deduplicates(self) -> None:
        t1 = _make_tool("a", guidelines=("Use X.",))
        t2 = _make_tool("b", guidelines=("Use X.",))
        result = build_tool_guidelines((t1, t2))
        assert result.count("Use X.") == 1

    def test_multiple_guidelines(self) -> None:
        t1 = _make_tool("a", guidelines=("Use X.", "Avoid Y."))
        result = build_tool_guidelines((t1,))
        lines = result.split("\n")
        assert len(lines) == 2
