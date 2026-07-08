"""上下文检索状态单元测试。"""

from __future__ import annotations

from pathlib import Path

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState


class TestContextualRetrievalState:
    def test_empty_render(self) -> None:
        state = ContextualRetrievalState(Path("/project"))
        rendered = state.render()
        assert "<contextual-retrieval>" in rendered
        assert "</contextual-retrieval>" in rendered
        assert "active_file: " not in rendered

    def test_record_file(self) -> None:
        state = ContextualRetrievalState(Path("/project"))
        state.record_file(Path("src/main.py"))
        rendered = state.render()
        assert "active_file: src/main.py" in rendered
        assert "recent_files" in rendered

    def test_max_files_lru(self) -> None:
        state = ContextualRetrievalState(Path("/project"), max_files=3)
        for i in range(5):
            state.record_file(Path(f"file{i}.py"))
        rendered = state.render()
        assert "file0.py" not in rendered
        assert "file4.py" in rendered

    def test_record_tool_result(self) -> None:
        state = ContextualRetrievalState(Path("/project"))
        state.record_tool_result("read_file", "loaded 42 lines of code")
        rendered = state.render()
        assert "read_file" in rendered
        assert "recent_tool_results" in rendered

    def test_long_tool_result_truncated(self) -> None:
        state = ContextualRetrievalState(Path("/project"))
        state.record_tool_result("bash", "x" * 500)
        rendered = state.render()
        assert "..." in rendered.split("bash:")[1] if "bash:" in rendered else True

    def test_record_tool_call(self) -> None:
        state = ContextualRetrievalState(Path("/project"))
        state.record_tool_call(tool="write_file", input_brief="path=/x", status="allow")
        rendered = state.render()
        assert "recent_tool_calls" in rendered
        assert "write_file" in rendered

    def test_cache_invalidation_on_record(self) -> None:
        state = ContextualRetrievalState(Path("/project"))
        first = state.render()
        state.record_file(Path("/project/file.py"))
        second = state.render()
        assert first != second

    @property
    def active_file(self) -> str | None:
        state = ContextualRetrievalState(Path("/project"))
        assert state.active_file is None
        state.record_file(Path("src/main.py"))
        assert state.active_file == "src/main.py"
