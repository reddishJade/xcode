"""todowrite 工具验证逻辑单元测试。"""

from __future__ import annotations

from xcode.coding_agent.tools.todowrite import build_todowrite_tool

tool = build_todowrite_tool()


class TestTodoWriteValidation:
    def test_valid_todo(self) -> None:
        result = tool.handler({"todos": [{"content": "Fix bug", "status": "pending"}]}, None)
        assert "Fix bug" in result
        assert "pending" in result

    def test_missing_todos(self) -> None:
        result = tool.handler({}, None)
        assert "must be an array" in result

    def test_empty_content_raises(self) -> None:
        result = tool.handler({"todos": [{"content": "", "status": "pending"}]}, None)
        assert "must not be empty" in result

    def test_invalid_status_raises(self) -> None:
        result = tool.handler({"todos": [{"content": "task", "status": "invalid"}]}, None)
        assert "invalid status" in result

    def test_valid_statuses(self) -> None:
        for s in ("pending", "in_progress", "completed", "cancelled"):
            result = tool.handler({"todos": [{"content": "task", "status": s}]}, None)
            assert "error" not in result.lower()

    def test_invalid_priority_raises(self) -> None:
        result = tool.handler(
            {"todos": [{"content": "task", "status": "pending", "priority": "urgent"}]},
            None,
        )
        assert "invalid priority" in result

    def test_valid_priorities(self) -> None:
        for p in ("high", "medium", "low"):
            result = tool.handler(
                {"todos": [{"content": "task", "status": "pending", "priority": p}]},
                None,
            )
            assert "error" not in result.lower()
