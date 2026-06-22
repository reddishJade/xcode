"""轻量会话待办工具、状态和事件测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from xcode.ai.events import FinalMessage, TextDelta, ToolCall, ToolCallEvent
from xcode.cli.repl_sessions import sync_agent_history
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.harness.agent_runtime.events import TodoUpdateStructuredEvent
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.harness.agent_runtime.result import RunState
from xcode.harness.session import SessionStore
from xcode.harness.session_todo import (
    build_session_todo_tools,
    SessionTodoState,
    TodoItem,
)
from xcode.tests.fixtures import FakeProvider
import pytest

INVALID_TODO_PAYLOADS: tuple[list[dict[str, str]], ...] = (
    [
        {"id": "a", "content": "one", "status": "pending"},
        {"id": "a", "content": "two", "status": "completed"},
    ],
    [{"id": "a", "content": " ", "status": "pending"}],
    [{"id": "a", "content": "one", "status": "unknown"}],
    [
        {"id": "a", "content": "one", "status": "in_progress"},
        {"id": "b", "content": "two", "status": "in_progress"},
    ],
)


class SessionTodoTests:
    """验证完整替换、不变量、持久化和上下文保护。"""

    @pytest.mark.parametrize("payload", INVALID_TODO_PAYLOADS)
    def test_replace_validates_ids_content_status_and_active_count(
        self, payload: list[dict[str, str]]
    ) -> None:
        """拒绝重复 id、空内容、无效状态和多个进行中项。"""
        state = SessionTodoState()
        with pytest.raises(ValueError):
            state.replace(payload)

        assert state.snapshot() == ()

    def test_update_tool_replaces_complete_list(self) -> None:
        """update_todo 每次以完整列表替换旧状态。"""
        state = SessionTodoState()
        (tool,) = build_session_todo_tools(state)

        tool.handler(
            {
                "items": [
                    {
                        "id": "first",
                        "content": "First task",
                        "status": "in_progress",
                    }
                ]
            }
        )
        tool.handler(
            {
                "items": [
                    {
                        "id": "second",
                        "content": "Second task",
                        "status": "completed",
                    }
                ]
            }
        )

        assert state.snapshot() == (TodoItem("second", "Second task", "completed"),)

    def test_structured_agent_emits_todo_event_and_run_state(self) -> None:
        """成功工具调用发射结构化事件并进入 RunState。"""
        state = SessionTodoState()
        provider = FakeProvider(
            [
                [
                    ToolCallEvent(
                        calls=[
                            ToolCall(
                                id="todo-1",
                                name="update_todo",
                                input={
                                    "items": [
                                        {
                                            "id": "implement",
                                            "content": "Implement feature",
                                            "status": "in_progress",
                                        }
                                    ]
                                },
                            )
                        ]
                    ),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
                [
                    TextDelta(chunk="done"),
                    FinalMessage(content="", stop_reason="end_turn"),
                ],
            ]
        )
        agent = StructuredAgent(
            provider=provider,
            registry=build_session_todo_tools(state),
            runtime=AgentRuntimeConfig(todo_state=state),
        )

        events = list(agent.run_stream("go"))

        todo_event = next(
            event for event in events if isinstance(event, TodoUpdateStructuredEvent)
        )
        assert todo_event.data[0].id == "implement"
        final = events[-1]
        run_state = cast(Any, final).data.run_state
        assert run_state.todos == state.snapshot()

    def test_run_state_round_trip_restores_todos(self) -> None:
        """RunState JSON 往返保留当前清单。"""
        state = RunState(
            messages=[],
            todos=(TodoItem("verify", "Run tests", "pending"),),
        )

        restored = RunState.from_dict(state.to_dict())

        assert restored.todos == state.todos

        target_state = SessionTodoState()
        agent = StructuredAgent(
            provider=FakeProvider([]),
            registry=(),
            runtime=AgentRuntimeConfig(todo_state=target_state),
        )
        agent.load_run_state(restored)
        assert target_state.snapshot() == state.todos

    def test_runtime_context_reinjects_current_todos(self) -> None:
        """待办状态不依赖历史消息，压缩后仍由动态上下文重新注入。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state = SessionTodoState()
            state.replace(
                [
                    {
                        "id": "test",
                        "content": "Run targeted tests",
                        "status": "in_progress",
                    }
                ]
            )
            provider = build_runtime_context_provider(
                root,
                (),
                todo_state=state,
            )

            rendered = "\n".join(provider("continue"))

        assert "<session-todo>" in rendered
        assert "Run targeted tests" in rendered

    def test_resume_restores_latest_todo_event(self) -> None:
        """REPL resume 从 transcript 中最后一次完整替换恢复清单。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = SessionStore(root / ".local" / "sessions", project_root=root)
            store.append(
                "event",
                {
                    "type": "todo_update",
                    "step": 1,
                    "data": [
                        {
                            "id": "old",
                            "content": "Old",
                            "status": "completed",
                        }
                    ],
                },
            )
            store.append(
                "event",
                {
                    "type": "todo_update",
                    "step": 2,
                    "data": [
                        {
                            "id": "current",
                            "content": "Current",
                            "status": "in_progress",
                        }
                    ],
                },
            )
            restored: list[list[dict[str, object]]] = []
            app = SimpleNamespace(
                agent=SimpleNamespace(load_history=lambda _messages: None),
                restore_todos=lambda items: restored.append(items),
                contextual_state=None,
            )

            sync_agent_history(cast(Any, app), store)

        assert restored[-1][0]["id"] == "current"


if __name__ == "__main__":
    pytest.main()
