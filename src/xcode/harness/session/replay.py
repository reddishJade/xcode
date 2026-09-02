"""从 append-only session transcript 重建 agent 运行状态。"""

from __future__ import annotations

from typing import Protocol

from xcode.agent.messages import AgentMessage

from .surface import project_session_surface
from .tree_store import TreeSessionRepo
from .types import SessionEntry


class ReplayAgent(Protocol):
    """会话投影器恢复运行状态所需的 agent 接口。"""

    @property
    def session_id(self) -> str: ...

    @session_id.setter
    def session_id(self, value: str) -> None: ...

    def set_history_session_id(self, session_id: str) -> None: ...

    def load_history(self, messages: list[AgentMessage]) -> None: ...

    def restore_run_state_metadata(self, payload: object) -> None: ...

    def restore_goal_state(self, payload: object) -> None: ...

    def set_resumed_notice(self, notice: str) -> None: ...


class ContextualReplayState(Protocol):
    """回放时恢复文件和工具上下文所需的状态接口。"""

    def record_file(self, path: str) -> None: ...

    def record_tool_result(self, tool: str, content: str) -> None: ...

    def clear(self) -> None: ...


def replay_session(
    agent: ReplayAgent,
    store: TreeSessionRepo,
    contextual_state: ContextualReplayState | None = None,
) -> None:
    """把当前 session branch 的事实投影回 agent 内存。"""
    agent.session_id = store.session_id
    agent.set_history_session_id(store.session_id)
    records = store.build_branch()
    surface = project_session_surface(records)
    agent.load_history(list(surface.messages))
    run_state = latest_run_state(records)
    if run_state is not None:
        agent.restore_run_state_metadata(run_state)
    goal_state = latest_goal_state(records)
    if goal_state is not None:
        agent.restore_goal_state(goal_state)
    if contextual_state is not None:
        contextual_state.clear()
        restore_contextual_state(contextual_state, records)
    if surface.generation > 0:
        agent.set_resumed_notice(
            "This session resumed in its latest context window. NOTE.md contains "
            "explicit working state and the lossless transcript is authoritative. "
            "Use history for older exact details instead of asking the user to "
            "restate them."
        )
    else:
        agent.set_resumed_notice(
            "This conversation was resumed from a previous session. "
            "The transcript history above has been loaded as context. "
            "Continue the task as if the session was uninterrupted."
        )


def latest_run_state(records: list[SessionEntry]) -> object | None:
    """读取当前 branch 最近一次 final event 中的 run_state。"""
    for record in reversed(records):
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        if record.content.get("type") != "final":
            continue
        data = record.content.get("data")
        if not isinstance(data, dict):
            continue
        run_state = data.get("run_state")
        if isinstance(run_state, dict):
            return run_state
    return None


def latest_goal_state(records: list[SessionEntry]) -> object | None:
    """按时间倒序读取 Goal 命令或 final 中的最新状态。"""
    for record in reversed(records):
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        event_type = record.content.get("type")
        data = record.content.get("data")
        if event_type == "goal_state" and isinstance(data, dict):
            return data
        if event_type != "final" or not isinstance(data, dict):
            continue
        run_state = data.get("run_state")
        if isinstance(run_state, dict) and isinstance(run_state.get("goal"), dict):
            return run_state["goal"]
    return None


def restore_contextual_state(
    contextual_state: ContextualReplayState,
    records: list[SessionEntry],
) -> None:
    """从 transcript 恢复上下文检索状态。"""
    for record in records:
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        event_type = str(record.content.get("type", ""))
        event_data = record.content.get("data")
        if event_type == "file_references" and isinstance(event_data, list):
            for ref in event_data:
                path = ref.get("path", "") if isinstance(ref, dict) else ""
                if isinstance(path, str) and path:
                    contextual_state.record_file(path)
        elif event_type == "tool_result" and isinstance(event_data, dict):
            tool_name = str(event_data.get("tool_use_id", "") or "")
            content = str(event_data.get("content", "") or "")
            if tool_name:
                contextual_state.record_tool_result(tool_name, content)


def records_to_agent_messages(records: list[SessionEntry]) -> list[AgentMessage]:
    """将 session 事实账本投影为 provider 使用的消息历史。"""
    return list(project_session_surface(records).messages)
