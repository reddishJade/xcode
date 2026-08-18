"""由运行时拥有的 session 记录器。"""

from __future__ import annotations

from typing import Protocol

from xcode.harness.agent_runtime.events import AgentHarnessEvent, FinalStructuredEvent

from .event_codec import SESSION_EVENT_SCHEMA_VERSION, encode_session_event
from .tree_store import TreeSessionRepo


_DURABLE_EVENT_TYPES = frozenset(
    {
        "assistant",
        "tool_use",
        "tool_result",
        "compaction",
        "final",
    }
)


class SessionBoundAgent(Protocol):
    @property
    def session_id(self) -> str: ...

    @session_id.setter
    def session_id(self, value: str) -> None: ...

    def set_history_session_id(self, session_id: str) -> None: ...

    def set_compaction_source_message_id(self, message_id: str | None) -> None: ...


class SessionRecorder:
    """记录一次运行产生的稳定语义事件，并绑定 agent 的 session 身份。"""

    def __init__(self, store: TreeSessionRepo) -> None:
        self.store = store

    def begin_turn(
        self,
        agent: SessionBoundAgent,
        display_question: str,
    ) -> str:
        message_id = self.store.append("user", display_question)
        self.bind_agent(agent, message_id)
        return message_id

    def bind_agent(
        self,
        agent: SessionBoundAgent,
        message_id: str | None = None,
    ) -> None:
        session_id = self.store.session_id
        agent.session_id = session_id
        agent.set_history_session_id(session_id)
        agent.set_compaction_source_message_id(message_id)

    def record_event(self, event: AgentHarnessEvent) -> None:
        if event.type not in _DURABLE_EVENT_TYPES:
            return
        self.store.append("event", encode_session_event(event))
        if isinstance(event, FinalStructuredEvent) and event.data.answer:
            self.record_assistant(event.data.answer)

    def record_assistant(self, text: str) -> None:
        if not text.strip():
            return
        self.store.append("assistant", text)
        self.store.update_summary()

    def record_compaction(
        self,
        *,
        summary: str,
        messages_before: int,
        messages_after: int,
        tokens_before: int,
        tokens_after: int,
    ) -> str:
        """追加一次 compaction epoch，不修改既有 transcript。"""
        return self.store.append(
            "event",
            {
                "schema_version": SESSION_EVENT_SCHEMA_VERSION,
                "type": "compaction",
                "step": 0,
                "data": {
                    "trigger": "manual",
                    "summary": summary,
                    "messages_before": messages_before,
                    "messages_after": messages_after,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                },
                "correlation": {},
            },
        )
