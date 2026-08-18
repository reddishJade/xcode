"""由运行时拥有的 session 记录器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from xcode.agent.messages import AgentMessage
from xcode.harness.agent_runtime.events import AgentHarnessEvent, FinalStructuredEvent

from .event_codec import SESSION_EVENT_SCHEMA_VERSION, encode_session_event
from .tree_store import TreeSessionRepo
from .types import JsonValue
from .subagent_runs import SubagentRunEvent
from .surface import encode_surface_messages, surface_digest


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


class ProviderRequestRecord(Protocol):
    @property
    def metadata(self) -> Mapping[str, object] | None: ...

    @property
    def timestamp(self) -> str: ...

    @property
    def session_id(self) -> str: ...

    @property
    def turn_id(self) -> str: ...

    @property
    def request_id(self) -> str: ...


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
        self.bind_agent(agent)
        return message_id

    def bind_agent(
        self,
        agent: SessionBoundAgent,
    ) -> None:
        session_id = self.store.session_id
        agent.session_id = session_id
        agent.set_history_session_id(session_id)

    def record_event(self, event: AgentHarnessEvent) -> None:
        if event.type not in _DURABLE_EVENT_TYPES:
            return
        encoded = encode_session_event(event)
        if event.type == "compaction":
            data = encoded.get("data")
            if not isinstance(data, dict):
                raise TypeError("compaction event data must be an object")
            replacement = list(event.data.replacement)
            data.update(self._surface_metadata(replacement))
        self.store.append("event", encoded)
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
        replacement: list[AgentMessage],
    ) -> str:
        """追加一次完整 surface replacement，不修改既有 transcript。"""
        metadata = self._surface_metadata(replacement)
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
                    "replacement": encode_surface_messages(replacement),
                    **metadata,
                },
                "correlation": {},
            },
        )

    def _surface_metadata(
        self,
        replacement: list[AgentMessage],
    ) -> dict[str, JsonValue]:
        branch = self.store.build_branch()
        generation = 1 + sum(
            1
            for entry in branch
            if entry.type == "event"
            and isinstance(entry.content, dict)
            and entry.content.get("type") == "compaction"
        )
        return {
            "generation": generation,
            "source_entry_ids": [entry.id for entry in branch],
            "surface_sha256": surface_digest(replacement),
        }

    def record_provider_request(self, record: ProviderRequestRecord) -> None:
        """保存 provider 实际收到的消息、工具和运行参数。"""
        metadata = record.metadata or {}
        self.store.append(
            "event",
            {
                "schema_version": SESSION_EVENT_SCHEMA_VERSION,
                "type": "provider_request",
                "step": 0,
                "data": _json_value(metadata),
                "correlation": {
                    "timestamp": record.timestamp,
                    "session_id": record.session_id,
                    "turn_id": record.turn_id,
                    "request_id": record.request_id,
                    "tool_call_id": "",
                },
            },
        )

    def record_subagent_run(self, event: SubagentRunEvent) -> str:
        """追加子代理生命周期事件，并记录父 session 归属。"""
        data = event.model_dump()
        data["parent_session_id"] = self.store.session_id
        return self.store.append(
            "event",
            {
                "schema_version": SESSION_EVENT_SCHEMA_VERSION,
                "type": "subagent_run",
                "step": 0,
                "data": data,
                "correlation": {},
            },
        )


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    raise TypeError(f"provider request contains non-JSON value: {type(value).__name__}")
