"""运行时 session 记录器测试。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from xcode.coding_agent.app import XcodeApp
from xcode.harness.agent_runtime.events import (
    AgentHarnessEvent,
    FinalStructuredEvent,
    TextDeltaStructuredEvent,
)
from xcode.harness.agent_runtime.result import AgentHarnessResult
from xcode.harness.session import SessionStore
from xcode.harness.session.recorder import SessionRecorder


class _Agent:
    def __init__(self) -> None:
        self._session_id = "local"
        self.history_session_id = ""
        self.source_message_id: str | None = None
        self.questions: list[str] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    def set_history_session_id(self, session_id: str) -> None:
        self.history_session_id = session_id

    def set_compaction_source_message_id(self, message_id: str | None) -> None:
        self.source_message_id = message_id

    def run_stream(
        self,
        question: str,
        mode: object = None,
    ) -> Iterator[AgentHarnessEvent]:
        del mode
        self.questions.append(question)
        yield TextDeltaStructuredEvent("text_delta", 1, "done")
        yield FinalStructuredEvent(
            "final",
            1,
            AgentHarnessResult(
                answer="done",
                messages=[],
                steps=1,
                tool_calls=[],
            ),
        )


def _recorder(tmp_path: Path) -> SessionRecorder:
    return SessionRecorder(
        SessionStore(tmp_path / "sessions", project_root=tmp_path)
    )


def test_begin_turn_binds_agent_to_real_session(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    agent = _Agent()

    message_id = recorder.begin_turn(agent, "visible question")

    assert agent.session_id == recorder.store.session_id
    assert agent.history_session_id == recorder.store.session_id
    assert agent.source_message_id == message_id
    assert recorder.store.build_branch()[0].content == "visible question"


def test_app_records_programmatic_turn_without_stream_fragments(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    agent = _Agent()
    app = XcodeApp(
        agent=cast(Any, agent),
        session_recorder=recorder,
    )

    events = list(
        app.ask_stream(
            "expanded question",
            display_question="visible question",
        )
    )

    assert [event.type for event in events] == ["text_delta", "final"]
    assert agent.questions == ["expanded question"]
    branch = recorder.store.build_branch()
    assert [entry.type for entry in branch] == ["user", "event", "assistant"]
    assert branch[0].content == "visible question"
    assert branch[1].content["type"] == "final"
    assert branch[2].content == "done"
