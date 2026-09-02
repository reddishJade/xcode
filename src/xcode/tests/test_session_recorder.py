"""运行时 session 记录器测试。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from xcode.agent.config import AgentContext
from xcode.agent.messages import SystemMessage, UserMessage
from xcode.agent.request import DefaultRequestAssembler
from xcode.coding_agent.app import XcodeApp
from xcode.harness.agent_runtime.config import _build_before_provider_request_closure
from xcode.harness.agent_runtime.events import (
    AgentHarnessEvent,
    FinalStructuredEvent,
    TextDeltaStructuredEvent,
)
from xcode.harness.agent_runtime.result import AgentHarnessResult
from xcode.harness.observability import RuntimeCorrelation
from xcode.harness.session import InboxLane, SessionInbox, SessionStore
from xcode.harness.session.recorder import SessionRecorder
from xcode.harness.session.subagent_runs import SubagentRunEvent


class _Agent:
    def __init__(self, inbox: SessionInbox) -> None:
        self._session_id = "local"
        self.inbox = inbox
        self.history_session_id = ""
        self.questions: list[str] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    def set_history_session_id(self, session_id: str) -> None:
        self.history_session_id = session_id

    def followup(
        self,
        message: UserMessage,
        *,
        display_text: str | None = None,
    ) -> None:
        self.inbox.insert(
            message,
            InboxLane.NEXT_TURN,
            display_text=display_text,
            wake=True,
        )

    def run_stream(
        self,
        question: str | None,
        mode: object = None,
        *,
        display_question: str | None = None,
    ) -> Iterator[AgentHarnessEvent]:
        del mode, display_question
        assert question is None
        claimed = self.inbox.claim_initial("fake-run")
        assert len(claimed) == 1
        content = claimed[0].content
        assert isinstance(content, str)
        self.questions.append(content)
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
    return SessionRecorder(SessionStore(tmp_path / "sessions", project_root=tmp_path))


def test_bind_agent_uses_real_session_identity(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    agent = _Agent(SessionInbox(recorder.store))

    recorder.bind_agent(agent)

    assert agent.session_id == recorder.store.session_id
    assert agent.history_session_id == recorder.store.session_id


def test_app_records_programmatic_turn_without_stream_fragments(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    agent = _Agent(SessionInbox(recorder.store))
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
    assert [entry.type for entry in branch] == [
        "event",
        "event",
        "event",
        "assistant",
    ]
    assert branch[0].content["type"] == "inbox/inserted"
    assert branch[0].content["data"]["display_text"] == "visible question"
    assert branch[1].content["type"] == "inbox/claimed"
    assert branch[2].content["type"] == "final"
    assert branch[3].content == "done"


def test_context_reset_appends_epoch_without_rewriting_history(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    inbox = SessionInbox(recorder.store)
    inbox.insert(
        UserMessage(content="keep this verbatim"), InboxLane.NEXT_TURN, wake=True
    )
    inbox.claim_initial("run-1")
    transcript = recorder.store.current_path
    original = transcript.read_bytes()

    recorder.record_context_window_reset(
        window_id="window-2",
        messages_before=12,
        messages_after=4,
        replacement=[UserMessage(content="current state")],
    )

    updated = transcript.read_bytes()
    assert updated.startswith(original)
    assert len(updated) > len(original)
    event = recorder.store.build_branch()[-1].content
    assert isinstance(event, dict)
    assert event["type"] == "context_window_reset"
    assert event["data"]["window_id"] == "window-2"
    assert event["data"]["trigger"] == "manual"
    assert event["data"]["generation"] == 1
    assert len(event["data"]["surface_sha256"]) == 64


def test_provider_request_records_exact_model_visible_envelope(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    inbox = SessionInbox(recorder.store)
    inbox.insert(UserMessage(content="question"), InboxLane.NEXT_TURN, wake=True)
    inbox.claim_initial("run-1")
    recorder.record_provider_request(
        SimpleNamespace(
            metadata={
                "messages": [{"role": "system", "content": "rules"}],
                "tools": [{"name": "read_file", "parameters": {}}],
                "provider": {"model": "test-model", "transport": "test"},
                "prompt_sha256": "prompt-hash",
                "request_sha256": "request-hash",
            },
            timestamp="2026-01-01T00:00:00+00:00",
            session_id=recorder.store.session_id,
            turn_id="turn-1",
            request_id="request-1",
        )
    )

    event = recorder.store.build_branch()[-1].content
    assert isinstance(event, dict)
    assert event["type"] == "provider_request"
    assert event["data"]["messages"] == [{"role": "system", "content": "rules"}]
    assert event["correlation"]["request_id"] == "request-1"


def test_provider_request_hook_adds_provider_and_request_fingerprint() -> None:
    records: list[object] = []
    provider = SimpleNamespace(
        model="test-model",
        base_url="https://provider.invalid",
        transport="test",
        thinking=False,
        reasoning_effort="low",
    )
    correlation = RuntimeCorrelation("session-1")
    correlation.begin_turn()
    closure = _build_before_provider_request_closure(
        cast(Any, records.append),
        lambda: "prompt-v1",
        correlation,
        cast(Any, provider),
        "generation-1",
    )

    assembly = DefaultRequestAssembler().assemble(
        AgentContext(messages=[SystemMessage(content="rules")]),
        current_step=1,
        options=None,
    )
    closure(assembly)

    record = cast(Any, records[0])
    assert record.metadata["provider"] == {
        "model": "test-model",
        "base_url": "https://provider.invalid",
        "transport": "test",
        "thinking": False,
        "reasoning_effort": "low",
    }
    assert record.metadata["assembly"] == {
        "current_step": 1,
        "hygiene_applied": True,
        "estimated_tokens": 1,
        "token_budget": 0,
        "budget_remaining": 0,
        "context_trace": [],
    }
    assert record.metadata["options"] == {}
    assert record.metadata["composition_id"] == "generation-1"
    assert len(record.metadata["request_sha256"]) == 64
    assert record.request_id == "session-1:request:1"


def test_subagent_lifecycle_records_parent_session_lineage(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_subagent_run(
        SubagentRunEvent(
            run_id="run-1",
            activation_id="activation-1",
            child_session_id="child-1",
            batch_id="batch-1",
            task_index=1,
            description="inspect runtime",
            subagent_type="coding",
            mode="one_shot",
            status="started",
        )
    )

    event = recorder.store.build_branch()[-1].content
    assert isinstance(event, dict)
    assert event["type"] == "subagent_run"
    assert event["data"]["parent_session_id"] == recorder.store.session_id
    assert event["data"]["run_id"] == "run-1"
    assert event["data"]["child_session_id"] == "child-1"
