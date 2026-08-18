"""Active run 生命周期与 durable inbox 调度测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from xcode.agent.messages import SystemMessage, UserMessage
from xcode.harness.agent_runtime.cancellation import CancellationToken
from xcode.harness.agent_runtime.run_control import (
    ActiveRunState,
    BusyMessageMode,
    SessionRunController,
    SubmitStatus,
)
from xcode.harness.session import SessionInbox, SessionStore


def _begin(
    tmp_path: Path,
) -> tuple[SessionRunController, CancellationToken]:
    store = SessionStore(tmp_path / "sessions", project_root=tmp_path)
    controller = SessionRunController(SessionInbox(store))
    token = CancellationToken()
    controller.begin_run(token)
    return controller, token


def test_active_run_has_identity_and_lifecycle(tmp_path: Path) -> None:
    controller, _ = _begin(tmp_path)
    handle = controller.active_run()

    assert handle is not None
    assert handle.run_id
    assert handle.session_id == controller.session_id
    assert handle.state() is ActiveRunState.RUNNING

    handle.begin_finishing()
    assert handle.state() is ActiveRunState.FINISHING

    controller.complete_run(handle)
    assert handle.state() is ActiveRunState.FINISHED
    assert handle.wait_finished(0)
    assert controller.active_run() is None


def test_steer_is_claimed_at_active_run_boundary(tmp_path: Path) -> None:
    controller, _ = _begin(tmp_path)
    handle = controller.active_run()
    assert handle is not None

    outcome = controller.submit(UserMessage(content="change direction"))

    assert outcome.status is SubmitStatus.STEER_ACCEPTED
    assert handle.claim_step_input() == [UserMessage(content="change direction")]


def test_internal_system_injection_is_durable(tmp_path: Path) -> None:
    controller, _ = _begin(tmp_path)
    handle = controller.active_run()
    assert handle is not None

    outcome = controller.inject_runtime(SystemMessage(content="runtime reminder"))

    assert outcome.status is SubmitStatus.STEER_ACCEPTED
    assert handle.claim_step_input() == [SystemMessage(content="runtime reminder")]


def test_late_steer_falls_back_to_waking_next_run(tmp_path: Path) -> None:
    controller, token = _begin(tmp_path)
    handle = controller.active_run()
    assert handle is not None
    handle.finish_step_input()

    outcome = controller.submit(UserMessage(content="do this next"))

    assert outcome.status is SubmitStatus.INJECT_QUEUED
    controller.complete_run(handle)
    assert controller.has_waking_input()
    next_handle = controller.begin_run(token)
    assert controller.claim_initial(next_handle) == [
        UserMessage(content="do this next")
    ]


def test_followup_waits_for_a_new_run(tmp_path: Path) -> None:
    controller, token = _begin(tmp_path)
    handle = controller.active_run()
    assert handle is not None

    outcome = controller.submit(
        UserMessage(content="next task"), BusyMessageMode.FOLLOW_UP
    )

    assert outcome.status is SubmitStatus.FOLLOW_UP_QUEUED
    assert handle.claim_step_input() == []
    controller.complete_run(handle)
    next_handle = controller.begin_run(token)
    assert controller.claim_initial(next_handle) == [UserMessage(content="next task")]


def test_interrupt_keeps_run_active_and_queues_replacement(tmp_path: Path) -> None:
    controller, token = _begin(tmp_path)
    handle = controller.active_run()
    assert handle is not None

    outcome = controller.submit(
        UserMessage(content="replacement"), BusyMessageMode.INTERRUPT
    )

    assert outcome.status is SubmitStatus.INTERRUPT_REQUESTED
    assert token.is_cancelled()
    assert handle.state() is ActiveRunState.CANCELLING
    assert controller.active_run() is handle

    handle.begin_finishing()
    controller.complete_run(handle)
    next_handle = controller.begin_run(token)
    assert controller.claim_initial(next_handle) == [UserMessage(content="replacement")]


def test_inbox_survives_controller_reconstruction(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions", project_root=tmp_path)
    first = SessionRunController(SessionInbox(store))
    first.submit(UserMessage(content="persist me"), BusyMessageMode.FOLLOW_UP)

    restored = SessionRunController(SessionInbox(store))
    handle = restored.begin_run(CancellationToken())

    assert restored.claim_initial(handle) == [UserMessage(content="persist me")]
    event_types = [entry.content["type"] for entry in store.build_branch()]
    assert event_types == ["inbox/inserted", "inbox/claimed"]


def test_overlapping_runs_are_rejected(tmp_path: Path) -> None:
    controller, token = _begin(tmp_path)

    with pytest.raises(RuntimeError, match="active run"):
        controller.begin_run(token)
