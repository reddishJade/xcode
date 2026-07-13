"""Active run 生命周期与 session 调度测试。"""

from __future__ import annotations

import pytest

from xcode.agent.agent import Agent
from xcode.agent.messages import SystemMessage, UserMessage
from xcode.harness.agent_runtime.cancellation import CancellationToken
from xcode.harness.agent_runtime.run_control import (
    ActiveRunState,
    BusyMessageMode,
    SessionRunController,
    SubmitStatus,
)


def _begin() -> tuple[SessionRunController, Agent, CancellationToken]:
    controller = SessionRunController("session-a")
    agent = Agent([])
    token = CancellationToken()
    controller.begin_run(agent, token)
    return controller, agent, token


def test_active_run_has_identity_and_lifecycle() -> None:
    controller, _, _ = _begin()
    handle = controller.active_run()

    assert handle is not None
    assert handle.run_id == "session-a:run:1"
    assert handle.session_id == "session-a"
    assert handle.state() is ActiveRunState.RUNNING

    handle.begin_finishing()
    assert handle.state() is ActiveRunState.FINISHING

    controller.complete_run(handle)
    assert handle.state() is ActiveRunState.FINISHED
    assert handle.wait_finished(0)
    assert controller.active_run() is None


def test_steer_is_bound_to_active_run() -> None:
    controller, agent, _ = _begin()
    outcome = controller.submit(UserMessage(content="change direction"))

    assert outcome.status is SubmitStatus.STEER_ACCEPTED
    assert agent._drain_steer_queue() == [UserMessage(content="change direction")]


def test_internal_system_steer_is_preserved() -> None:
    controller, agent, _ = _begin()
    handle = controller.active_run()
    assert handle is not None

    outcome = handle.steer(SystemMessage(content="runtime reminder"))

    assert outcome.status is SubmitStatus.STEER_ACCEPTED
    assert agent._drain_steer_queue() == [SystemMessage(content="runtime reminder")]


def test_late_steer_falls_back_to_next_run() -> None:
    controller, agent, _ = _begin()
    handle = controller.active_run()
    assert handle is not None
    agent.close_steering()

    outcome = controller.submit(UserMessage(content="do this next"))

    assert outcome.status is SubmitStatus.FOLLOW_UP_QUEUED
    controller.complete_run(handle)
    assert controller.take_follow_up() == UserMessage(content="do this next")


def test_follow_up_waits_until_run_is_finished() -> None:
    controller, _, _ = _begin()
    handle = controller.active_run()
    assert handle is not None

    outcome = controller.submit(
        UserMessage(content="next task"), BusyMessageMode.FOLLOW_UP
    )

    assert outcome.status is SubmitStatus.FOLLOW_UP_QUEUED
    assert controller.take_follow_up() is None
    controller.complete_run(handle)
    assert controller.take_follow_up() == UserMessage(content="next task")


def test_interrupt_keeps_run_active_until_completion() -> None:
    controller, _, token = _begin()
    handle = controller.active_run()
    assert handle is not None

    outcome = controller.submit(
        UserMessage(content="replacement"), BusyMessageMode.INTERRUPT
    )

    assert outcome.status is SubmitStatus.INTERRUPT_REQUESTED
    assert token.is_cancelled()
    assert handle.state() is ActiveRunState.CANCELLING
    assert controller.active_run() is handle
    assert controller.take_follow_up() is None

    handle.begin_finishing()
    controller.complete_run(handle)
    assert controller.take_follow_up() == UserMessage(content="replacement")


def test_collect_merges_messages_after_current_run() -> None:
    controller, _, _ = _begin()
    handle = controller.active_run()
    assert handle is not None
    controller.submit(UserMessage(content="first"), BusyMessageMode.COLLECT)
    controller.submit(UserMessage(content="second"), BusyMessageMode.COLLECT)

    controller.complete_run(handle)

    assert controller.take_follow_up() == UserMessage(content="first\n\nsecond")


def test_unconsumed_steer_is_not_lost() -> None:
    controller, _, _ = _begin()
    handle = controller.active_run()
    assert handle is not None

    controller.complete_run(handle, [UserMessage(content="late")])

    assert controller.take_follow_up() == UserMessage(content="late")


def test_overlapping_runs_are_rejected() -> None:
    controller, _, token = _begin()

    with pytest.raises(RuntimeError, match="active run"):
        controller.begin_run(Agent([]), token)
