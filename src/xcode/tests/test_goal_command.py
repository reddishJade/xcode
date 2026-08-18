"""Goal 斜杠命令测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from xcode.ai.providers.base import ModelProvider
from xcode.cli.commands import CommandContext, ReplState
from xcode.cli.repl_commands import cmd_goal
from xcode.harness.session.replay import latest_goal_state, replay_session
from xcode.cli.tui.app import _is_live_command
from xcode.coding_agent.state import CodingRunState
from xcode.harness.agent_runtime.goal import GoalController, GoalState
from xcode.harness.session.types import JsonValue, SessionEntry
from xcode.harness.session import SessionStore


class _Store:
    def __init__(self) -> None:
        self.events: list[tuple[str, JsonValue]] = []

    def append(self, record_type: str, content: JsonValue) -> str:
        self.events.append((record_type, content))
        return str(len(self.events))


class _ResumeStore:
    session_id = "session-1"

    def __init__(self, records: list[SessionEntry]) -> None:
        self._records = records

    def build_branch(self) -> list[SessionEntry]:
        return self._records


class _ResumeAgent:
    def __init__(self) -> None:
        self.goal = GoalState()
        self.session_id = ""

    def set_history_session_id(self, _session_id: str) -> None:
        return

    def load_history(self, _messages: object) -> None:
        return

    def restore_goal_state(self, payload: object) -> None:
        self.goal = GoalState.from_dict(payload)

    def restore_run_state_metadata(self, _payload: object) -> None:
        return

    def set_resumed_notice(self, _notice: str) -> None:
        return


class _GoalAgent:
    def __init__(self) -> None:
        self._goal = GoalController(cast(ModelProvider, object()))

    def set_goal(self, condition: str) -> None:
        self._goal.set(condition)

    def clear_goal(self) -> None:
        self._goal.clear()

    def pause_goal(self) -> None:
        self._goal.pause()

    def resume_goal(self) -> None:
        self._goal.resume()

    @property
    def goal_state(self) -> dict[str, str | bool | int | None]:
        return self._goal.state.to_dict()

    def restore_goal_state(self, payload: object) -> None:
        self._goal.restore(GoalState.from_dict(payload))

    @property
    def goal_condition(self) -> str | None:
        return self._goal.condition

    @property
    def goal_paused(self) -> bool:
        return self._goal.paused


def _context(agent: _GoalAgent, state: ReplState) -> CommandContext:
    return cast(
        CommandContext,
        SimpleNamespace(
            app=SimpleNamespace(agent=agent),
            state=state,
            store=_Store(),
        ),
    )


def test_goal_pause_and_resume_preserve_objective_and_continue(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _GoalAgent()
    state = ReplState()
    ctx = _context(agent, state)
    cmd_goal("/goal Finish the migration", ctx)
    state.pending_inject = None

    cmd_goal("/goal pause", ctx)

    assert agent.goal_condition == "Finish the migration"
    assert agent.goal_paused is True
    assert capsys.readouterr().out == (
        "Goal set: Finish the migration\nGoal paused: Finish the migration\n"
    )

    cmd_goal("/goal resume", ctx)

    assert agent.goal_condition == "Finish the migration"
    assert agent.goal_paused is False
    assert state.pending_inject == (
        "Continue working toward the active goal:\n\nFinish the migration"
    )
    assert capsys.readouterr().out == "Goal resumed: Finish the migration\n"
    assert cast(_Store, ctx.store).events == [
        (
            "event",
            {
                "type": "goal_state",
                "data": {
                    "condition": "Finish the migration",
                    "paused": False,
                    "reacts": 0,
                },
            },
        ),
        (
            "event",
            {
                "type": "goal_state",
                "data": {
                    "condition": "Finish the migration",
                    "paused": True,
                    "reacts": 0,
                },
            },
        ),
        (
            "event",
            {
                "type": "goal_state",
                "data": {
                    "condition": "Finish the migration",
                    "paused": False,
                    "reacts": 0,
                },
            },
        ),
    ]


def test_goal_status_reports_active_and_paused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _GoalAgent()
    state = ReplState()
    ctx = _context(agent, state)
    agent.set_goal("Run all tests")

    cmd_goal("/goal", ctx)
    agent.pause_goal()
    cmd_goal("/goal", ctx)

    assert capsys.readouterr().out == (
        "Goal: Run all tests [active]\nGoal: Run all tests [paused]\n"
    )


def test_goal_command_is_available_during_active_run() -> None:
    assert _is_live_command("/goal pause") is True
    assert _is_live_command("/model") is False


def test_latest_goal_state_prefers_newer_pause_command_over_final_state() -> None:
    records = [
        SessionEntry(
            id="1",
            parent_id=None,
            type="event",
            content={
                "type": "final",
                "data": {
                    "run_state": {
                        "goal": {
                            "condition": "Finish the migration",
                            "paused": False,
                            "reacts": 1,
                        }
                    }
                },
            },
            created_at="2026-07-24T00:00:00Z",
        ),
        SessionEntry(
            id="2",
            parent_id="1",
            type="event",
            content={
                "type": "goal_state",
                "data": {
                    "condition": "Finish the migration",
                    "paused": True,
                    "reacts": 1,
                },
            },
            created_at="2026-07-24T00:00:01Z",
        ),
    ]

    assert latest_goal_state(records) == {
        "condition": "Finish the migration",
        "paused": True,
        "reacts": 1,
    }


def test_coding_run_state_round_trips_paused_goal() -> None:
    state = CodingRunState(
        messages=[],
        goal=GoalState(
            condition="Finish the migration",
            paused=True,
            reacts=2,
        ),
    )

    restored = CodingRunState.from_dict(state.to_dict())

    assert restored.goal == GoalState(
        condition="Finish the migration",
        paused=True,
        reacts=2,
    )


def test_session_resume_restores_paused_goal() -> None:
    record = SessionEntry(
        id="1",
        parent_id=None,
        type="event",
        content={
            "type": "goal_state",
            "data": {
                "condition": "Finish the migration",
                "paused": True,
                "reacts": 2,
            },
        },
        created_at="2026-07-24T00:00:00Z",
    )
    agent = _ResumeAgent()
    store = cast(SessionStore, _ResumeStore([record]))

    replay_session(cast(Any, agent), store)

    assert agent.goal == GoalState(
        condition="Finish the migration",
        paused=True,
        reacts=2,
    )
