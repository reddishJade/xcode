"""真实应用组装与 session 回放契约测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from xcode.ai.events import FinalMessage, Message, ProviderEvent, TextDelta
from xcode.ai.providers.registry import ProviderBundle
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.coding_agent.app import build_app
from xcode.harness.config import XcodeRuntimeConfig


class _ContractProvider:
    model = "contract-model"
    base_url = "https://contract.invalid"
    transport = "contract"
    thinking = False
    reasoning_effort = None

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.requests: list[tuple[list[Message], list[ToolDefinition]]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ):
        del options, kwargs
        self.requests.append((deepcopy(messages), deepcopy(tools)))
        yield cast(ProviderEvent, TextDelta(self.answer))
        yield cast(
            ProviderEvent,
            FinalMessage(content=self.answer, stop_reason="end_turn"),
        )


def _install_contract_provider(
    monkeypatch: pytest.MonkeyPatch,
    providers: list[_ContractProvider],
) -> None:
    def build_bundle(_settings: object) -> ProviderBundle:
        provider = _ContractProvider(f"answer-{len(providers) + 1}")
        providers.append(provider)
        typed = cast(Any, provider)
        return ProviderBundle(
            llm=typed,
            llms={
                "main": typed,
                "subagent": typed,
                "judge": typed,
                "refiner": typed,
            },
        )

    monkeypatch.setattr("xcode.coding_agent.app.build_provider_bundle", build_bundle)


def _provider_request_events(app: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entry in app.session_store.build_branch():
        if (
            entry.type == "event"
            and isinstance(entry.content, dict)
            and entry.content.get("type") == "provider_request"
        ):
            events.append(entry.content)
    return events


def test_real_build_app_minimal_run_and_replay_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers: list[_ContractProvider] = []
    _install_contract_provider(monkeypatch, providers)
    sessions_dir = tmp_path / "sessions"
    runtime_config = XcodeRuntimeConfig()

    first = build_app(
        tmp_path,
        runtime_config=runtime_config,
        sessions_dir=sessions_dir,
    )
    first_answer = first.ask("first question")
    session_id = first.session_store.session_id

    assert first_answer == "answer-1"
    assert first.registry
    assert {tool.name for tool in first.registry} >= {
        "read_file",
        "bash",
        "subagent",
    }
    first_request = providers[0].requests[0]
    first_envelope = _provider_request_events(first)[0]["data"]
    assert first_envelope["messages"] == first_request[0]
    assert first_envelope["tools"] == [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in first_request[1]
    ]
    first.close()

    resumed = build_app(
        tmp_path,
        runtime_config=runtime_config,
        sessions_dir=sessions_dir,
    )
    resumed.session_store.resume(session_id)
    resumed.restore_session()
    second_answer = resumed.ask("second question")

    assert second_answer == "answer-2"
    second_messages = providers[1].requests[0][0]
    assert any(
        message.get("role") == "user" and message.get("content") == "first question"
        for message in second_messages
    )
    assert any(
        message.get("role") == "assistant" and message.get("content") == "answer-1"
        for message in second_messages
    )
    second_envelope = _provider_request_events(resumed)[-1]["data"]
    assert second_envelope["messages"] == second_messages
    resumed.close()
