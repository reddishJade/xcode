"""ToolGate 纯函数单元测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from xcode.agent.request import DefaultRequestAssembler
from xcode.harness.agent_runtime.composition import AgentComposition
from xcode.harness.agent_runtime.config import AgentRuntimeConfig, GateConfig
from xcode.harness.agent_runtime.harness import AgentHarness
from xcode.harness.config import AgentConfig
from xcode.harness.agent_runtime.tool_gate import _permission_notice, _stricter_decision
from xcode.harness.security import PermissionEngineResult
from xcode.harness.security.permission_model import (
    ApprovalResult,
    ExternalDirectory,
    SensitivePathOverride,
)
from xcode.harness.session import SessionInbox, SessionStore


def _runtime(tmp_path: Path) -> AgentRuntimeConfig:
    store = SessionStore(tmp_path / "sessions", project_root=tmp_path)
    return AgentRuntimeConfig(
        session_inbox=SessionInbox(store),
        project_root=tmp_path,
    )


def _composition(
    gate: GateConfig,
    provider: Any | None = None,
) -> AgentComposition:
    return AgentComposition.create(
        primary_provider=cast(Any, provider or object()),
        fallback_provider=None,
        registry=(),
        config=AgentConfig(),
        gate=gate,
        request_assembler=DefaultRequestAssembler(),
        runtime_context_provider=None,
    )


class TestStricterDecision:
    def test_stricter_wins(self) -> None:
        assert _stricter_decision("allow", "deny") == "deny"
        assert _stricter_decision("ask", "deny") == "deny"

    def test_current_is_stricter(self) -> None:
        assert _stricter_decision("deny", "allow") == "deny"
        assert _stricter_decision("deny", "ask") == "deny"

    def test_same_level(self) -> None:
        assert _stricter_decision("allow", "allow") == "allow"
        assert _stricter_decision("ask", "ask") == "ask"


def test_permission_notice_describes_automatic_session_grant() -> None:
    result = PermissionEngineResult(
        decision="allow",
        blocked=False,
        matched_rule="session_grant",
        approval_result=ApprovalResult(
            decision="allow",
            scope="session",
            grant_id="grant-1",
        ),
    )

    assert _permission_notice(result) == "Allowed by session grant"


def test_agent_harness_propagates_external_directories(tmp_path: Path) -> None:
    external = ExternalDirectory(path=tmp_path / "shared", access="read")

    harness = AgentHarness(
        composition=_composition(GateConfig(external_directories=(external,))),
        runtime=_runtime(tmp_path),
    )

    assert harness.external_directories == (external,)
    assert harness._gate.snapshot().external_directories == (external,)


def test_agent_harness_propagates_sensitive_path_overrides(tmp_path: Path) -> None:
    override = SensitivePathOverride(path=tmp_path / ".env", access="read")

    harness = AgentHarness(
        composition=_composition(GateConfig(sensitive_path_overrides=(override,))),
        runtime=_runtime(tmp_path),
    )

    assert harness.sensitive_path_overrides == (override,)
    assert harness._gate.snapshot().sensitive_path_overrides == (override,)


def test_agent_harness_replaces_the_whole_provider_generation(tmp_path: Path) -> None:
    first_provider = object()
    second_provider = object()
    harness = AgentHarness(
        composition=_composition(GateConfig(), first_provider),
        runtime=_runtime(tmp_path),
    )
    first_generation = harness.composition.generation_id

    second_generation = harness.replace_primary_provider(
        cast(Any, second_provider)
    )

    assert second_generation != first_generation
    assert harness.composition.generation_id == second_generation
    assert harness.composition.primary_provider is second_provider
    assert harness.provider is second_provider
