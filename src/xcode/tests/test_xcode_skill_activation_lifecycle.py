"""Skill 激活状态恢复与会话压缩测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

from xcode.agent.messages import AssistantMessage, ToolResultMessage
from xcode.cli.repl_sessions import records_to_agent_messages
from xcode.cli.repl_skills import activate_skill
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig, GateConfig
from xcode.harness.observability import PermissionPolicy, StaticPermission
from xcode.harness.session import SessionStore
from xcode.harness.agent_skills import (
    SkillRegistry,
    build_load_skill_tool,
    build_skill_search_dirs,
)
from xcode.tests.fixtures import FakeProvider
import pytest


class _ResettableFakeProvider(FakeProvider):
    """记录显式激活后的 provider 会话重置。"""

    def __init__(self) -> None:
        super().__init__([])
        self.reset_count = 0

    def reset_conversation_state(self) -> None:
        """记录 provider 会话状态重置次数。"""
        self.reset_count += 1


def _skill_registry(root: Path) -> SkillRegistry:
    """创建包含单一可激活技能的测试注册表。"""
    skill_dir = root / ".xcode" / "skills" / "review"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        ("---\nname: code-review\ndescription: Review.\n---\n\nFULL_SKILL_BODY"),
        encoding="utf-8",
    )
    registry = SkillRegistry()
    registry.discover(build_skill_search_dirs(root))
    return registry


class XcodeSkillActivationLifecycleTests:
    """验证 Skill 激活状态可恢复且不会被 transcript 压缩破坏。"""

    def test_explicit_activation_uses_canonical_history_pair(self) -> None:
        """显式激活会执行 load_skill 并记录可保护的工具消息对。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = _skill_registry(root)
            provider = _ResettableFakeProvider()
            agent = StructuredAgent(
                provider=provider,
                registry=(build_load_skill_tool(registry),),
                runtime=AgentRuntimeConfig(skill_registry=registry),
            )

            result = agent.activate_skill("code-review")
            repeated = agent.activate_skill("code-review")
            history = agent.history_messages()

        assert result.status == "activated"
        assert result.tool_call_id is not None
        assert "FULL_SKILL_BODY" in result.content
        assert repeated.status == "already_active"
        assert provider.reset_count == 1
        assert len(history) == 2
        assert isinstance(history[0], AssistantMessage)
        assert isinstance(history[1], ToolResultMessage)
        assert isinstance(history[1], ToolResultMessage)
        assert history[1].tool_call_id == result.tool_call_id
        assert "<skill-activation-state>" in str(history[1].content)

    def test_explicit_activation_reports_unknown_disabled_and_blocked(self) -> None:
        """未知、禁用和权限阻止状态会返回明确诊断。"""
        disabled_agent = StructuredAgent(
            provider=FakeProvider([]),
            registry=(),
        )
        disabled = disabled_agent.activate_skill("code-review")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = _skill_registry(root)
            tool = build_load_skill_tool(registry)
            agent = StructuredAgent(
                provider=FakeProvider([]),
                registry=(tool,),
                gate=GateConfig(
                    permission_policy=PermissionPolicy(
                        (StaticPermission("load_skill", "deny"),)
                    )
                ),
                runtime=AgentRuntimeConfig(skill_registry=registry),
            )
            unknown = agent.activate_skill("missing")
            blocked = agent.activate_skill("code-review")

        assert disabled.status == "disabled"
        assert unknown.status == "unknown"
        assert blocked.status == "blocked"
        assert "Unknown skill" in unknown.message

    def test_explicit_activation_round_trips_through_repl_session(self) -> None:
        """显式激活事件可从 transcript 恢复到同一 activation 状态。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = _skill_registry(root)
            agent = StructuredAgent(
                provider=FakeProvider([]),
                registry=(build_load_skill_tool(registry),),
                runtime=AgentRuntimeConfig(skill_registry=registry),
            )
            store = SessionStore(root / "sessions", project_root=root)

            result = activate_skill(
                SimpleNamespace(agent=agent),
                store,
                "code-review",
            )
            compacted = store.compact_current_session(max_tool_result_chars=10)
            restored_messages = records_to_agent_messages(store.load_records())

            resumed_registry = _skill_registry(root)
            resumed_agent = StructuredAgent(
                provider=FakeProvider([]),
                registry=(build_load_skill_tool(resumed_registry),),
                runtime=AgentRuntimeConfig(skill_registry=resumed_registry),
            )
            resumed_agent.load_history(restored_messages)

        assert result.status == "activated"
        assert compacted == 0
        assert resumed_registry.activated_names() == ("code-review",)
        assert resumed_agent.activate_skill("code-review").status == "already_active"

    def test_registry_restores_activation_from_history(self) -> None:
        """恢复历史后重复加载返回 already-active。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / ".xcode" / "skills" / "review"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: code-review\ndescription: Review.\n---\n\nBODY",
                encoding="utf-8",
            )
            registry = SkillRegistry()
            registry.discover(build_skill_search_dirs(root))
            first = build_load_skill_tool(registry).handler({"name": "code-review"})

            resumed_registry = SkillRegistry()
            resumed_registry.discover(build_skill_search_dirs(root))
            agent = StructuredAgent(
                provider=FakeProvider([]),
                registry=(),
                runtime=AgentRuntimeConfig(skill_registry=resumed_registry),
            )
            agent.load_history(
                [ToolResultMessage(tool_call_id="skill-1", content=first)]
            )
            repeated = build_load_skill_tool(resumed_registry).handler(
                {"name": "code-review"}
            )

        assert resumed_registry.activated_names() == ("code-review",)
        assert 'status="already-active"' in repeated
        assert "BODY" not in repeated

    def test_session_compaction_preserves_activation_result(self) -> None:
        """会话 transcript 压缩不截断技能激活正文。"""
        activation = (
            '<skill name="review" root="C:/skills/review" activated="true">\n'
            '<skill-activation-state>{"name": "review"}</skill-activation-state>\n'
            "FULL_SKILL_BODY\n"
            "</skill>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp) / "sessions", project_root=Path(tmp))
            store.append("user", "review")
            store.append(
                "event",
                {
                    "type": "tool_result",
                    "data": {
                        "tool_use_id": "skill-1",
                        "content": activation,
                        "status": "ok",
                    },
                },
            )

            compacted = store.compact_current_session(max_tool_result_chars=10)
            records = store.load_records()

        assert compacted == 0
        assert "FULL_SKILL_BODY" in str(records[1].content)


if __name__ == "__main__":
    pytest.main()
