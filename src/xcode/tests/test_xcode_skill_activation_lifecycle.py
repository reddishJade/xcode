"""Skill 激活状态恢复与会话压缩测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from xcode.agent.messages import ToolResultMessage
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.harness.session import SessionStore
from xcode.harness.skills_registry import (
    SkillRegistry,
    build_load_skill_tool,
    build_skill_search_dirs,
)
from xcode.tests.fixtures import FakeProvider


class XcodeSkillActivationLifecycleTests(unittest.TestCase):
    """验证 Skill 激活状态可恢复且不会被 transcript 压缩破坏。"""

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

        self.assertEqual(resumed_registry.activated_names(), ("code-review",))
        self.assertIn('status="already-active"', repeated)
        self.assertNotIn("BODY", repeated)

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

        self.assertEqual(compacted, 0)
        self.assertIn("FULL_SKILL_BODY", str(records[1].content))


if __name__ == "__main__":
    unittest.main()
