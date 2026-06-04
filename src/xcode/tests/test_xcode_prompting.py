from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from xcode.harness.agent_runtime import (
    ContextualRetrievalState,
    PromptContext,
    SystemPromptBuilder,
    build_runtime_context_provider,
)
from xcode.harness.agent_runtime.git_preflight import build_git_preflight
from xcode.harness.agent_runtime.prompting import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
from xcode.harness.skill_loader import SkillLoader
from xcode.harness.skills import ToolSpec


def _echo_handler(text: dict[str, object]) -> str:
    return str(text)


class XcodePromptingTests(unittest.TestCase):
    def test_builder_includes_stable_modules_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Use tests.", encoding="utf-8")
            tool = ToolSpec(
                "echo",
                "Echo input.",
                "text",
                _echo_handler,
                examples=[{"input": "hello"}],
            )

            prompt = SystemPromptBuilder().build(
                PromptContext(project_root=root, registry=(tool,), question="hello")
            )

            self.assertLess(
                prompt.index("You are Xcode"), prompt.index("Available tools")
            )
            self.assertIn("echo", prompt)
            self.assertIn('Example: {"input": "hello"}', prompt)
            self.assertIn("prefer edit_file", prompt)
            self.assertIn("<environment>", prompt)
            self.assertIn("<git-preflight>", prompt)
            self.assertIn("<search-strategy>", prompt)
            self.assertIn("smallest targeted change", prompt)
            self.assertIn("<cwd-info>", prompt)
            self.assertIn("Use tests.", prompt)
            self.assertLess(
                prompt.index("<search-strategy>"),
                prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY),
            )
            self.assertLess(
                prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY),
                prompt.index("<environment>"),
            )

    def test_volatile_context_changes_do_not_rewrite_stable_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool = ToolSpec("echo", "Echo input.", "text", _echo_handler)
            first_state = ContextualRetrievalState(root)
            first_state.record_file("src/xcode/main.py")
            second_state = ContextualRetrievalState(root)
            second_state.record_file("src/xcode/core/app.py")

            first_prompt = SystemPromptBuilder().build(
                PromptContext(
                    project_root=root,
                    registry=(tool,),
                    question="hello",
                    contextual_state=first_state,
                    resumed_notice="first resume",
                )
            )
            second_prompt = SystemPromptBuilder().build(
                PromptContext(
                    project_root=root,
                    registry=(tool,),
                    question="hello",
                    contextual_state=second_state,
                    interrupted_notice="second interrupt",
                )
            )

            first_stable_prefix = first_prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1)[
                0
            ]
            second_stable_prefix = second_prompt.split(
                SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1
            )[0]
            first_dynamic_suffix = first_prompt.split(
                SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1
            )[1]
            second_dynamic_suffix = second_prompt.split(
                SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1
            )[1]

            self.assertEqual(first_stable_prefix, second_stable_prefix)
            self.assertNotEqual(first_dynamic_suffix, second_dynamic_suffix)
            self.assertIn("src/xcode/main.py", first_dynamic_suffix)
            self.assertIn("src/xcode/core/app.py", second_dynamic_suffix)

    def test_builder_includes_skill_catalog_without_full_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "skills" / "review"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: review\n"
                "description: Review code.\n"
                "use_when: review diffs, review changes\n"
                "dont_use_when: generate release notes\n"
                "---\n\n"
                "Review workflow.",
                encoding="utf-8",
            )
            loader = SkillLoader(root / "skills")

            prompt = SystemPromptBuilder().build(
                PromptContext(
                    project_root=root,
                    registry=(),
                    question="please do a code review",
                    skill_loader=loader,
                )
            )

            self.assertIn("<skill-catalog>", prompt)
            self.assertIn('<skill name="review"', prompt)
            self.assertIn("use_when: review diffs; review changes", prompt)
            self.assertIn("dont_use_when: generate release notes", prompt)
            self.assertIn('load_skill({"name": "review"})', prompt)
            self.assertNotIn("Review workflow.", prompt)

    def test_runtime_context_provider_adds_notices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = build_runtime_context_provider(
                Path(tmp),
                (),
                resumed_notice=lambda: "resumed session",
                interrupted_notice=lambda: "previous run interrupted",
            )

            context = provider("hello")[0]

            self.assertIn("resumed session", context)
            self.assertIn("previous run interrupted", context)

    def test_runtime_context_provider_adds_contextual_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = ContextualRetrievalState(Path(tmp))
            state.record_file("src/xcode/main.py")
            state.record_file("src/xcode/harness/app.py")
            state.record_tool_result("grep_search", "src/xcode/main.py:10:def main")
            state.record_tool_call(
                tool="write_file",
                input_brief="write_file: tmp/hello.py",
                status="ok",
                risk="high",
                approval_scope="once",
                target_path="tmp/hello.py",
            )
            provider = build_runtime_context_provider(
                Path(tmp),
                (),
                contextual_state=state,
            )

            context = provider("hello")[0]

            self.assertIn("<contextual-retrieval>", context)
            self.assertIn("active_file: src/xcode/harness/app.py", context)
            self.assertIn("src/xcode/main.py", context)
            self.assertIn("src/xcode/harness/app.py", context)
            self.assertIn("grep_search", context)
            self.assertIn("recent_tool_calls:", context)
            self.assertIn("write_file", context)
            self.assertIn("approval=once", context)

    def test_git_preflight_reports_non_git_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = build_git_preflight(Path(tmp))

            self.assertIn("<git-preflight>", text)
            self.assertIn("status: unavailable", text)

    def test_git_preflight_includes_dirty_diff_stat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git(root, "init")
            _git(root, "config", "user.email", "test@example.com")
            _git(root, "config", "user.name", "Test User")
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            _git(root, "commit", "-m", "initial")
            (root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")

            text = build_git_preflight(root)

            self.assertIn("status:", text)
            self.assertIn("M a.txt", text)
            self.assertIn("last_commit:", text)
            self.assertIn("dirty_diff_stat:", text)
            self.assertIn("pre-existing changes", text)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    unittest.main()
