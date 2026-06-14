from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from xcode.harness.agent_runtime.contextual import ContextualRetrievalState
from xcode.harness.agent_runtime.prompting import (
    PromptContext,
    SystemPromptBuilder,
    build_runtime_context_provider,
)
from xcode.harness.agent_runtime.git_preflight import build_git_preflight
from xcode.harness.agent_runtime.prompting.identity import (
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
)
from xcode.harness.agent_runtime.prompting.token_budget import MAX_INSTRUCTION_BYTES
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
                prompt_snippet="Echo text",
                prompt_guidelines=("Use echo for echo tests.",),
            )

            prompt = SystemPromptBuilder().build(
                PromptContext(project_root=root, registry=(tool,), question="hello")
            )

            boundary_index = prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
            self.assertTrue(prompt.startswith("# Identity\n\nYou are Xcode"))
            self.assertIn("preserve user-owned changes", prompt)
            self.assertIn("validate changed behavior", prompt)
            self.assertIn("## Operating Principles", prompt)
            self.assertIn("## Communication Contract", prompt)
            self.assertIn("## Coding Contract", prompt)
            self.assertIn("## Tool And Evidence Discipline", prompt)
            self.assertIn("## Editing Safety", prompt)
            self.assertIn("## Validation Contract", prompt)
            self.assertIn("## Review Mode", prompt)
            self.assertIn("## Prompt Boundary Discipline", prompt)
            self.assertIn("Do not invent command output", prompt)
            self.assertIn("Do not edit generated files directly", prompt)
            self.assertIn("Lead with findings ordered by severity", prompt)
            self.assertLess(
                prompt.index("You are Xcode"),
                prompt.index('<instruction-source name="AGENTS.md"'),
            )
            self.assertLess(
                prompt.index('<instruction-source name="AGENTS.md"'),
                prompt.index("<tool-discipline>"),
            )
            self.assertLess(
                prompt.index("<tool-discipline>"), prompt.index("Available tools")
            )
            self.assertLess(
                prompt.index("Available tools"), prompt.index("<search-strategy>")
            )
            self.assertIn("echo", prompt)
            self.assertIn("- echo: Echo text", prompt)
            self.assertIn("Guidelines:", prompt)
            self.assertIn("- Use echo for echo tests.", prompt)
            self.assertIn("<environment>", prompt)
            self.assertIn("<git-preflight>", prompt)
            self.assertIn("<search-strategy>", prompt)
            self.assertIn("smallest targeted change", prompt)
            self.assertIn("<cwd-info>", prompt)
            self.assertIn("Use tests.", prompt)
            self.assertLess(
                prompt.index("<search-strategy>"),
                boundary_index,
            )
            self.assertLess(boundary_index, prompt.index("<environment>"))
            self.assertLess(prompt.index("<environment>"), prompt.index("<cwd-info>"))
            self.assertLess(
                prompt.index("<cwd-info>"),
                prompt.index("<git-preflight>", boundary_index),
            )

    def test_large_project_instruction_warns_without_condensing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instruction = "A" * (24 * 1024 + 1)
            (root / "AGENTS.md").write_text(instruction, encoding="utf-8")

            prompt = SystemPromptBuilder().build(
                PromptContext(project_root=root, registry=(), question="hello")
            )

            self.assertIn("<instruction-warning>", prompt)
            self.assertIn("above the 24576 byte warning threshold", prompt)
            self.assertIn(instruction, prompt)

    def test_oversized_project_instruction_preserves_key_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_example = "\n".join(f"print({index})" for index in range(5000))
            instruction = (
                "# Xcode Agent Guide\n\n"
                "Opening context must stay.\n\n"
                "| Document | Purpose | When to read |\n"
                "| --- | --- | --- |\n"
                "| [docs/git-workflow.md](docs/git-workflow.md) | Git | Before commit |\n\n"
                "## Background\n\n"
                + ("background " * 4000)
                + "\n\n## Python Coding Principles\n\n"
                "Every function must have type annotations.\n\n"
                "```python\n" + long_example + "\n```\n\n"
                "## Git Safety\n\n"
                "Never rewrite history without confirmation.\n\n"
                "## Validation\n\n"
                "Run targeted validation for modified files.\n"
            )
            (root / "AGENTS.md").write_text(instruction, encoding="utf-8")

            prompt = SystemPromptBuilder().build(
                PromptContext(project_root=root, registry=(), question="hello")
            )

            start = prompt.index('<instruction-source name="AGENTS.md"')
            end = prompt.index("</instruction-source>", start)
            source_prompt = prompt[start:end]

            self.assertIn("above the 32768 byte hard limit", prompt)
            self.assertIn("Opening context must stay.", source_prompt)
            self.assertIn("docs/git-workflow.md", source_prompt)
            self.assertIn("Every function must have type annotations.", source_prompt)
            self.assertIn("Never rewrite history without confirmation.", source_prompt)
            self.assertIn("Run targeted validation for modified files.", source_prompt)
            self.assertIn("<instruction-omissions>", source_prompt)
            self.assertNotIn("print(4999)", source_prompt)
            self.assertLessEqual(
                len(source_prompt.encode("utf-8")),
                MAX_INSTRUCTION_BYTES + 800,
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

    def test_contextual_retrieval_render_reuses_cache_until_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = ContextualRetrievalState(Path(tmp))
            state.record_file("a.py")

            first = state.render()
            second = state.render()
            state.record_file("b.py")
            third = state.render()

            self.assertIs(first, second)
            self.assertIn("a.py", first)
            self.assertNotEqual(first, third)
            self.assertIn("b.py", third)

    def test_cwd_info_cache_invalidates_when_visible_entries_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            builder = SystemPromptBuilder()
            context = PromptContext(
                project_root=root,
                registry=(),
                question="hello",
                modules=("identity", "cwd"),
            )

            first = builder.build(context)
            (root / "b.txt").write_text("b", encoding="utf-8")
            second = builder.build(context)

            self.assertIn("a.txt", first)
            self.assertNotIn("b.txt", first)
            self.assertIn("b.txt", second)

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

    def test_git_preflight_reuses_snapshot_cache_after_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _git(root, "init")
            _git(root, "config", "user.email", "test@example.com")
            _git(root, "config", "user.name", "Test User")
            (root / "a.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "a.txt")
            _git(root, "commit", "-m", "initial")

            from xcode.harness.agent_runtime import git_preflight

            git_preflight._ttl_cache.clear()
            git_preflight._snapshot_cache.clear()
            calls: list[tuple[str, ...]] = []
            original = git_preflight._run_git

            def counting_run(project_root: Path, *args: str) -> str | None:
                calls.append(args)
                return original(project_root, *args)

            with mock.patch.object(
                git_preflight,
                "_run_git",
                side_effect=counting_run,
            ):
                first = git_preflight.build_git_preflight(root)
                git_preflight._ttl_cache.clear()
                second = git_preflight.build_git_preflight(root)

            self.assertEqual(first, second)
            self.assertEqual(calls.count(("status", "--short")), 2)
            self.assertEqual(calls.count(("show", "--stat", "--oneline", "-1")), 1)


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
