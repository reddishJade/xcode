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
from xcode.harness.skills import ToolSpec


def _echo_handler(text: dict[str, object]) -> str:
    return str(text)


class XcodePromptingTests(unittest.TestCase):
    def test_builder_includes_stable_modules_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
