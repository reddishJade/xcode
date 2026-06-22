from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

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
import pytest
def _echo_handler(text: dict[str, object]) -> str:
    return str(text)

class XcodePromptingTests:
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
            assert prompt.startswith("# Identity\n\nYou are Xcode")
            assert "preserve user-owned changes" in prompt
            assert "validate changed behavior" in prompt
            assert "## Operating Principles" in prompt
            assert "## Communication Contract" in prompt
            assert "## Coding Contract" in prompt
            assert "## Tool And Evidence Discipline" in prompt
            assert "## Editing Safety" in prompt
            assert "## Validation Contract" in prompt
            assert "## Review Mode" in prompt
            assert "## Prompt Boundary Discipline" in prompt
            assert "Do not invent command output" in prompt
            assert "Do not edit generated files directly" in prompt
            assert "Lead with findings ordered by severity" in prompt
            assert prompt.index("<tool-discipline>") < prompt.index("Available tools")
            assert prompt.index("Available tools") < prompt.index("<search-strategy>")
            assert "echo" in prompt
            assert "- echo: Echo text" in prompt
            assert "Guidelines:" in prompt
            assert "- Use echo for echo tests." in prompt
            assert "<environment>" in prompt
            assert "<git-preflight>" in prompt
            assert "<search-strategy>" in prompt
            assert "smallest targeted change" in prompt
            assert "<cwd-info>" in prompt
            assert prompt.index("<search-strategy>") < boundary_index
            assert boundary_index < prompt.index("<environment>")
            assert prompt.index("<environment>") < prompt.index("<cwd-info>")
            assert prompt.index("<cwd-info>") < prompt.index("<git-preflight>", boundary_index)

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

            assert first_stable_prefix == second_stable_prefix
            assert first_dynamic_suffix != second_dynamic_suffix
            assert "src/xcode/main.py" in first_dynamic_suffix
            assert "src/xcode/core/app.py" in second_dynamic_suffix

    def test_runtime_context_provider_adds_notices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = build_runtime_context_provider(
                Path(tmp),
                (),
                resumed_notice=lambda: "resumed session",
                interrupted_notice=lambda: "previous run interrupted",
            )

            context = provider("hello")[0]

            assert "resumed session" in context
            assert "previous run interrupted" in context

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

            assert "<contextual-retrieval>" in context
            assert "active_file: src/xcode/harness/app.py" in context
            assert "src/xcode/main.py" in context
            assert "src/xcode/harness/app.py" in context
            assert "grep_search" in context
            assert "recent_tool_calls:" in context
            assert "write_file" in context
            assert "approval=once" in context

    def test_contextual_retrieval_render_reuses_cache_until_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = ContextualRetrievalState(Path(tmp))
            state.record_file("a.py")

            first = state.render()
            second = state.render()
            state.record_file("b.py")
            third = state.render()

            assert first is second
            assert "a.py" in first
            assert first != third
            assert "b.py" in third

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

            assert "a.txt" in first
            assert "b.txt" not in first
            assert "b.txt" in second

    def test_git_preflight_reports_non_git_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = build_git_preflight(Path(tmp))

            assert "<git-preflight>" in text
            assert "status: unavailable" in text

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

            assert "status:" in text
            assert "M a.txt" in text
            assert "last_commit:" in text
            assert "dirty_diff_stat:" in text
            assert "pre-existing changes" in text

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

            assert first == second
            assert calls.count(("status", "--short")) == 2
            assert calls.count(("show", "--stat", "--oneline", "-1")) == 1

def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

if __name__ == "__main__":
    pytest.main()
