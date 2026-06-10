from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from xcode.harness.agent_runtime.compaction import (
    LayeredCompactor,
    context_collapse_clean,
)
from xcode.harness.agent_runtime.prompting import PromptContext, SystemPromptBuilder
from xcode.harness.task_store import TaskStore


class XcodeContextRestorationTests(unittest.TestCase):
    def test_context_collapse_clean_with_summary_tags(self) -> None:
        raw_text = (
            "[Compressed]\n"
            "<analysis>\n"
            "We have completed tasks 1 and 2, but step 3 requires more tools.\n"
            "</analysis>\n"
            "<summary>\n"
            "Detailed summary of user's task and progress.\n"
            "</summary>"
        )
        cleaned = context_collapse_clean(raw_text)
        self.assertEqual(
            cleaned, "[Compressed]\nDetailed summary of user's task and progress."
        )

    def test_context_collapse_clean_strips_thinking_blocks(self) -> None:
        raw_text = (
            "<think>Thinking process to exclude</think>\n"
            "<analysis>Analysis block to exclude</analysis>\n"
            "This is the actual summary text."
        )
        cleaned = context_collapse_clean(raw_text)
        self.assertEqual(cleaned, "This is the actual summary text.")

    def test_layered_compactor_applies_context_collapse_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            messages = [{"role": "user", "content": "root"}]
            # Create enough messages to trigger compaction
            for i in range(10):
                messages.append({"role": "assistant", "content": f"msg {i}"})

            # Mock on_compact callback to verify cleaned summary
            captured_summaries = []

            def on_compact_callback(summary: str) -> None:
                captured_summaries.append(summary)

            compactor = LayeredCompactor(
                Path(tmp), max_recent_messages=2, on_compact=on_compact_callback
            )

            # Monkeypatch summarize_messages to return a double-tagged output
            from xcode.harness.agent_runtime import compaction

            original_summarize = compaction.summarize_messages
            try:
                compaction.summarize_messages = lambda msgs, **kwargs: [
                    msgs[0],
                    {
                        "role": "user",
                        "content": "[Compressed]\n<analysis>Thought info</analysis>\n<summary>Clean dynamic summary</summary>",
                    },
                    *msgs[-kwargs.get("max_recent_messages", 2) :],
                ]
                compacted = compactor(messages)
            finally:
                compaction.summarize_messages = original_summarize

            # Verify that both the history message content and the compact callback received the cleaned content
            self.assertEqual(
                compacted[1]["content"], "[Compressed]\nClean dynamic summary"
            )
            self.assertEqual(
                captured_summaries[0], "[Compressed]\nClean dynamic summary"
            )

    def test_active_metadata_restoration_injections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)

            # Setup task store with pending task and checklist
            store = TaskStore(project_root)
            store.create(
                "Implement features",
                payload={
                    "feature_list": [
                        {"title": "Subtask 1", "status": "completed"},
                        {"title": "Subtask 2", "status": "pending"},
                    ],
                    "blocked_by": "Task #99",
                },
            )

            class FakeLoader:
                def get_catalog(self, question: str | None = None) -> str:
                    return (
                        "<skill-catalog>\n"
                        '<skill name="compile_skill" path="skills/compile/SKILL.md" risk="low">\n'
                        "description: Compile helper skill\n"
                        'load: load_skill({"name": "compile_skill"})\n'
                        "</skill>\n"
                        "</skill-catalog>"
                    )

            context = PromptContext(
                project_root=project_root,
                registry=(),
                question="How to build?",
                skill_loader=cast(Any, FakeLoader()),
            )

            builder = SystemPromptBuilder()
            prompt = builder.build(context)

            # Verify active tasks metadata block
            self.assertIn("<active-tasks-graph>", prompt)
            self.assertIn(
                "- [PENDING] Task #1: Implement features (1/2 subtasks completed) [Blocked by: Task #99]",
                prompt,
            )

            # Verify skill catalog block
            self.assertIn("<skill-catalog>", prompt)
            self.assertIn("Compile helper skill", prompt)

            # Verify that all metadata is injected strictly into the Volatile Region (below dynamic boundary)
            boundary_marker = "<system-prompt-dynamic-boundary />"
            self.assertIn(boundary_marker, prompt)

            parts = prompt.split(boundary_marker)
            volatile_region = parts[1]

            self.assertIn("<post-compact-metadata>", volatile_region)
            self.assertNotIn("<post-compact-metadata>", parts[0])


if __name__ == "__main__":
    unittest.main()
