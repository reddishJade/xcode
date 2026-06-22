from __future__ import annotations

import tempfile
from pathlib import Path
from xcode.harness.agent_runtime.compaction import (
    LayeredCompactor,
    context_collapse_clean,
)
from xcode.harness.agent_runtime.prompting import PromptContext, SystemPromptBuilder
from xcode.harness.task_store import TaskStore
import pytest


class XcodeContextRestorationTests:
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
        assert cleaned == "[Compressed]\nDetailed summary of user's task and progress."

    def test_context_collapse_clean_strips_thinking_blocks(self) -> None:
        raw_text = (
            "<think>Thinking process to exclude</think>\n"
            "<analysis>Analysis block to exclude</analysis>\n"
            "This is the actual summary text."
        )
        cleaned = context_collapse_clean(raw_text)
        assert cleaned == "This is the actual summary text."

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
            assert compacted[1]["content"] == "[Compressed]\nClean dynamic summary"
            assert captured_summaries[0] == "[Compressed]\nClean dynamic summary"

    def test_active_metadata_restoration_injections(self) -> None:
        """Task state no longer enters through system prompt (removed dual injection).

        Task state now enters exclusively through TaskStateCollector via the
        context pipeline, not through _build_post_compact_metadata in the
        system prompt. Verify that the system prompt does not contain
        <active-tasks-graph> or <post-compact-metadata>.
        """
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)

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

            context = PromptContext(
                project_root=project_root,
                registry=(),
                question="How to build?",
            )

            builder = SystemPromptBuilder()
            prompt = builder.build(context)

            # Task state must NOT appear in system prompt — it enters through
            # TaskStateCollector (USER_CONTEXT) via the context pipeline only.
            assert "<active-tasks-graph>" not in prompt
            assert "<post-compact-metadata>" not in prompt


if __name__ == "__main__":
    pytest.main()
