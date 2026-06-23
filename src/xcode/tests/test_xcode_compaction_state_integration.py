"""压缩状态与结构化上下文系统的集成测试。

设计：

* [Compressed] 是规范化的压实对话历史表示，保留在消息列表中。
* TaskStateCollector 仅注入结构化任务状态（状态、标题、blocked_by），
  严禁注入压实摘要内容。
* 两条路径包含不同的内容，互不重叠。
* 压实不复制其他 collector 已处理的内容（项目清单、活动 diff、技能、笔记）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from xcode.harness.agent_runtime.compaction import LayeredCompactor
import pytest
# ── 编译辅助 ──


def _compacted_summary(messages: list[dict[str, Any]]) -> str | None:
    """从压实后的消息列表中提取 [Compressed] 摘要文本（如果存在）。"""
    if len(messages) > 1:
        content = messages[1].get("content", "")
        if isinstance(content, str) and content.startswith("[Compressed]"):
            return content
    return None


class TestCompactedHistoryPath:
    """验证 [Compressed] 保留在消息列表中作为唯一的压实对话历史路径。"""

    def test_compressed_message_remains_in_message_list(self) -> None:
        """[Compressed] 是压实后的对话历史表示。"""
        with tempfile.TemporaryDirectory() as tmp:
            messages: list[dict[str, Any]] = [{"role": "user", "content": "root"}]
            for i in range(6):
                messages.append({"role": "assistant", "content": f"old msg {i}"})

            compactor = LayeredCompactor(
                transcript_dir=Path(tmp), max_recent_messages=2
            )
            compacted = compactor(messages)
            summary = _compacted_summary(compacted)

            assert summary is not None
            assert summary.startswith("[Compressed]")

    def test_compacted_output_is_deterministic(self) -> None:
        """相同输入 + 相同压缩器 = 完全相同的输出。"""
        with tempfile.TemporaryDirectory() as tmp:
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "root"},
                {"role": "assistant", "content": "msg 1"},
                {"role": "user", "content": "msg 2"},
                {"role": "assistant", "content": "msg 3"},
                {"role": "user", "content": "msg 4"},
            ]
            compactor = LayeredCompactor(
                transcript_dir=Path(tmp), max_recent_messages=2
            )
            result1 = compactor(messages)
            result2 = compactor(messages)
            assert [m.get("content") for m in result1] == [
                m.get("content") for m in result2
            ]


class TestTaskStateCollectorNoDuplicate:
    """验证 TaskStateCollector 不注入与 [Compressed] 重复的内容。"""

    def test_task_state_does_not_contain_compaction_summary(self) -> None:
        """TaskStateCollector 的任务状态上下文块不应包含 [Compressed] 内容。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = None
            from xcode.harness.task_store import TaskStore

            store = TaskStore(root)
            store.create("active task")

            from xcode.agent.context_collector import (
                ContextCollectionInput,
                TaskStateCollector,
            )

            provider = None
            from xcode.harness.agent_runtime.config import _build_task_state_provider

            provider = _build_task_state_provider(root)
            assert provider is not None

            collector = TaskStateCollector(provider)
            blocks = collector.collect(ContextCollectionInput())
            assert len(blocks) == 1
            block = blocks[0]
            assert block.source.value == "task_state"
            # 任务状态块必不包含 [Compressed] 或 compaction-state
            assert "[Compressed]" not in block.content
            assert "compaction-state" not in block.content
            # 应包含任务标题
            assert "active task" in block.content
            assert "#1" in block.content

    def test_task_state_only_injects_task_metadata(self) -> None:
        """验证 TaskStateCollector 仅注入结构化任务元数据，而非对话摘要。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from xcode.harness.task_store import TaskStore

            store = TaskStore(root)
            store.create("my task", payload={"blocked_by": [5]})

            from xcode.harness.agent_runtime.config import _build_task_state_provider

            provider = _build_task_state_provider(root)
            assert provider is not None
            state_text = provider()

            assert "my task" in state_text
            assert "Blocked by: [5]" in state_text
            assert "[Compressed]" not in state_text
            assert "compaction-state" not in state_text

    def test_compacted_message_does_not_contain_task_state(self) -> None:
        """[Compressed] 压实历史不应包含任务状态元数据。"""
        with tempfile.TemporaryDirectory() as tmp:
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "root"},
            ]
            for i in range(5):
                messages.append({"role": "assistant", "content": f"msg {i}"})

            compactor = LayeredCompactor(
                transcript_dir=Path(tmp), max_recent_messages=2
            )
            compacted = compactor(messages)
            summary = _compacted_summary(compacted)
            assert summary is not None
            assert "pending" not in (summary or "").lower()
            assert "#1" not in summary or ""


class TestCompactionDoesNotDuplicateCollectors:
    """验证压实不复制其他 collector 处理的内容。"""

    def test_no_project_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "root"},
                {"role": "assistant", "content": "some work"},
                {"role": "user", "content": "more work"},
            ]
            compactor = LayeredCompactor(
                transcript_dir=Path(tmp), max_recent_messages=1
            )
            compacted = compactor(messages)
            summary = _compacted_summary(compacted)
            assert summary is not None
            assert "AGENTS.md" not in summary or ""
            assert "agent.md" not in (summary or "").lower()

    def test_no_active_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "root"},
                {"role": "assistant", "content": "some work"},
                {"role": "user", "content": "more work"},
            ]
            compactor = LayeredCompactor(
                transcript_dir=Path(tmp), max_recent_messages=1
            )
            compacted = compactor(messages)
            summary = _compacted_summary(compacted)
            assert summary is not None
            assert "diff --git" not in summary or ""

    def test_no_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "root"},
                {"role": "assistant", "content": "some work"},
                {"role": "user", "content": "more work"},
            ]
            compactor = LayeredCompactor(
                transcript_dir=Path(tmp), max_recent_messages=1
            )
            compacted = compactor(messages)
            summary = _compacted_summary(compacted)
            assert summary is not None
            assert "skills" not in (summary or "").lower()

    def test_no_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "root"},
                {"role": "assistant", "content": "some work"},
                {"role": "user", "content": "more work"},
            ]
            compactor = LayeredCompactor(
                transcript_dir=Path(tmp), max_recent_messages=1
            )
            compacted = compactor(messages)
            summary = _compacted_summary(compacted)
            assert summary is not None


if __name__ == "__main__":
    pytest.main()
