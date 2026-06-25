"""REPL memory 命令测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from xcode.cli.repl_commands import (
    COMMAND_REGISTRY,
    _add_memory,
    _list_memory,
    _search_memory,
)
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.harness.assembly import _extend_registry_with_features
from xcode.harness.mcp import McpRuntimeRegistry
from xcode.harness.memory import MemoryManager, build_memory_tools


def test_memory_command_helpers_add_list_and_search(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """显式命令可以写入用户层并跨层级检索。"""
    manager = MemoryManager(
        tmp_path,
        user_memory_file=tmp_path / "home" / ".xcode" / "memory" / "MEMORY.md",
    )

    assert not _add_memory(
        manager,
        (
            "user Shared retry rule | Provider timeout across projects | "
            "Use bounded exponential backoff | provider clients | "
            "Retry transient failures only"
        ),
    )
    add_output = capsys.readouterr().out
    assert "Added user memory: Shared retry rule" in add_output

    assert not _list_memory(manager, "user")
    list_output = capsys.readouterr().out
    assert "[user] Shared retry rule" in list_output

    assert not _search_memory(manager, "provider timeout")
    search_output = capsys.readouterr().out
    assert "[user]" in search_output
    assert "Shared retry rule" in search_output


def test_memory_command_is_registered() -> None:
    """REPL 注册表公开 `/memory` 入口。"""
    entry = COMMAND_REGISTRY["/memory"]

    assert entry.accepts_args
    assert "search" in entry.args_desc


def test_runtime_context_provider_injects_relevant_memory(tmp_path: Path) -> None:
    """运行时按当前问题主动召回并隔离注入记忆。"""
    manager = MemoryManager(
        tmp_path,
        user_memory_file=tmp_path / "user-memory" / "MEMORY.md",
    )
    assert manager.add_memory_block(
        (
            "## Provider timeout retry\n"
            "- Context/Query: Provider connection timeout\n"
            "- Solution: Retry transient provider failures\n"
            "- Files: src/provider.py\n"
            "- Takeaways: Bound retries and preserve the root cause\n"
        )
    )
    provider = build_runtime_context_provider(
        tmp_path,
        (),
        memory_manager=manager,
    )

    contexts = provider("How should provider timeout retries work?")

    assert any("<memory>" in context for context in contexts)
    assert any('type="episodic"' in context for context in contexts)
    assert any("<conclusion>" in context for context in contexts)
    assert any("Retry transient provider failures" in context for context in contexts)
    assert any('layer="project"' in context for context in contexts)
    assert all("- Context/Query:" not in context for context in contexts if "<memory>" in context)
    record = manager.read_memory_records()[0]
    assert manager.get_last_used_at(record) is not None
    trace_events = manager.drain_trace_events()
    assert any(event.type == "retrieved" for event in trace_events)
    assert any(event.type == "used" and event.source == "prompt" for event in trace_events)
    injected = [event for event in trace_events if event.type == "injected"]
    assert injected
    assert all(event.token_count and event.token_count > 0 for event in injected)
    assert all("Provider connection timeout" not in repr(event) for event in trace_events)


def test_memory_group_registers_search_tool(tmp_path: Path) -> None:
    """memory 组只注册显式只读检索工具。"""
    from xcode.harness.assembly import build_shared_services
    from xcode.harness.config import XcodeRuntimeConfig

    runtime_registry = McpRuntimeRegistry()
    try:
        tools = _extend_registry_with_features(
            (),
            tmp_path,
            runtime_registry,
            XcodeRuntimeConfig(),
            build_shared_services(tmp_path, XcodeRuntimeConfig()),
        )
    finally:
        runtime_registry.close()

    memory_tool = next(tool for tool in tools if tool.name == "search_memory")
    assert memory_tool.group == "memory"
    assert memory_tool.read_only


def test_memory_search_merges_project_and_user_layers(tmp_path: Path) -> None:
    """BM25 检索合并两个层级并保留来源。"""
    user_memory_file = tmp_path / "home" / "memory" / "MEMORY.md"
    manager = MemoryManager(tmp_path, user_memory_file=user_memory_file)
    manager.memory_file.write_text(
        (
            "## Project provider retry\n"
            "- Context/Query: Provider timeout in this repository\n"
            "- Solution: Retry the project provider request\n"
            "- Files: src/provider.py\n"
            "- Takeaways: Keep project retry policy local\n"
        ),
        encoding="utf-8",
    )
    user_memory_file.parent.mkdir(parents=True)
    user_memory_file.write_text(
        (
            "## User provider preference\n"
            "- Context/Query: Provider timeout across repositories\n"
            "- Solution: Prefer bounded exponential backoff\n"
            "- Files: shared provider clients\n"
            "- Takeaways: Reuse the user retry preference\n"
        ),
        encoding="utf-8",
    )

    records = manager.search_memory_records("provider timeout", limit=5)

    assert {record.layer for record in records} == {"project", "user"}
    assert {record.title for record in records} == {
        "Project provider retry",
        "User provider preference",
    }


def test_memory_tool_searches_both_layers(tmp_path: Path) -> None:
    """agent 工具返回匹配块及其层级。"""
    manager = MemoryManager(
        tmp_path,
        user_memory_file=tmp_path / "user" / "MEMORY.md",
    )
    block = (
        "## Shared timeout rule\n"
        "- Context/Query: Network timeout across projects\n"
        "- Solution: Use bounded retries\n"
        "- Files: provider clients\n"
        "- Takeaways: Retry only transient failures\n"
    )
    assert manager.add_memory_block(block, layer="user")
    tool = build_memory_tools(manager)[0]

    output = tool.handler({"query": "network timeout"})

    assert tool.group == "memory"
    assert tool.read_only
    assert "[user] id=mem_" in output
    assert "type=" in output
    assert "Shared timeout rule" in output
    assert "- Context/Query:" in output
    record = manager.read_memory_records(layer="user")[0]
    assert manager.get_last_used_at(record) is not None
    trace_events = manager.drain_trace_events()
    assert any(event.type == "retrieved" for event in trace_events)
    assert any(event.type == "tool_searched" for event in trace_events)
    assert any(event.type == "used" and event.source == "tool" for event in trace_events)


def test_memory_context_skips_weak_matches_when_below_threshold(tmp_path: Path) -> None:
    manager = MemoryManager(
        tmp_path,
        user_memory_file=tmp_path / "user-memory" / "MEMORY.md",
        min_retrieval_score=0.8,
    )
    assert manager.add_memory_block(
        (
            "## Provider timeout retry\n"
            "- Context/Query: Provider connection timeout\n"
            "- Solution: Retry transient provider failures\n"
            "- Files: src/provider.py\n"
            "- Takeaways: Bound retries and preserve the root cause\n"
        )
    )
    provider = build_runtime_context_provider(
        tmp_path,
        (),
        memory_manager=manager,
    )

    contexts = provider("unrelated grocery list question")

    assert all("<memory>" not in context for context in contexts)
