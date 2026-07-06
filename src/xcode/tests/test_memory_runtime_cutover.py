"""End-to-end contract for the typed memory read-path cutover."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xcode.ai.events import FinalMessage, TextDelta
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import AgentRuntimeConfig
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.harness.memory import MemoryManager
from xcode.tests.fixtures import FakeProvider


def _memory_block() -> str:
    return (
        "## Provider retry boundary\n"
        "- Context/Query: Provider timeout retry handling\n"
        "- Solution: Retry transient provider failures with bounded backoff.\n"
        "- Files: src/xcode/provider.py\n"
        "- Takeaways: Preserve the root cause in final diagnostics.\n"
    )


def test_runtime_injects_memory_only_as_typed_user_context(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path)
    assert manager.add_memory_block(_memory_block(), source="fixture", layer="project")
    captured: list[list[dict[str, Any]]] = []

    def factory(
        messages: list[dict[str, Any]],
        _tools: list[Any],
    ) -> list[object]:
        captured.append(messages)
        return [TextDelta(chunk="done"), FinalMessage(content="", stop_reason="end_turn")]

    agent = StructuredAgent(
        provider=FakeProvider(factory),
        registry=(),
        runtime=AgentRuntimeConfig(
            project_root=tmp_path,
            runtime_context_provider=build_runtime_context_provider(
                tmp_path,
                (),
                memory_manager=manager,
            ),
            memory_manager=manager,
        ),
    )

    result = agent.run("How should provider timeout retries work?")

    assert result.answer == "done"
    assert len(captured) == 1
    messages = captured[0]
    system_messages = [
        str(message["content"])
        for message in messages
        if message.get("role") == "system"
    ]
    user_messages = [
        str(message["content"])
        for message in messages
        if message.get("role") == "user"
    ]
    assert "<memory>" not in "\n".join(system_messages)
    assert "<memory-overview>" not in "\n".join(system_messages)
    assert not any("Provider retry boundary" in item for item in system_messages)
    memory_context = next(item for item in user_messages if "<memory-digest>" in item)
    assert "authority=memory" in memory_context
    assert "scope=repository" in memory_context
    assert "Provider retry boundary" in memory_context
