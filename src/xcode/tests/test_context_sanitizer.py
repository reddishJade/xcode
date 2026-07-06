"""Provider-boundary tests for legacy memory prompt sections."""

from __future__ import annotations

import pytest

from xcode.agent.context_sanitizer import demote_embedded_memory_sections
from xcode.agent.messages import SystemMessage, UserMessage


@pytest.mark.parametrize("tag", ["memory", "memory-overview"])
def test_embedded_memory_section_is_demoted_from_system_message(tag: str) -> None:
    messages = [
        SystemMessage(
            content=(
                "Host rule before.\n\n"
                f"<{tag}>\nPrior learning.\n</{tag}>\n\n"
                "Host rule after."
            )
        ),
        UserMessage(content="Continue."),
    ]

    sanitized = demote_embedded_memory_sections(messages)

    assert [message.role for message in sanitized] == ["system", "user", "user"]
    assert "Host rule before." in sanitized[0].content
    assert "Host rule after." in sanitized[0].content
    assert f"<{tag}>" not in sanitized[0].content
    assert "source=memory" in sanitized[1].content
    assert "authority=memory" in sanitized[1].content
    assert "system_target=demoted" in sanitized[1].content
    assert f"<{tag}>\nPrior learning.\n</{tag}>" in sanitized[1].content


def test_system_message_without_memory_section_is_unchanged() -> None:
    message = SystemMessage(content="Host policy only.")

    sanitized = demote_embedded_memory_sections([message])

    assert sanitized == [message]
