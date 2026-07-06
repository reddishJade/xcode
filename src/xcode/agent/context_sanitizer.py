"""Provider-boundary sanitizers for legacy prompt sections."""

from __future__ import annotations

import re

from xcode.agent.messages import AgentMessage, SystemMessage, UserMessage

_MEMORY_SECTION_RE = re.compile(
    r"<(?P<tag>memory(?:-overview)?)>.*?</(?P=tag)>",
    flags=re.DOTALL,
)


def demote_embedded_memory_sections(
    messages: list[AgentMessage],
) -> list[AgentMessage]:
    """Move legacy memory XML sections out of SystemMessage instances.

    Existing runtime-context code renders memory as ``<memory>`` or
    ``<memory-overview>`` inside a composite system prompt. Memory is evidence-backed
    background context, not host policy, so provider-boundary sanitation extracts those
    exact sections into user-context messages while retaining all non-memory system text.
    """
    sanitized: list[AgentMessage] = []
    for message in messages:
        if getattr(message, "role", "") != "system" or not isinstance(
            getattr(message, "content", None), str
        ):
            sanitized.append(message)
            continue

        content = message.content
        matches = list(_MEMORY_SECTION_RE.finditer(content))
        if not matches:
            sanitized.append(message)
            continue

        retained_system = _MEMORY_SECTION_RE.sub("", content).strip()
        if retained_system:
            sanitized.append(SystemMessage(content=retained_system))

        for match in matches:
            sanitized.append(
                UserMessage(
                    content=(
                        "[context source=memory authority=memory "
                        "trust=runtime_internal system_target=demoted]\n"
                        f"{match.group(0)}"
                    )
                )
            )
    return sanitized
