from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from ...agent.types import ToolDefinition
from .events import Message, ProviderEvent


class ModelProvider(Protocol):
    """Minimal protocol boundary for an LLM provider."""

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> AsyncIterator[ProviderEvent]: ...
