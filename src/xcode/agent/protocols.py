from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

from xcode.agent.types import (
    FileContent,
    ImageContent,
    ShellCallOutputContent,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)

"""Agent core protocols and base types."""

type QueueMode = Literal["all", "one-at-a-time"]
type ToolExecutionMode = Literal["sequential", "parallel"]

type ContentBlock = (
    TextContent | ImageContent | FileContent | ToolCallContent | ThinkingContent
)
type ToolResultContentBlock = (
    TextContent | ImageContent | FileContent | ShellCallOutputContent
)


class AgentToolResult:
    content: list[ToolResultContentBlock]
    details: Any | None = None
    terminate: bool = False

    def __init__(
        self,
        content: list[ToolResultContentBlock] | None = None,
        details: Any | None = None,
        terminate: bool = False,
    ) -> None:
        self.content = content or []
        self.details = details
        self.terminate = terminate


type ToolUpdateCallback = Callable[[AgentToolResult], None]


class CancellationSignal(Protocol):
    @property
    def reason(self) -> str: ...

    def is_cancelled(self) -> bool: ...


class AgentTool(Protocol):
    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    execution_mode: ToolExecutionMode | None = None
    examples: list[dict[str, Any]] = []

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: CancellationSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult: ...
