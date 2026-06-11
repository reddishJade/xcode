"""Agent 核心协议与基础类型。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, Protocol

from xcode.agent.types import (
    FileContent,
    ImageContent,
    ShellCallOutputContent,
    TextContent,
    ThinkingContent,
    ToolArguments,
    ToolCallContent,
    ToolResultContent,
)


type QueueMode = Literal["all", "one-at-a-time"]
type ToolExecutionMode = Literal["sequential", "parallel"]
type ToolResultDetails = object

type ContentBlock = (
    TextContent | ImageContent | FileContent | ToolCallContent | ThinkingContent
)
type ToolResultContentBlock = (
    TextContent
    | ImageContent
    | FileContent
    | ToolResultContent
    | ShellCallOutputContent
)


class AgentToolResult:
    content: list[ToolResultContentBlock]
    details: ToolResultDetails | None = None
    is_error: bool = False
    terminate: bool = False

    def __init__(
        self,
        content: list[ToolResultContentBlock] | None = None,
        details: ToolResultDetails | None = None,
        is_error: bool = False,
        terminate: bool = False,
    ) -> None:
        self.content = content or []
        self.details = details
        self.is_error = is_error
        self.terminate = terminate


type ToolUpdateCallback = Callable[[AgentToolResult], None]


class CancellationSignal(Protocol):
    @property
    def reason(self) -> str: ...

    def is_cancelled(self) -> bool: ...


class AgentTool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, object]: ...

    @property
    def execution_mode(self) -> ToolExecutionMode | None: ...

    @property
    def examples(self) -> list[dict[str, object]]: ...

    async def execute(
        self,
        tool_call_id: str,
        params: ToolArguments,
        signal: CancellationSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult: ...
