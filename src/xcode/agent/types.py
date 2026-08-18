"""Agent 层类型定义：内容块、协议、工具描述、回调签名。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from xcode.ai.types import ToolArguments

type ContentSource = dict[str, object]


class TextContent(BaseModel):
    type: str = "text"
    text: str = ""
    model_config = ConfigDict(frozen=True, extra="forbid")


class ImageContent(BaseModel):
    type: str = "image"
    source: ContentSource | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    def __repr__(self) -> str:
        source = self.source or {}
        source_type = source.get("type", "unknown")
        media_type = source.get("media_type", "unknown")
        return (
            f"ImageContent(type={self.type!r}, source_type={source_type!r}, "
            f"media_type={media_type!r})"
        )


class FileContent(BaseModel):
    type: str = "file"
    source: ContentSource | None = None
    file_id: str | None = None
    filename: str | None = None
    file_data: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    def __repr__(self) -> str:
        identity = self.filename or self.file_id or "unnamed"
        return f"FileContent(type={self.type!r}, identity={identity!r})"


class ToolCallContent(BaseModel):
    type: str = "tool_call"
    id: str = ""
    name: str = ""
    arguments: ToolArguments | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class ThinkingContent(BaseModel):
    type: str = "thinking"
    thinking: str = ""
    signature: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolResultContent(BaseModel):
    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    status: str = "ok"
    model_config = ConfigDict(frozen=True, extra="forbid")


class ShellCallOutputContent(BaseModel):
    type: str = "shell_call_output"
    call_id: str = ""
    output: list[dict[str, object]] = Field(default_factory=list)
    max_output_length: int | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


type QueueMode = Literal["all", "one-at-a-time"]
type ToolExecutionMode = Literal["sequential", "parallel"]
type ToolResultDetails = object


class TerminalRenderIntent(BaseModel):
    """将工具结果呈现为一次本地终端执行。"""

    kind: Literal["terminal"] = "terminal"
    command: str
    cwd: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class DiffRenderIntent(BaseModel):
    """将工具结果呈现为结构化文件差异。"""

    kind: Literal["diff"] = "diff"
    patch: str
    files: tuple[str, ...] = ()
    first_changed_line: int | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class LocationRenderIntent(BaseModel):
    """将工具结果关联到文件或目录位置。"""

    kind: Literal["location"] = "location"
    path: str
    line_start: int | None = None
    line_end: int | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class SubagentRenderIntent(BaseModel):
    """将工具结果关联到一批可追踪的子代理运行。"""

    kind: Literal["subagent"] = "subagent"
    batch_id: str
    run_ids: tuple[str, ...]
    model_config = ConfigDict(frozen=True, extra="forbid")


type ToolRenderIntent = Annotated[
    TerminalRenderIntent
    | DiffRenderIntent
    | LocationRenderIntent
    | SubagentRenderIntent,
    Field(discriminator="kind"),
]

_TOOL_RENDER_INTENT_ADAPTER = TypeAdapter(ToolRenderIntent)


def parse_tool_render_intent(value: object) -> ToolRenderIntent | None:
    """从持久化事件解码严格的工具呈现意图。"""
    if value is None:
        return None
    try:
        return _TOOL_RENDER_INTENT_ADAPTER.validate_python(value)
    except ValidationError:
        return None


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
    render_intent: ToolRenderIntent | None = None

    def __init__(
        self,
        content: list[ToolResultContentBlock] | None = None,
        details: ToolResultDetails | None = None,
        is_error: bool = False,
        terminate: bool = False,
        render_intent: ToolRenderIntent | None = None,
    ) -> None:
        self.content = content or []
        self.details = details
        self.is_error = is_error
        self.terminate = terminate
        self.render_intent = render_intent


type ToolUpdateCallback = Callable[[AgentToolResult], None]


ToolInput = dict[str, Any]
ActionHandler = Callable[[ToolInput, Callable[[str], None] | None], str]
HITLResult = Any
ApprovalScope = Literal["once", "session", "permanent"]


@dataclass(frozen=True)
class ApprovalRequest:
    """权限引擎传给交互层的审批请求。"""

    tool: "ToolSpec"
    action_input: ToolInput
    allowed_scopes: tuple[ApprovalScope, ...]
    reason: str


ApprovalCallback = Callable[[ApprovalRequest], HITLResult]


@dataclass(frozen=True)
class CitationSource:
    """模型可引用的本地证据来源。"""

    kind: Literal["file", "search"]
    path: str
    start_line: int
    end_line: int
    text: str


class ToolOutput(str):
    """带结构化元数据的工具输出文本。"""

    metadata: dict[str, object]
    is_error: bool
    render_intent: ToolRenderIntent | None

    def __new__(
        cls,
        content: str,
        metadata: Mapping[str, object] | None = None,
        is_error: bool = False,
        render_intent: ToolRenderIntent | None = None,
    ) -> "ToolOutput":
        output = str.__new__(cls, content)
        output.metadata = dict(metadata) if metadata else {}
        output.is_error = is_error
        output.render_intent = render_intent
        return output


@dataclass(frozen=True)
class ToolSpec:
    """工具描述。"""

    name: str
    description: str
    input_hint: str
    handler: ActionHandler
    schema: Mapping[str, Any] | None = None
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()


AGENT_CONTENT_BLOCKS_METADATA_KEY = "agent_content_blocks"
CITATION_SOURCES_METADATA_KEY = "citation_sources"


def stringify_tool_input(action_input: ToolInput) -> str:
    return json.dumps(action_input, ensure_ascii=False, sort_keys=True)


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


class ToolSpecAdapter:
    """ToolSpec → AgentTool 适配器（无 redaction，可用于子代理等场景）。"""

    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def label(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def parameters(self) -> Mapping[str, object]:
        return self._spec.schema or {}

    @property
    def execution_mode(self) -> None:
        return None

    @property
    def examples(self) -> list[dict[str, object]]:
        return []

    async def execute(
        self,
        tool_call_id: str,
        params: ToolArguments,
        signal: CancellationSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        def _text_update(text: str) -> None:
            if on_update is not None:
                on_update(AgentToolResult(content=[TextContent(text=text)]))

        content = await asyncio.to_thread(
            self._spec.handler, dict(params), _text_update
        )
        metadata = getattr(content, "metadata", None)
        render_intent = getattr(content, "render_intent", None)
        return AgentToolResult(
            content=[TextContent(text=str(content))],
            details=metadata if isinstance(metadata, dict) else None,
            is_error=bool(getattr(content, "is_error", False)),
            render_intent=render_intent,
        )
