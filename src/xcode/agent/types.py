"""Agent 消息内容块类型定义。

这些类型表示 agent 消息中的各种 content block，用于构建和解析 LLM 消息。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from xcode.ai.types import ToolArguments

type ContentSource = dict[str, object]


class TextContent(BaseModel):
    """纯文本内容块。"""

    type: str = "text"
    text: str = ""
    model_config = ConfigDict(frozen=True, extra="forbid")


class ImageContent(BaseModel):
    """图像内容块。"""

    type: str = "image"
    source: ContentSource | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    def __repr__(self) -> str:
        """返回不包含内联图片数据的诊断表示。"""
        source = self.source or {}
        source_type = source.get("type", "unknown")
        media_type = source.get("media_type", "unknown")
        return (
            f"ImageContent(type={self.type!r}, source_type={source_type!r}, "
            f"media_type={media_type!r})"
        )


class FileContent(BaseModel):
    """文件内容块。"""

    type: str = "file"
    source: ContentSource | None = None
    file_id: str | None = None
    filename: str | None = None
    file_data: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")

    def __repr__(self) -> str:
        """返回不包含内联文件数据的诊断表示。"""
        identity = self.filename or self.file_id or "unnamed"
        return f"FileContent(type={self.type!r}, identity={identity!r})"


class ToolCallContent(BaseModel):
    """工具调用内容块。"""

    type: str = "tool_call"
    id: str = ""
    name: str = ""
    arguments: ToolArguments | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class ThinkingContent(BaseModel):
    """思考内容块。"""

    type: str = "thinking"
    thinking: str = ""
    signature: str | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolResultContent(BaseModel):
    """工具执行结果内容块。"""

    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    status: str = "ok"
    model_config = ConfigDict(frozen=True, extra="forbid")


class ShellCallOutputContent(BaseModel):
    """Shell 调用输出内容块。"""

    type: str = "shell_call_output"
    call_id: str = ""
    output: list[dict[str, object]] = Field(default_factory=list)
    max_output_length: int | None = None
    model_config = ConfigDict(frozen=True, extra="forbid")
