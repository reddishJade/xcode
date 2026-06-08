"""Agent 消息内容块类型定义。

这些类型表示 agent 消息中的各种 content block，用于构建和解析 LLM 消息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextContent:
    """纯文本内容块。"""

    type: str = "text"
    text: str = ""


@dataclass(frozen=True)
class ImageContent:
    """图像内容块。"""

    type: str = "image"
    source: dict[str, Any] | None = None


@dataclass(frozen=True)
class FileContent:
    """文件内容块。"""

    type: str = "file"
    source: dict[str, Any] | None = None
    file_id: str | None = None
    filename: str | None = None
    file_data: str | None = None


@dataclass(frozen=True)
class ToolCallContent:
    """工具调用内容块。"""

    type: str = "tool_call"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class ThinkingContent:
    """思考内容块。"""

    type: str = "thinking"
    thinking: str = ""
    signature: str | None = None


@dataclass(frozen=True)
class ToolResultContent:
    """工具执行结果内容块。"""

    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    status: str = "ok"
