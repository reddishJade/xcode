"""Hook 类型别名。

从 config.py 提取的可独立存在的 callable type aliases。
引用 config.py 上下文类型的别名（BeforeToolCallHook 等）保留在 config.py 中。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xcode.ai.types import ToolDefinition

from .messages import AgentMessage, ToolResultMessage
from .protocols import CancellationSignal
from .types import ToolCallContent


type MessageConverter = Callable[[list[AgentMessage]], list[dict[str, Any]]]
type ContextTransformer = Callable[
    [list[AgentMessage], CancellationSignal | None], list[AgentMessage]
]
type MessageQueueGetter = Callable[[], list[AgentMessage]]
type ArchiveWriter = Callable[[list[AgentMessage]], str | None]
type ShouldCompactHook = Callable[[list[AgentMessage]], bool]
type CompactHook = Callable[[list[AgentMessage]], list[AgentMessage]]
type IsToolProductiveHook = Callable[
    [list[ToolCallContent], list[ToolResultMessage]], bool
]
type BeforeProviderRequestHook = Callable[
    [list[dict[str, Any]], list[ToolDefinition]], None
]
