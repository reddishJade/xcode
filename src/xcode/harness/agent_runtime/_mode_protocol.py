"""ToolGateMode — ToolGate 对执行模式的最小协议依赖。

ExecutionModeState（位于 coding_agent/execution_modes.py）实现此协议，
从而使 ToolGate 不直接依赖 coding-agent 层的 execition modes 逻辑。
"""

from __future__ import annotations

from typing import Protocol

from xcode.ai.events import ToolCall
from ..security.permissions import PermissionDecision


class ToolGateMode(Protocol):
    """ToolGate 对执行模式的唯二需求：当前模式名称 + 准入检查。"""

    @property
    def current_mode(self) -> str: ...

    def check_call(self, call: ToolCall) -> PermissionDecision: ...
