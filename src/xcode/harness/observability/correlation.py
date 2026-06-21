"""单次 agent 运行的观测关联状态。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict


@dataclass(frozen=True)
class EventCorrelation:
    """可序列化的事件关联字段。"""

    timestamp: str = ""
    session_id: str = ""
    turn_id: str = ""
    request_id: str = ""
    tool_call_id: str = ""


class HookCorrelationFields(TypedDict):
    """HookRecord 构造使用的关联字段。"""

    timestamp: str
    session_id: str
    turn_id: str
    request_id: str
    tool_call_id: str


def hook_correlation_fields(
    correlation: EventCorrelation,
) -> HookCorrelationFields:
    """把关联快照映射为 HookRecord 关键字参数。"""
    return {
        "timestamp": correlation.timestamp,
        "session_id": correlation.session_id,
        "turn_id": correlation.turn_id,
        "request_id": correlation.request_id,
        "tool_call_id": correlation.tool_call_id,
    }


class RuntimeCorrelation:
    """在 hook 和结构化事件之间共享关联标识。"""

    def __init__(self, session_id: str) -> None:
        """初始化指定 session 的空运行状态。"""
        self.session_id = session_id
        self._turn_index = 0
        self._request_index = 0
        self.turn_id = ""
        self.request_id = ""

    def reset(self, session_id: str | None = None) -> None:
        """开始新的 agent 运行。"""
        if session_id is not None:
            self.session_id = session_id
        self._turn_index = 0
        self._request_index = 0
        self.turn_id = ""
        self.request_id = ""

    def begin_turn(self) -> EventCorrelation:
        """进入下一模型 turn。"""
        self._turn_index += 1
        self.turn_id = f"{self.session_id}:turn:{self._turn_index}"
        self.request_id = ""
        return self.snapshot()

    def begin_request(self) -> EventCorrelation:
        """记录当前 turn 的下一次 provider 请求。"""
        self._request_index += 1
        self.request_id = f"{self.session_id}:request:{self._request_index}"
        return self.snapshot()

    def snapshot(self, tool_call_id: str = "") -> EventCorrelation:
        """返回带 UTC 时间戳的当前关联快照。"""
        return EventCorrelation(
            timestamp=datetime.now(UTC).isoformat(),
            session_id=self.session_id,
            turn_id=self.turn_id,
            request_id=self.request_id,
            tool_call_id=tool_call_id,
        )
