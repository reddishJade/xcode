"""Web 事件序列化。

把 AgentHarnessEvent 及其携带的嵌套值（pydantic / dataclass / enum）递归
转换为 JSON 安全结构，供 WebSocket 通道推送。
"""

from __future__ import annotations

import dataclasses
import enum
from datetime import date, datetime
from pathlib import PurePath
from typing import Any

from xcode.harness.agent_runtime import AgentHarnessEvent

# 长文本（如审批 transcript）只推送头部，避免撑爆浏览器内存
_TRUNCATE_LONG_TEXT = 12_000
_RUN_STATE_MESSAGES_LIMIT = 200


def to_jsonable(value: Any) -> Any:
    """递归转换任意值为 JSON 安全结构。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {to_jsonable(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump(mode="json"))
    if dataclasses.is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.init and not field.name.startswith("_")
        }
    return str(value)


def event_to_dict(event: AgentHarnessEvent) -> dict[str, Any]:
    """把单个 AgentHarnessEvent 转成 WebSocket 推送载荷。"""
    payload = to_jsonable(event)
    assert isinstance(payload, dict)
    return _trim_event(payload)


def _trim_event(payload: dict[str, Any]) -> dict[str, Any]:
    """裁剪超大字段，仅保留前端渲染所需结构。"""
    data = payload.get("data")
    if not isinstance(data, dict):
        return payload
    event_type = payload.get("type")
    if event_type == "approval":
        for key in ("transcript",):
            raw = data.get(key)
            if isinstance(raw, str) and len(raw) > _TRUNCATE_LONG_TEXT:
                data[key] = raw[:_TRUNCATE_LONG_TEXT] + "\n…[truncated]"
    if event_type == "context_window_reset":
        data["replacement"] = []
    if event_type == "final":
        messages = data.get("messages")
        if isinstance(messages, list) and len(messages) > _RUN_STATE_MESSAGES_LIMIT:
            data["messages"] = messages[:_RUN_STATE_MESSAGES_LIMIT]
        run_state = data.get("run_state")
        if isinstance(run_state, dict):
            msgs = run_state.get("messages")
            if isinstance(msgs, list) and len(msgs) > _RUN_STATE_MESSAGES_LIMIT:
                run_state["messages"] = msgs[:_RUN_STATE_MESSAGES_LIMIT]
    return payload
