"""子代理运行生命周期的稳定事件模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


type SubagentRunStatus = Literal["started", "completed", "failed", "cancelled"]
type SubagentMode = Literal["one_shot", "continuable"]


class SubagentDescriptor(BaseModel):
    """写入 child session 的稳定身份与组合来源。"""

    child_session_id: str
    parent_session_id: str
    mode: SubagentMode
    description: str
    subagent_type: str
    provider_model: str
    composition_id: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class SubagentRunEvent(BaseModel):
    """一条子代理运行状态变化。"""

    run_id: str
    child_session_id: str
    batch_id: str
    task_index: int
    description: str
    subagent_type: str
    mode: SubagentMode
    status: SubagentRunStatus
    summary: str = ""
    error: str = ""
    model_config = ConfigDict(frozen=True, extra="forbid")
