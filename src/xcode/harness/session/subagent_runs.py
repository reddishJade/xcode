"""子代理运行生命周期的稳定事件模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


type SubagentRunStatus = Literal["started", "completed", "failed", "cancelled"]


class SubagentRunEvent(BaseModel):
    """一条子代理运行状态变化。"""

    run_id: str
    batch_id: str
    task_index: int
    description: str
    subagent_type: str
    status: SubagentRunStatus
    summary: str = ""
    error: str = ""
    model_config = ConfigDict(frozen=True, extra="forbid")
