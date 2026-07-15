"""通过正式 build_app 路径运行真实 Xcode Trial。"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
from time import monotonic
from typing import Any

from pydantic import BaseModel

from xcode.harness.app import build_app
from xcode.harness.agent_runtime.events import FinalStructuredEvent
from xcode.harness.agent_runtime.result import CodingAgentHarnessResult
from xcode.harness.config import XcodeRuntimeConfig

from .policy import approve_eval_action
from .schema import ResourceUsage, Task, Trial


class ExecutorError(RuntimeError):
    """真实 Xcode 未能形成可记录的 Agent 结果。"""


@dataclass(frozen=True)
class AgentExecution:
    """Agent 阶段的结果与原始资源消耗。"""

    started_at: datetime
    finished_at: datetime
    result: CodingAgentHarnessResult
    usage: ResourceUsage
    trace_lines: tuple[str, ...]


class RealXcodeExecutor:
    """装配真实 provider 和工具，并逐事件保存 JSONL trace。"""

    async def run(
        self,
        *,
        task: Task,
        trial: Trial,
        workspace: Path,
        runtime_config: XcodeRuntimeConfig,
        trace_path: Path | None,
        env_files: tuple[Path, ...] = (),
    ) -> AgentExecution:
        """执行单次真实 Trial，墙钟预算由 Eval 控制面强制。"""
        if (
            trial.task_id != task.task_id
            or trial.dataset_version != task.dataset_version
        ):
            raise ExecutorError("trial and task identity do not match")
        runtime = runtime_config.model_copy(
            update={
                "agent": runtime_config.agent.model_copy(
                    update={"max_steps": trial.budget.model_calls}
                )
            }
        )
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text("", encoding="utf-8")
        started_at = datetime.now(UTC)
        started_clock = monotonic()
        final: CodingAgentHarnessResult | None = None
        trace_lines: list[str] = []
        app = build_app(
            project_root=workspace,
            env_files=env_files,
            runtime_config=runtime,
        )
        app.agent.approval_callback = approve_eval_action
        try:
            async with asyncio.timeout(trial.budget.wall_time_seconds):
                async for event in app.aask_stream(task.prompt, mode="build"):
                    line = (
                        json.dumps(
                            _jsonable(event),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    trace_lines.append(line)
                    if trace_path is not None:
                        with trace_path.open("a", encoding="utf-8") as trace:
                            trace.write(line)
                    if isinstance(event, FinalStructuredEvent):
                        final = event.data
        except TimeoutError as error:
            raise ExecutorError("Agent exceeded wall-time budget") from error
        finally:
            app.close()
        if final is None:
            raise ExecutorError("Xcode stream ended without a final event")
        metrics = final.metrics or {}
        usage = ResourceUsage(
            wall_time_seconds=monotonic() - started_clock,
            model_calls=int(metrics.get("llm_calls", 0)),
            tool_calls=int(metrics.get("tool_calls", len(final.tool_calls))),
            input_tokens=_optional_int(metrics.get("estimated_prompt_tokens")),
            output_tokens=_optional_int(metrics.get("estimated_completion_tokens")),
        )
        return AgentExecution(
            started_at=started_at,
            finished_at=datetime.now(UTC),
            result=final,
            usage=usage,
            trace_lines=tuple(trace_lines),
        )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _jsonable(value: object) -> Any:
    """将 runtime dataclass、枚举和嵌套对象转为稳定 JSON 值。"""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump())
    return str(value)
