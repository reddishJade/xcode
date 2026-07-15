"""bubblewrap 内运行的真实 Xcode worker；不挂载 verifier 控制面。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

from xcode.harness.config import XcodeRuntimeConfig

from .executor import RealXcodeExecutor, _jsonable
from .schema import Task, Trial


async def _main() -> int:
    request = json.loads(sys.stdin.read())
    task = Task.model_validate(request["task"])
    trial = Trial.model_validate(request["trial"])
    runtime = XcodeRuntimeConfig.model_validate(request["runtime_config"])
    execution = await RealXcodeExecutor().run(
        task=task,
        trial=trial,
        workspace=Path("/workspace"),
        runtime_config=runtime,
        trace_path=None,
    )
    summary = {
        "started_at": execution.started_at.isoformat(),
        "finished_at": execution.finished_at.isoformat(),
        "usage": execution.usage.model_dump(mode="json"),
        "termination_reason": execution.result.termination_reason.value,
        "answer": execution.result.answer,
        "error_detail": execution.result.error_detail,
        "watchdog_reason": execution.result.watchdog_reason,
        "result": _jsonable(execution.result),
    }
    # Agent 已完全结束后才写专用输出挂载，覆盖其可能预写的同名文件。
    trace_path = Path("/output/trace.jsonl")
    summary_path = Path("/output/execution.json")
    trace_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)
    trace_path.write_text(
        "".join(execution.trace_lines),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
