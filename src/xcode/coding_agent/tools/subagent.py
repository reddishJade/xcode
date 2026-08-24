from __future__ import annotations

import asyncio
from asyncio import run as async_run
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from xcode.agent.types import SubagentRenderIntent, ToolOutput, ToolSpec
from xcode.harness.agent_runtime.cancellation import CancellationToken
from xcode.harness.agent_runtime.composition import AgentComposition
from xcode.harness.agent_runtime.subagents import (
    SubagentSessionManager,
    SubagentTaskResult,
)
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.harness.session.subagent_runs import SubagentMode

BUILD_SUBAGENT_PROMPTS: dict[str, str] = {
    "coding": (
        "You are an expert software engineer with local file system access. "
        "Complete the assigned task using available tools.\n\n"
        "- Explore before making changes.\n"
        "- Run focused tests, linters, or build commands.\n"
        "- Prefer small, focused changes and verify them.\n"
        "- Return a concise summary of what you did."
    ),
    "research": (
        "You are a thorough research assistant with local file and web access.\n\n"
        "- Inspect relevant local files.\n"
        "- Use web tools for current external information.\n"
        "- Cite sources and files when possible.\n"
        "- Return a structured summary of your findings."
    ),
    "default": (
        "You are a helpful local AI assistant. Complete the assigned task and "
        "return a concise summary."
    ),
}


MAX_BATCH_TASKS = 16
DEFAULT_MAX_CONCURRENT = 4


@dataclass(frozen=True)
class _SubagentRun:
    run_id: str
    batch_id: str
    task_index: int
    task: dict[str, str]


class _SubagentHandler:
    """通过 session manager 创建 one-shot 或 continuable child。"""

    def __init__(self, manager: SubagentSessionManager) -> None:
        self.manager = manager

    def __call__(
        self,
        data: dict[str, Any],
        on_update: Callable[[str], None] | None = None,
    ) -> str:
        tasks_or_error = _parse_tasks(data)
        if isinstance(tasks_or_error, str):
            return tasks_or_error
        tasks = tasks_or_error
        max_concurrent = _max_concurrent(data.get("max_concurrent"))
        batch_id = uuid4().hex
        runs = [
            _SubagentRun(
                run_id=uuid4().hex,
                batch_id=batch_id,
                task_index=index,
                task=task,
            )
            for index, task in enumerate(tasks, start=1)
        ]

        async def execute() -> tuple[str, tuple[str, ...]]:
            if len(runs) == 1:
                result = await _run_one(runs[0], self.manager, on_update)
                return _format_single(result, runs[0].task), (result.run_id,)
            results = await _run_batch(
                runs,
                max_concurrent,
                self.manager,
                on_update,
            )
            return results, tuple(run.run_id for run in runs)

        text, run_ids = async_run(execute())
        return ToolOutput(
            text,
            render_intent=SubagentRenderIntent(
                batch_id=batch_id,
                run_ids=run_ids,
            ),
        )


class _SubagentContinueHandler:
    def __init__(self, manager: SubagentSessionManager) -> None:
        self.manager = manager

    def __call__(
        self,
        data: dict[str, Any],
        on_update: Callable[[str], None] | None = None,
    ) -> str:
        session_id = str(data.get("session_id", "")).strip()
        prompt = str(data.get("prompt", "")).strip()
        if not session_id or not prompt:
            return "Error: session_id and prompt are required"
        result = async_run(
            self.manager.send(
                session_id,
                _bounded_prompt(prompt),
                on_update=on_update,
            )
        )
        return ToolOutput(
            _format_continuation(result),
            render_intent=SubagentRenderIntent(
                batch_id=result.run_id,
                run_ids=(result.run_id,),
            ),
        )


class _SubagentListHandler:
    def __init__(self, manager: SubagentSessionManager) -> None:
        self.manager = manager

    def __call__(
        self,
        _data: dict[str, Any],
        _on_update: Callable[[str], None] | None = None,
    ) -> str:
        children = self.manager.list_children()
        if not children:
            return "No direct child sessions."
        return "\n".join(
            f"- {child.child_session_id}: {child.description} "
            f"({child.mode}, {child.subagent_type})"
            for child in children
        )


class _SubagentControlHandler:
    def __init__(self, manager: SubagentSessionManager) -> None:
        self.manager = manager

    def __call__(
        self,
        data: dict[str, Any],
        _on_update: Callable[[str], None] | None = None,
    ) -> str:
        session_id = str(data.get("session_id", "")).strip()
        action = str(data.get("action", "")).strip()
        if not session_id or action not in {"interrupt", "release"}:
            return "Error: session_id and action=interrupt|release are required"
        if action == "interrupt":
            interrupted = self.manager.interrupt(session_id)
            return (
                "Child turn interrupted." if interrupted else "Child is already idle."
            )
        self.manager.release(session_id)
        return "Child activation released; its durable session remains resumable."


def bind_subagent_runtime(
    registry: tuple[ToolSpec, ...],
    composition_provider: Callable[[], AgentComposition],
    gate: ToolGate,
    cancellation_token: CancellationToken,
) -> None:
    """把父 composition 与权限域绑定到 registry 中的唯一 manager。"""
    managers = {
        cast(Any, spec.handler).manager
        for spec in registry
        if isinstance(
            spec.handler,
            _SubagentHandler
            | _SubagentContinueHandler
            | _SubagentListHandler
            | _SubagentControlHandler,
        )
    }
    for manager in managers:
        manager.bind_parent(composition_provider, gate, cancellation_token)


def build_subagent_tools(
    manager: SubagentSessionManager,
) -> tuple[ToolSpec, ToolSpec, ToolSpec, ToolSpec]:
    """构建创建、续接、枚举与控制 child session 的明确工具。"""
    return (
        _build_subagent_tool(manager),
        _build_subagent_continue_tool(manager),
        _build_subagent_list_tool(manager),
        _build_subagent_control_tool(manager),
    )


def _build_subagent_tool(manager: SubagentSessionManager) -> ToolSpec:
    return ToolSpec(
        name="subagent",
        description=(
            "Create local session-backed child agents. Use one_shot for bounded work; "
            "use continuable when later follow-up turns will be required. Parallel tasks "
            "are always independent one_shot children."
        ),
        input_hint=(
            'JSON: {"description":"label","prompt":"...","mode":"one_shot",'
            '"subagent_type":"coding"}'
        ),
        handler=_SubagentHandler(manager),
        schema={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "prompt": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["one_shot", "continuable"],
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["coding", "research", "default"],
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "prompt": {"type": "string"},
                            "subagent_type": {
                                "type": "string",
                                "enum": ["coding", "research", "default"],
                            },
                        },
                        "required": ["description", "prompt"],
                        "additionalProperties": False,
                    },
                },
                "max_concurrent": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_BATCH_TASKS,
                },
            },
            "oneOf": [
                {"required": ["description", "prompt", "mode"]},
                {"required": ["tasks"]},
            ],
            "additionalProperties": False,
        },
        prompt_snippet="Delegate work to an independent durable child session",
        prompt_guidelines=(
            "Use one_shot for bounded independent work.",
            "Use continuable only when later subagent_continue turns are expected.",
            "Child sessions receive the explicit task, not the parent transcript.",
        ),
    )


def _build_subagent_continue_tool(manager: SubagentSessionManager) -> ToolSpec:
    return ToolSpec(
        name="subagent_continue",
        description="Send the next FIFO turn to a direct continuable child session.",
        input_hint='JSON: {"session_id":"...","prompt":"..."}',
        handler=_SubagentContinueHandler(manager),
        schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "required": ["session_id", "prompt"],
            "additionalProperties": False,
        },
    )


def _build_subagent_list_tool(manager: SubagentSessionManager) -> ToolSpec:
    return ToolSpec(
        name="subagent_list",
        description="List durable direct child sessions without activating them.",
        input_hint="JSON: {}",
        handler=_SubagentListHandler(manager),
        schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def _build_subagent_control_tool(manager: SubagentSessionManager) -> ToolSpec:
    return ToolSpec(
        name="subagent_control",
        description=(
            "Interrupt a direct child turn or release an idle child activation. "
            "Release never deletes the durable child session."
        ),
        input_hint=('JSON: {"session_id":"...","action":"interrupt|release"}'),
        handler=_SubagentControlHandler(manager),
        schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["interrupt", "release"],
                },
            },
            "required": ["session_id", "action"],
            "additionalProperties": False,
        },
    )


def _parse_tasks(data: dict[str, Any]) -> list[dict[str, str]] | str:
    raw_tasks = data.get("tasks")
    if raw_tasks is None:
        prompt = str(data.get("prompt", "")).strip()
        mode = str(data.get("mode", "")).strip()
        if not prompt:
            return "Error: prompt is required"
        if mode not in {"one_shot", "continuable"}:
            return "Error: mode must be one_shot or continuable"
        return [
            {
                "description": str(data.get("description", "subagent")).strip()
                or "subagent",
                "prompt": prompt,
                "subagent_type": str(data.get("subagent_type", "coding")).strip()
                or "coding",
                "mode": mode,
            }
        ]
    if not isinstance(raw_tasks, list):
        return "Error: tasks must be an array"
    if not raw_tasks:
        return "Error: tasks must not be empty"
    if len(raw_tasks) > MAX_BATCH_TASKS:
        return f"Error: tasks may contain at most {MAX_BATCH_TASKS} items"
    tasks: list[dict[str, str]] = []
    for index, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, dict):
            return "Error: each task must be an object"
        prompt = str(raw_task.get("prompt", "")).strip()
        if not prompt:
            return f"Error: task {index} prompt is required"
        tasks.append(
            {
                "description": str(raw_task.get("description", f"task-{index}")).strip()
                or f"task-{index}",
                "prompt": prompt,
                "subagent_type": str(
                    raw_task.get("subagent_type", data.get("subagent_type", "coding"))
                ).strip()
                or "coding",
                "mode": "one_shot",
            }
        )
    return tasks


def _max_concurrent(raw: object) -> int:
    if isinstance(raw, int):
        return max(1, min(MAX_BATCH_TASKS, raw))
    return DEFAULT_MAX_CONCURRENT


async def _run_batch(
    runs: list[_SubagentRun],
    max_concurrent: int,
    manager: SubagentSessionManager,
    on_update: Callable[[str], None] | None,
) -> str:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(run: _SubagentRun) -> tuple[int, str, SubagentTaskResult]:
        async with semaphore:
            label = run.task["description"]
            if on_update is not None:
                on_update(f"[{run.task_index}] → {label}")
            result = await _run_one(
                run,
                manager,
                _task_update(run.task_index, on_update),
            )
            if on_update is not None:
                on_update(f"[{run.task_index}] ✓ {label}")
            return run.task_index, label, result

    results = await asyncio.gather(*(limited(run) for run in runs))
    lines = [f"Subagent batch completed: {len(results)} task(s)"]
    for index, label, result in sorted(results, key=lambda item: item[0]):
        lines.append(f"\n## {index}. {label}\n{_format_result(result)}")
    return "\n".join(lines)


async def _run_one(
    run: _SubagentRun,
    manager: SubagentSessionManager,
    on_update: Callable[[str], None] | None,
) -> SubagentTaskResult:
    task = run.task
    return await manager.execute(
        description=task["description"],
        prompt=_bounded_prompt(task["prompt"]),
        subagent_type=task["subagent_type"],
        mode=cast(SubagentMode, task["mode"]),
        run_id=run.run_id,
        batch_id=run.batch_id,
        task_index=run.task_index,
        on_update=on_update,
    )


def _task_update(
    index: int,
    on_update: Callable[[str], None] | None,
) -> Callable[[str], None] | None:
    if on_update is None:
        return None

    def update(line: str) -> None:
        on_update(f"[{index}]   {line}")

    return update


def _format_single(result: SubagentTaskResult, task: dict[str, str]) -> str:
    body = _format_result(result)
    if task["mode"] == "continuable":
        return f"Continuable child session: {result.child_session_id}\n\n{body}"
    return body


def _format_continuation(result: SubagentTaskResult) -> str:
    return f"Child session: {result.child_session_id}\n\n{_format_result(result)}"


def _format_result(result: SubagentTaskResult) -> str:
    if result.status == "completed":
        return result.answer
    if result.status == "cancelled":
        return "Cancelled"
    return f"Error: {result.error}"


def _bounded_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\nKeep this subagent task bounded: inspect only what is needed, "
        + "avoid broad test suites or exhaustive scans unless explicitly requested, "
        + "and return a concise summary."
    )
