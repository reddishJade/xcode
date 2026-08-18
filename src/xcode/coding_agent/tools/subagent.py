from __future__ import annotations

import asyncio
from asyncio import run as async_run
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from xcode.agent.agent import Agent
from xcode.agent.config import AgentLoopConfig
from xcode.ai.providers.base import ModelProvider
from xcode.agent.types import (
    CancellationSignal,
    SubagentRenderIntent,
    ToolOutput,
    ToolSpec,
)
from xcode.harness.agent_runtime.tool_gate import ToolGate
from xcode.harness.session.subagent_runs import SubagentRunEvent, SubagentRunStatus


BUILD_SUBAGENT_PROMPTS: dict[str, str] = {
    "coding": (
        "You are an expert software engineer with full file system access. "
        "Complete the assigned task using available tools.\n\n"
        "- Use read/grep/glob/list_dir to explore before making changes.\n"
        "- Use bash to run tests, linters, or build commands.\n"
        "- Use edit/write to make changes.\n"
        "- Prefer small, focused changes and verify them.\n"
        "- Return a concise summary of what you did."
    ),
    "research": (
        "You are a thorough research assistant with file system and web access.\n\n"
        "- Use read/grep/glob/list_dir to inspect local files.\n"
        "- Use websearch/webfetch to gather current external information.\n"
        "- Cite sources and files when possible.\n"
        "- Return a structured summary of your findings."
    ),
    "default": (
        "You are a helpful AI assistant with file system access. "
        "Complete the assigned task and return a concise summary."
    ),
}


MAX_BATCH_TASKS = 16
DEFAULT_MAX_CONCURRENT = 4
SubagentLifecycleSink = Callable[[SubagentRunEvent], object]


@dataclass(frozen=True)
class _SubagentRun:
    run_id: str
    batch_id: str
    task_index: int
    task: dict[str, str]


class _SubagentHandler:
    """子代理处理器；门控由产品装配完成后绑定，未绑定时拒绝运行。"""

    def __init__(
        self,
        model: ModelProvider,
        coding_tools: list[ToolSpec],
        research_tools: list[ToolSpec],
        cancellation_token: CancellationSignal | None,
        lifecycle_sink: SubagentLifecycleSink,
    ) -> None:
        self._model = model
        self._coding_tools = coding_tools
        self._research_tools = research_tools
        self._cancellation_token = cancellation_token
        self._lifecycle_sink = lifecycle_sink
        self._permission_gate: ToolGate | None = None

    def bind_permission_gate(self, gate: ToolGate) -> None:
        """绑定父代理权限门控。"""
        self._permission_gate = gate

    def __call__(
        self,
        data: dict[str, Any],
        on_update: Callable[[str], None] | None = None,
    ) -> str:
        gate = self._permission_gate
        if gate is None:
            return "Error: subagent permission gate is not configured"
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

        async def _run() -> str:
            if len(tasks) == 1:
                return await _run_one(
                    runs[0],
                    self._model,
                    self._coding_tools,
                    self._research_tools,
                    self._cancellation_token,
                    on_update,
                    gate,
                    self._lifecycle_sink,
                )
            return await _run_batch(
                runs,
                max_concurrent,
                self._model,
                self._coding_tools,
                self._research_tools,
                self._cancellation_token,
                on_update,
                gate,
                self._lifecycle_sink,
            )

        result = async_run(_run())
        return ToolOutput(
            result,
            render_intent=SubagentRenderIntent(
                batch_id=batch_id,
                run_ids=tuple(run.run_id for run in runs),
            ),
        )


def bind_subagent_permission_gate(
    registry: tuple[ToolSpec, ...], gate: ToolGate
) -> None:
    """把产品层已装配的权限门控绑定到 subagent 工具。"""
    for spec in registry:
        if spec.name == "subagent" and isinstance(spec.handler, _SubagentHandler):
            spec.handler.bind_permission_gate(gate)


def build_subagent_tool(
    model: ModelProvider,
    coding_tools: list[ToolSpec],
    research_tools: list[ToolSpec],
    lifecycle_sink: SubagentLifecycleSink,
    cancellation_token: CancellationSignal | None = None,
) -> ToolSpec:
    handler = _SubagentHandler(
        model,
        coding_tools,
        research_tools,
        cancellation_token,
        lifecycle_sink,
    )

    return ToolSpec(
        name="subagent",
        description=(
            "Launch one or more subagents for self-contained tasks. "
            "Each subagent runs independently with file system and web access. "
            "Use tasks for parallel fan-out when the work items do not depend on each other.\n\n"
            "Available subagent types:\n"
            "- coding: Expert software engineer (default)\n"
            "- research: Research assistant with web access\n"
            "- default: General-purpose assistant"
        ),
        input_hint=(
            'JSON: {"description":"short label","prompt":"...", "subagent_type":"coding"} '
            'or {"tasks":[{"description":"label","prompt":"..."}],"max_concurrent":4}'
        ),
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short 3-7 word label for a single delegated task",
                },
                "prompt": {
                    "type": "string",
                    "description": "Complete task prompt for a single subagent",
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["coding", "research", "default"],
                    "description": "Type of subagent (default: coding)",
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
                    "description": "Independent tasks to run in parallel",
                },
                "max_concurrent": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_BATCH_TASKS,
                    "description": "Maximum parallel subagents for tasks (default: 4)",
                },
            },
            "additionalProperties": False,
        },
        prompt_snippet="Delegate independent work to one or more subagents",
        prompt_guidelines=(
            "Use subagent for substantial independent investigation or implementation tasks.",
            "Use tasks only when the work items can run independently; the main agent synthesizes results.",
            "Include all necessary context in each prompt; subagents do not inherit conversation history.",
            "Do not poll delegated tasks; subagent returns the final result when done.",
        ),
    )


def _parse_tasks(data: dict[str, Any]) -> list[dict[str, str]] | str:
    raw_tasks = data.get("tasks")
    if raw_tasks is None:
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return "Error: prompt is required"
        return [
            {
                "description": str(data.get("description", "subagent")).strip()
                or "subagent",
                "prompt": prompt,
                "subagent_type": str(data.get("subagent_type", "coding")).strip()
                or "coding",
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
    model: ModelProvider,
    coding_tools: list[ToolSpec],
    research_tools: list[ToolSpec],
    cancellation_token: CancellationSignal | None,
    on_update: Callable[[str], None] | None,
    permission_gate: ToolGate,
    lifecycle_sink: SubagentLifecycleSink,
) -> str:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(run: _SubagentRun) -> tuple[int, str, str]:
        async with semaphore:
            index = run.task_index
            task = run.task
            label = task["description"]
            task_update = _task_update(index, on_update)
            if on_update is not None:
                on_update(f"[{index}] → {label}")
            result = await _run_one(
                run,
                model,
                coding_tools,
                research_tools,
                cancellation_token,
                task_update,
                permission_gate,
                lifecycle_sink,
            )
            if on_update is not None:
                on_update(f"[{index}] ✓ {label}")
            return index, label, result

    results = await asyncio.gather(
        *(limited(run) for run in runs)
    )
    lines = [f"Subagent batch completed: {len(results)} task(s)"]
    for index, label, result in sorted(results):
        lines.append(f"\n## {index}. {label}\n{result.strip()}")
    return "\n".join(lines)


def _task_update(
    index: int,
    on_update: Callable[[str], None] | None,
) -> Callable[[str], None] | None:
    if on_update is None:
        return None

    def update(line: str) -> None:
        on_update(f"[{index}]   {line}")

    return update


async def _run_one(
    run: _SubagentRun,
    model: ModelProvider,
    coding_tools: list[ToolSpec],
    research_tools: list[ToolSpec],
    cancellation_token: CancellationSignal | None,
    on_update: Callable[[str], None] | None,
    permission_gate: ToolGate,
    lifecycle_sink: SubagentLifecycleSink,
) -> str:
    task = run.task
    lifecycle_sink(_lifecycle_event(run, "started"))
    subagent_type = task["subagent_type"]
    system_prompt = BUILD_SUBAGENT_PROMPTS.get(
        subagent_type, BUILD_SUBAGENT_PROMPTS["default"]
    )
    raw_tools = coding_tools if subagent_type == "coding" else research_tools
    child_gate = permission_gate.fork_for_subagent()
    registry = tuple(raw_tools)
    gate_snapshot = child_gate.snapshot_for(registry)
    adapted = child_gate.adapt_tools(registry)
    loop_config = AgentLoopConfig(
        provider=model,
        before_tool_call=child_gate.build_before_tool_hook(gate_snapshot),
        after_tool_call=child_gate.build_after_tool_hook(gate_snapshot),
        is_tool_productive=child_gate.build_is_tool_productive_hook(gate_snapshot),
    )
    agent = Agent(tools=adapted, model=model, system_prompt=system_prompt)
    outcomes = await asyncio.gather(
        agent.prompt(
            _bounded_prompt(task["prompt"]),
            loop_config=loop_config,
            signal=cancellation_token,
            on_update=on_update,
        ),
        return_exceptions=True,
    )
    outcome = outcomes[0]
    if isinstance(outcome, BaseException):
        status: SubagentRunStatus = (
            "cancelled" if isinstance(outcome, asyncio.CancelledError) else "failed"
        )
        error = f"{type(outcome).__name__}: {outcome}"
        lifecycle_sink(_lifecycle_event(run, status, error=error))
        return f"Error: {error}"
    if cancellation_token is not None and cancellation_token.is_cancelled():
        lifecycle_sink(_lifecycle_event(run, "cancelled", summary=outcome))
        return outcome
    lifecycle_sink(_lifecycle_event(run, "completed", summary=outcome))
    return outcome


def _lifecycle_event(
    run: _SubagentRun,
    status: SubagentRunStatus,
    *,
    summary: str = "",
    error: str = "",
) -> SubagentRunEvent:
    return SubagentRunEvent(
        run_id=run.run_id,
        batch_id=run.batch_id,
        task_index=run.task_index,
        description=run.task["description"],
        subagent_type=run.task["subagent_type"],
        status=status,
        summary=summary.strip()[:4000],
        error=error.strip()[:1000],
    )


def _bounded_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\nKeep this subagent task bounded: inspect only what is needed, "
        + "avoid broad test suites or exhaustive scans unless explicitly requested, "
        + "and return a concise summary."
    )
