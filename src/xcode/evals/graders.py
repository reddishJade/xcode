from __future__ import annotations

from xcode.harness.agent_runtime import StructuredAgentEvent

from .schema import EvalTask, GraderResult


def grade_events(
    task: EvalTask,
    events: list[StructuredAgentEvent],
    answer: str,
    runtime_error: BaseException | None = None,
) -> tuple[GraderResult, ...]:
    tool_calls = [
        event.data.name
        for event in events
        if event.type == "tool_use" and hasattr(event.data, "name")
    ]
    tool_errors = [
        event
        for event in events
        if event.type == "tool_result"
        and getattr(event.data, "status", "ok") not in {"ok", "interrupted"}
    ]
    graders: list[GraderResult] = []
    graders.append(
        GraderResult(
            name="runtime_error",
            passed=runtime_error is None,
            details=""
            if runtime_error is None
            else f"{type(runtime_error).__name__}: {runtime_error}",
        )
    )
    graders.append(
        GraderResult(
            name="final_event",
            passed=any(event.type == "final" for event in events),
            details="missing final event"
            if not any(event.type == "final" for event in events)
            else "",
        )
    )
    for expected in task.expected_answer_contains:
        graders.append(
            GraderResult(
                name=f"answer_contains:{expected}",
                passed=expected in answer,
                details=""
                if expected in answer
                else f"answer did not contain {expected!r}",
            )
        )
    for tool_name in task.expected_tool_calls:
        graders.append(
            GraderResult(
                name=f"expected_tool:{tool_name}",
                passed=tool_name in tool_calls,
                details=""
                if tool_name in tool_calls
                else f"observed tools: {tool_calls}",
            )
        )
    for tool_name in task.disallowed_tool_calls:
        graders.append(
            GraderResult(
                name=f"disallowed_tool:{tool_name}",
                passed=tool_name not in tool_calls,
                details=""
                if tool_name not in tool_calls
                else f"disallowed tool was called: {tool_name}",
            )
        )
    graders.append(
        GraderResult(
            name="max_tool_errors",
            passed=len(tool_errors) <= task.max_tool_errors,
            details=(
                ""
                if len(tool_errors) <= task.max_tool_errors
                else f"tool errors {len(tool_errors)} exceeded {task.max_tool_errors}"
            ),
        )
    )
    return tuple(graders)
