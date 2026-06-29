"""Eval 确定性与 LLM grader。"""

from __future__ import annotations

from xcode.ai.events import FinalMessage, TextDelta
from xcode.ai.providers.protocol import StreamProvider
from xcode.ai.types import StreamOptions
from xcode.harness.agent_runtime import StructuredAgentEvent
from xcode.harness.agent_runtime.events import (
    ToolResultStructuredEvent,
    ToolUseStructuredEvent,
)

from .schema import EvalTask, GraderResult


# ── 确定性 grader ──


def grade_events(
    task: EvalTask,
    events: list[StructuredAgentEvent],
    answer: str,
    runtime_error: BaseException | None = None,
) -> tuple[GraderResult, ...]:
    tool_use_events = [
        event
        for event in events
        if event.type == "tool_use" and hasattr(event.data, "name")
    ]
    tool_calls = [event.data.name for event in tool_use_events]
    tool_results = [
        event
        for event in events
        if event.type == "tool_result" and hasattr(event.data, "tool_use_id")
    ]
    tool_errors = [
        tool_result
        for tool_result in tool_results
        if getattr(tool_result.data, "status", "ok") not in {"ok", "interrupted"}
    ]
    graders: list[GraderResult] = []
    graders.append(
        GraderResult(
            name="runtime_error",
            passed=runtime_error is None,
            details=""
            if runtime_error is None
            else f"{type(runtime_error).__name__}: {runtime_error}",
            failure_category="environment" if runtime_error is not None else None,
        )
    )
    graders.append(
        GraderResult(
            name="final_event",
            passed=any(event.type == "final" for event in events),
            details="missing final event"
            if not any(event.type == "final" for event in events)
            else "",
            failure_category=(
                "stopping_condition"
                if not any(event.type == "final" for event in events)
                else None
            ),
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
                failure_category="understanding" if expected not in answer else None,
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
                failure_category="tool_selection"
                if tool_name not in tool_calls
                else None,
                required=False,
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
                failure_category=(
                    "tool_selection" if tool_name in tool_calls else None
                ),
                required=False,
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
            failure_category=(
                "tool_execution" if len(tool_errors) > task.max_tool_errors else None
            ),
            required=False,
        )
    )
    graders.extend(_grade_tool_policy(task, tool_use_events, tool_results, answer))
    return tuple(graders)


def _grade_tool_policy(
    task: EvalTask,
    tool_use_events: list[ToolUseStructuredEvent],
    tool_results: list[ToolResultStructuredEvent],
    answer: str,
) -> list[GraderResult]:
    policy = task.metadata.get("tool_policy")
    if not isinstance(policy, dict):
        return []
    graders: list[GraderResult] = []
    tool_names = [event.data.name for event in tool_use_events]
    result_by_tool_use_id = {
        str(event.data.tool_use_id): str(getattr(event.data, "content", ""))
        for event in tool_results
    }
    name_by_tool_use_id = {
        str(event.data.id): str(event.data.name) for event in tool_use_events
    }
    ordered_tools = _string_tuple_or_empty(policy.get("ordered_tools"))
    if ordered_tools:
        observed_positions: list[int] = []
        cursor = 0
        passed = True
        for tool_name in ordered_tools:
            try:
                position = tool_names.index(tool_name, cursor)
            except ValueError:
                passed = False
                break
            observed_positions.append(position)
            cursor = position + 1
        graders.append(
            GraderResult(
                name="tool_policy:ordered_tools",
                passed=passed,
                details=""
                if passed
                else f"expected ordered tools {ordered_tools}, observed {tool_names}",
                required=False,
                failure_category="planning" if not passed else None,
                evidence={
                    "expected": ordered_tools,
                    "observed": tool_names,
                    "positions": observed_positions,
                },
            )
        )
    for index, check in enumerate(
        _dict_tuple_or_empty(policy.get("argument_contains")), start=1
    ):
        tool_name = str(check.get("tool", "")).strip()
        arguments = check.get("arguments")
        if not tool_name or not isinstance(arguments, dict):
            continue
        matching_call = next(
            (
                event
                for event in tool_use_events
                if event.data.name == tool_name
                and _arguments_contain(event.data.input, arguments)
            ),
            None,
        )
        graders.append(
            GraderResult(
                name=f"tool_policy:arguments:{index}:{tool_name}",
                passed=matching_call is not None,
                details=""
                if matching_call is not None
                else f"missing {tool_name} call containing arguments {arguments!r}",
                required=False,
                failure_category="retrieval" if matching_call is None else None,
                evidence={"tool": tool_name, "arguments": arguments},
            )
        )
    for index, check in enumerate(
        _dict_tuple_or_empty(policy.get("result_contains")), start=1
    ):
        tool_name = str(check.get("tool", "")).strip()
        substrings = _string_tuple_or_empty(check.get("substrings"))
        if not tool_name or not substrings:
            continue
        matching_results = [
            text
            for tool_use_id, text in result_by_tool_use_id.items()
            if name_by_tool_use_id.get(tool_use_id) == tool_name
        ]
        passed = any(
            all(expected in text for expected in substrings)
            for text in matching_results
        )
        graders.append(
            GraderResult(
                name=f"tool_policy:result:{index}:{tool_name}",
                passed=passed,
                details=""
                if passed
                else f"{tool_name} results did not contain {substrings!r}",
                required=False,
                failure_category="tool_execution" if not passed else None,
                evidence={"tool": tool_name, "substrings": substrings},
            )
        )
    for index, check in enumerate(
        _dict_tuple_or_empty(policy.get("answer_contains_from_tool")), start=1
    ):
        tool_name = str(check.get("tool", "")).strip()
        substrings = _string_tuple_or_empty(check.get("substrings"))
        if not tool_name or not substrings:
            continue
        matching_results = [
            text
            for tool_use_id, text in result_by_tool_use_id.items()
            if name_by_tool_use_id.get(tool_use_id) == tool_name
        ]
        observed = next(
            (
                expected
                for expected in substrings
                if any(expected in text for text in matching_results)
            ),
            None,
        )
        passed = observed is not None and observed in answer
        graders.append(
            GraderResult(
                name=f"tool_policy:adopted_result:{index}:{tool_name}",
                passed=passed,
                details=""
                if passed
                else f"answer did not adopt any expected {tool_name} result snippet",
                required=False,
                failure_category="understanding" if not passed else None,
                evidence={
                    "tool": tool_name,
                    "substrings": substrings,
                    "answer": answer,
                },
            )
        )
    return graders


def _string_tuple_or_empty(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, list | tuple):
        return ()
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)


def _dict_tuple_or_empty(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list | tuple):
        return ()
    result: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
    return tuple(result)


def _arguments_contain(
    observed: object,
    expected: dict[str, object],
) -> bool:
    if not isinstance(observed, dict):
        return False
    for key, value in expected.items():
        if key not in observed:
            return False
        actual = observed[key]
        if isinstance(value, str):
            if value not in str(actual):
                return False
            continue
        if actual != value:
            return False
    return True


# ── LLM-as-judge grader ──


JUDGE_PROMPT_TEMPLATE = """You are a strict judge. Evaluate an AI Agent's output against the following criteria.

## Task Description
{task_prompt}

## Evaluation Criteria
{criteria_list}

## Agent's Final Answer
{answer}

## Tools Called by the Agent
{tool_calls_text}

## Requirements
For each criterion, output one line in the following format (PASS|FAIL means pick one):
PASS|FAIL: <number>: <reason>

Example:
PASS: 1: Code compiles successfully
FAIL: 2: Code contains undefined variables

Output only the evaluation lines, nothing else."""


async def run_llm_judge(
    task: EvalTask,
    answer: str,
    events: list[StructuredAgentEvent],
    judge_provider: StreamProvider | None = None,
) -> tuple[GraderResult, ...]:
    """通过统一 StreamProvider 协议执行 LLM-as-judge。"""
    if not task.llm_judge_criteria:
        return ()

    if judge_provider is None:
        return _judge_unavailable(task, "judge provider is unavailable")

    tool_calls: list[str] = []
    for event in events:
        if event.type == "tool_use":
            tc = event
            tool_calls.append(f"{tc.data.name}({tc.data.input})")
    tool_calls_text = "\n".join(tool_calls) if tool_calls else "(no tool calls)"

    criteria_list = "\n".join(
        f"{i + 1}. {criterion}" for i, criterion in enumerate(task.llm_judge_criteria)
    )

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        task_prompt=task.prompt,
        criteria_list=criteria_list,
        answer=answer or "(Agent did not produce a final answer)",
        tool_calls_text=tool_calls_text,
    )

    response_parts: list[str] = []
    final_content = ""
    try:
        async for event in judge_provider.stream(
            [{"role": "user", "content": prompt}],
            [],
            StreamOptions(temperature=0.0, max_tokens=2_000),
        ):
            if isinstance(event, TextDelta):
                response_parts.append(event.chunk)
            elif isinstance(event, FinalMessage):
                final_content = event.content
    except Exception as exc:
        return _judge_unavailable(
            task,
            f"judge provider failed: {type(exc).__name__}: {exc}",
        )

    judge_response = "".join(response_parts).strip() or final_content.strip()
    if not judge_response:
        return _judge_unavailable(task, "judge provider returned no text")

    parsed = _parse_judge_response(
        judge_response,
        task.llm_judge_criteria,
        required=task.llm_judge_required,
    )
    if not parsed:
        return _judge_unavailable(task, "judge output could not be parsed")
    return parsed


def _judge_unavailable(task: EvalTask, reason: str) -> tuple[GraderResult, ...]:
    """根据 task 配置构建 judge unavailable 结果。"""
    if task.llm_judge_required:
        return (
            GraderResult(
                name="llm_judge:required",
                passed=False,
                details=reason,
                required=True,
                skipped=False,
                failure_category="judge",
            ),
        )
    return (
        GraderResult(
            name="llm_judge:skipped",
            passed=True,
            details=reason,
            skipped=True,
            required=False,
            failure_category="judge",
        ),
    )


def _parse_judge_response(
    response: str,
    criteria: tuple[str, ...],
    *,
    required: bool,
) -> tuple[GraderResult, ...]:
    """解析 LLM judge 的结构化输出。

    支持 ``PASS: <编号>: <理由>`` / ``FAIL: <编号>: <理由>`` 和
    ``PASS|FAIL: <编号>: <理由>`` 两种格式。
    当无任何一行被成功解析时返回空 tuple。
    """
    results: list[GraderResult] = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        passed = False
        rest: str | None = None
        # 标准格式 PASS: / FAIL:
        if line.startswith("PASS:"):
            passed = True
            rest = line[5:].strip()
        elif line.startswith("FAIL:"):
            rest = line[5:].strip()
        # 容错：LLM 可能直接从模板粘贴 PASS|FAIL:
        elif line.startswith("PASS|FAIL:"):
            rest = line[10:].strip()
            if rest and rest[0] == "1":
                passed = True
        else:
            continue

        parts = rest.split(":", 1)
        criterion_name = parts[0].strip() if parts else "unknown"
        details = parts[1].strip() if len(parts) > 1 else ""
        results.append(
            GraderResult(
                name=f"llm_judge:{criterion_name}",
                passed=passed,
                details=details,
                required=required,
                failure_category="judge" if not passed else None,
            )
        )

    # 全都没解析出来 —— 调用方自行决定如何处理
    if not results:
        return ()

    # 补充 LLM 遗漏评判的标准
    parsed_names = {r.name.replace("llm_judge:", "") for r in results}
    for i, criterion in enumerate(criteria):
        key = str(i + 1)
        if key not in parsed_names and criterion not in parsed_names:
            results.append(
                GraderResult(
                    name=f"llm_judge:{key}",
                    passed=False,
                    details=f"judge did not evaluate: {criterion}",
                    required=required,
                    failure_category="judge",
                )
            )

    return tuple(results)
