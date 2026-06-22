"""Eval 确定性与 LLM grader。"""

from __future__ import annotations

from xcode.ai.events import FinalMessage, TextDelta
from xcode.ai.providers.protocol import StreamProvider
from xcode.ai.types import StreamOptions
from xcode.harness.agent_runtime import StructuredAgentEvent

from .schema import EvalTask, GraderResult


# ── 确定性 grader ──


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
        return _skipped_llm_judge("judge provider is unavailable")

    tool_calls = [
        f"{event.data.name}({getattr(event.data, 'input', {})})"
        for event in events
        if event.type == "tool_use" and hasattr(event.data, "name")
    ]
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
        return _skipped_llm_judge(f"judge provider failed: {type(exc).__name__}: {exc}")

    judge_response = "".join(response_parts).strip() or final_content.strip()
    if not judge_response:
        return _skipped_llm_judge("judge provider returned no text")

    parsed = _parse_judge_response(judge_response, task.llm_judge_criteria)
    if not parsed:
        return _skipped_llm_judge("judge output could not be parsed")
    return parsed


def _skipped_llm_judge(reason: str) -> tuple[GraderResult, ...]:
    """构建不影响 trial 成败的显式 skipped grader。"""
    return (
        GraderResult(
            name="llm_judge:skipped",
            passed=True,
            details=reason,
            skipped=True,
        ),
    )


def _parse_judge_response(
    response: str,
    criteria: tuple[str, ...],
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
                )
            )

    return tuple(results)
