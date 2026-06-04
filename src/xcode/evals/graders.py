from __future__ import annotations

from typing import Protocol, TypeGuard

from xcode.harness.agent_runtime import StructuredAgentEvent

from .schema import EvalTask, GraderResult


class JudgeAskProvider(Protocol):
    def ask(self, prompt: str) -> str: ...


class JudgeRunResult(Protocol):
    answer: str


class JudgeRunProvider(Protocol):
    def run(self, prompt: str) -> JudgeRunResult: ...


def _has_ask(provider: object) -> TypeGuard[JudgeAskProvider]:
    return hasattr(provider, "ask")


def _has_run(provider: object) -> TypeGuard[JudgeRunProvider]:
    return hasattr(provider, "run")


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


JUDGE_PROMPT_TEMPLATE = """你是一个严格的评测员。你需要根据以下标准评判一个 AI Agent 的输出。

## 任务描述
{task_prompt}

## 评判标准
{criteria_list}

## Agent 的最终回答
{answer}

## Agent 调用过的工具
{tool_calls_text}

## 评测要求
对每条评判标准，输出一行，格式如下（PASS|FAIL 表示二者选一）：
PASS|FAIL: <编号>: <理由>

例如：
PASS: 1: 代码编译通过
FAIL: 2: 代码存在未定义变量

仅输出评测行，不要输出其他内容。"""


def run_llm_judge(
    task: EvalTask,
    answer: str,
    events: list[StructuredAgentEvent],
    judge_provider: object | None = None,
) -> tuple[GraderResult, ...]:
    """使用 LLM 作为评判标准评委。

    judge_provider 必须具有 `ask(question: str) -> str` 接口。
    当 judge_provider 不可用、调用异常或输出解析失败时返回空 tuple，
    不产生 grader 条目 —— 调用方据此判断 judge 未执行，不计入 success 判定。
    """
    if not task.llm_judge_criteria:
        return ()

    if judge_provider is None:
        return ()

    tool_calls = [
        f"{event.data.name}({getattr(event.data, 'input', {})})"
        for event in events
        if event.type == "tool_use" and hasattr(event.data, "name")
    ]
    tool_calls_text = "\n".join(tool_calls) if tool_calls else "（无工具调用）"

    criteria_list = "\n".join(
        f"{i + 1}. {criterion}" for i, criterion in enumerate(task.llm_judge_criteria)
    )

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        task_prompt=task.prompt,
        criteria_list=criteria_list,
        answer=answer or "（Agent 未输出最终回答）",
        tool_calls_text=tool_calls_text,
    )

    try:
        if _has_ask(judge_provider):
            judge_response = judge_provider.ask(prompt)
        elif _has_run(judge_provider):
            # StructuredAgent.run() 返回具有 .answer 属性的对象
            result = judge_provider.run(prompt)
            judge_response = result.answer
        else:
            return ()
    except Exception:
        return ()

    return _parse_judge_response(judge_response, task.llm_judge_criteria)


def _parse_judge_response(
    response: str,
    criteria: tuple[str, ...],
) -> tuple[GraderResult, ...]:
    """解析 LLM judge 的结构化输出。

    支持 ``PASS: <编号>: <理由>`` / ``FAIL: <编号>: <理由>`` 和
    ``PASS|FAIL: <编号>: <理由>`` 两种格式。
    当无任何一行被成功解析时返回空 tuple（调用方据此不纳入 success 判定）。
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

        if rest is None:
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
