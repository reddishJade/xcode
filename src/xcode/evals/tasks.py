from __future__ import annotations

from .schema import EvalTask

"""预定义的 eval task 套件。每套侧重一个能力维度。

设计原则（agent.md §评测）：
- task → trial → grader → transcript → outcome
- grader 优先代码评分器（编译、测试、文件内容）
- LLM-as-judge 作为补充
- pass@k / pass^k 区分探索能力与回归稳定性
"""


def coding() -> tuple[EvalTask, ...]:
    """编码能力：读代码、写代码、改代码，最终由文件证据和编译验证。"""
    return (
        EvalTask(
            id="write-python-function",
            prompt=(
                "Create a file called fix_bug.py in the project root. "
                "It should contain a function is_palindrome(s) that returns True "
                "if string s is a palindrome (case-insensitive, ignoring spaces and punctuation), "
                "False otherwise. Use a two-pointer approach."
            ),
            expected_tool_calls=("write_file",),
            tags=("coding", "write"),
            metadata={
                "evidence": {
                    "files": [
                        {
                            "path": "fix_bug.py",
                            "exists": True,
                            "contains": ("is_palindrome", "def "),
                        },
                    ],
                },
            },
        ),
        EvalTask(
            id="edit-function-logic",
            prompt=(
                "Read src/xcode/evals/runner.py, find the function _build_run_metrics. "
                "Add a new metric key 'total_steps' that sums trial.metrics['steps'] across all trials. "
                "Write the change."
            ),
            expected_tool_calls=("read_file", "edit_file"),
            tags=("coding", "edit"),
            metadata={
                "evidence": {
                    "files": [
                        {
                            "path": "src/xcode/evals/runner.py",
                            "contains": ("total_steps",),
                        },
                    ],
                },
            },
        ),
        EvalTask(
            id="find-and-fix-bug",
            prompt=(
                "Read src/xcode/evals/graders.py and find a bug: "
                "the function _parse_judge_response splits on 'PASS:' and 'FAIL:' but "
                "the output format in JUDGE_PROMPT_TEMPLATE uses 'PASS|FAIL: <number>: <reason>'. "
                "Fix the parsing to handle the 'PASS|FAIL:' prefix correctly."
            ),
            expected_tool_calls=("read_file", "edit_file"),
            tags=("coding", "debug"),
            metadata={
                "evidence": {
                    "files": [
                        {
                            "path": "src/xcode/evals/graders.py",
                            "not_contains": ('line.startswith("PASS:")',),
                        },
                    ],
                },
            },
        ),
    )


def plan_tasks() -> tuple[EvalTask, ...]:
    """规划 + 执行：先调研再实现，验证最终产出。"""
    return (
        EvalTask(
            id="plan-research-then-implement",
            mode="act",
            prompt=(
                "First, find all functions in src/xcode/evals/graders.py that take "
                "a parameter named 'events'. "
                "Then, add a new function `grade_tool_errors(task, events)` "
                "that returns a GraderResult checking if len(tool_errors) <= task.max_tool_errors. "
                "Append it to the end of the file."
            ),
            expected_tool_calls=("grep_search", "read_file", "edit_file"),
            tags=("plan", "implement"),
            metadata={
                "evidence": {
                    "files": [
                        {
                            "path": "src/xcode/evals/graders.py",
                            "contains": ("grade_tool_errors", "tool_errors"),
                        },
                    ],
                },
            },
        ),
    )


def all_suites() -> tuple[EvalTask, ...]:
    result: list[EvalTask] = []
    result.extend(coding())
    result.extend(plan_tasks())
    return tuple(result)


SUITES: dict[str, tuple[EvalTask, ...]] = {
    "coding": coding(),
    "plan": plan_tasks(),
    "all": all_suites(),
}
