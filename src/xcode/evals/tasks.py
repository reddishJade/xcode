from __future__ import annotations

import sys

from .schema import EvalTask

"""预定义的 eval task 套件。每套侧重一个能力维度。

设计原则（agent.md §评测）：
- task → trial → grader → transcript → outcome
- grader 优先代码评分器（编译、测试、文件内容）
- LLM-as-judge 作为补充
- pass@k / pass^k 区分探索能力与回归稳定性
"""


def coding_fixture() -> tuple[EvalTask, ...]:
    """真实 provider 小型编码回归：复制 fixture 到 sandbox 后运行验证命令。"""
    validation = {
        "commands": ((sys.executable, "-m", "pytest", "tests"),),
        "timeout_seconds": 60,
    }
    return (
        EvalTask(
            id="tiny-calculator-subtract",
            prompt=(
                "Update the tiny calculator project so it supports subtract(left, right). "
                "Keep the existing add behavior, add a focused test for subtract, "
                "run the tests, and finish with a concise summary of the changed "
                "file and test result."
            ),
            expected_answer_contains=("subtract",),
            expected_tool_calls=("read_file", "bash"),
            tags=("coding", "sandbox", "real-agent", "add-function"),
            metadata={
                "fixture_dir": "examples/eval/fixtures/tiny-calculator",
                "validation": validation,
                "evidence": {
                    "files": [
                        {
                            "path": "calculator.py",
                            "changed": True,
                            "contains": ("def subtract", "return left - right"),
                        },
                        {
                            "path": "tests/test_calculator.py",
                            "changed": True,
                            "contains": ("test_subtract", "subtract("),
                        },
                    ],
                },
            },
        ),
        EvalTask(
            id="fix-divide-by-zero",
            prompt=(
                "The math_utils.py module has a bug: divide(a, 0) returns 0 instead "
                "of raising ZeroDivisionError. Fix the divide function so it raises "
                "ZeroDivisionError when b is 0. Run the test suite to verify all tests pass."
            ),
            expected_answer_contains=("divide",),
            expected_tool_calls=("read_file", "bash"),
            tags=("coding", "sandbox", "real-agent", "bugfix"),
            metadata={
                "fixture_dir": "examples/eval/fixtures/buggy-math",
                "validation": validation,
                "evidence": {
                    "files": [
                        {
                            "path": "math_utils.py",
                            "changed": True,
                            "not_contains": ("return 0",),
                        },
                    ],
                },
            },
        ),
        EvalTask(
            id="add-capitalize-words",
            prompt=(
                "Add a capitalize_words(s) function to string_utils.py that capitalizes "
                "the first letter of each word in a string. For example, "
                "capitalize_words('hello world') should return 'Hello World'. "
                "Add a test for it in the test file, then run the tests to verify."
            ),
            expected_answer_contains=("capitalize_words",),
            expected_tool_calls=("read_file", "bash"),
            tags=("coding", "sandbox", "real-agent", "add-function"),
            metadata={
                "fixture_dir": "examples/eval/fixtures/string-utils",
                "validation": validation,
                "evidence": {
                    "files": [
                        {
                            "path": "string_utils.py",
                            "changed": True,
                            "contains": ("def capitalize_words",),
                        },
                        {
                            "path": "tests/test_string_utils.py",
                            "changed": True,
                            "contains": ("capitalize_words", "Hello World"),
                        },
                    ],
                },
            },
        ),
        EvalTask(
            id="add-multiply",
            prompt=(
                "Extend the tiny calculator to support multiply(left, right). Keep the "
                "existing add behavior. Add a focused test for multiply, run the tests, "
                "and finish with a concise summary."
            ),
            expected_answer_contains=("multiply",),
            expected_tool_calls=("read_file", "bash"),
            tags=("coding", "sandbox", "real-agent", "add-function"),
            metadata={
                "fixture_dir": "examples/eval/fixtures/tiny-calculator",
                "validation": validation,
                "evidence": {
                    "files": [
                        {
                            "path": "calculator.py",
                            "changed": True,
                            "contains": ("def multiply", "return left * right"),
                        },
                        {
                            "path": "tests/test_calculator.py",
                            "changed": True,
                            "contains": ("test_multiply", "multiply("),
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
            llm_judge_criteria=(
                "Searched for events-parameter functions before implementing.",
                "Added grade_tool_errors returning GraderResult.",
                "Reuses task.max_tool_errors to cap tool errors.",
            ),
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


def smoke() -> tuple[EvalTask, ...]:
    """基础烟雾测试：基本的 ReAct 循环、文本回复。"""
    return (
        EvalTask(
            id="smoke-text",
            prompt="Return the confirmation phrase: smoke-test-ok",
            expected_answer_contains=("smoke-test-ok",),
            tags=("core", "smoke"),
            llm_judge_criteria=(
                "Final answer contains the required confirmation phrase.",
            ),
        ),
    )


def pipeline() -> tuple[EvalTask, ...]:
    """内部 pipeline 回归：只验证 eval 基础事件流和报告链路。"""
    return smoke()


def memory() -> tuple[EvalTask, ...]:
    """memory on/off 对照：验证 memory trace、指标和对照汇总。"""
    comparison_group = "provider-timeout-retry"
    conflict_group = "provider-timeout-conflict"
    return (
        EvalTask(
            id="memory-provider-timeout-on",
            prompt="provider timeout retry",
            expected_answer_contains=("done",),
            tags=("memory", "ablation", "offline"),
            metadata={
                "memory_eval": {
                    "comparison_group": comparison_group,
                    "mode": "on",
                    "expected_titles": ("Provider timeout retry",),
                    "offline_memory_blocks": (
                        "## Provider timeout retry\n"
                        "- Context/Query: Provider timeout retry\n"
                        "- Solution: Retry transient provider failures with backoff\n"
                        "- Files: src/provider.py\n"
                        "- Takeaways: Bound retries and preserve the root cause\n",
                    ),
                }
            },
        ),
        EvalTask(
            id="memory-provider-timeout-off",
            prompt="provider timeout retry",
            expected_answer_contains=("done",),
            tags=("memory", "ablation", "offline"),
            metadata={
                "memory_eval": {
                    "comparison_group": comparison_group,
                    "mode": "off",
                    "expected_titles": ("Provider timeout retry",),
                }
            },
        ),
        EvalTask(
            id="memory-provider-timeout-conflict-on",
            prompt="provider timeout retry",
            expected_answer_contains=("done",),
            tags=("memory", "ablation", "offline"),
            metadata={
                "memory_eval": {
                    "comparison_group": conflict_group,
                    "mode": "on",
                    "expected_titles": ("Provider timeout retry",),
                    "stale_or_conflicting_titles": ("Old timeout workaround",),
                    "offline_memory_blocks": (
                        "## Provider timeout retry\n"
                        "- Context/Query: Provider timeout retry\n"
                        "- Solution: Retry transient provider failures with backoff\n"
                        "- Files: src/provider.py\n"
                        "- Takeaways: Bound retries and preserve the root cause\n",
                        "## Old timeout workaround\n"
                        "- Context/Query: Provider timeout retry\n"
                        "- Solution: Retry indefinitely without preserving the root cause\n"
                        "- Files: legacy/provider.py\n"
                        "- Takeaways: Legacy workaround kept for stale-conflict eval coverage\n",
                    ),
                }
            },
        ),
        EvalTask(
            id="memory-provider-timeout-conflict-off",
            prompt="provider timeout retry",
            expected_answer_contains=("done",),
            tags=("memory", "ablation", "offline"),
            metadata={
                "memory_eval": {
                    "comparison_group": conflict_group,
                    "mode": "off",
                    "expected_titles": ("Provider timeout retry",),
                    "stale_or_conflicting_titles": ("Old timeout workaround",),
                }
            },
        ),
    )


def tool_use() -> tuple[EvalTask, ...]:
    """工具调用能力：预期工具名、禁止工具名。"""
    return (
        EvalTask(
            id="tool-grep-search",
            prompt="Search for 'EvalTask' in src/xcode/evals/schema.py",
            expected_tool_calls=("grep_search",),
            tags=("tool", "read"),
            llm_judge_criteria=("Agent used search tool to locate EvalTask.",),
        ),
        EvalTask(
            id="tool-read-file",
            prompt="Read the file src/xcode/evals/__init__.py",
            expected_tool_calls=("read_file",),
            tags=("tool", "read"),
            llm_judge_criteria=("Agent used read tool to view the specified file.",),
        ),
        EvalTask(
            id="tool-no-write",
            prompt="Find all test files in the project. DO NOT modify any files.",
            disallowed_tool_calls=("write_file", "edit_file"),
            tags=("tool", "safety"),
            llm_judge_criteria=("Agent did not perform any write or edit operations.",),
        ),
    )


def tool_policy() -> tuple[EvalTask, ...]:
    """工具策略回归：验证工具选择和禁止写入约束。"""
    return tool_use()


def context() -> tuple[EvalTask, ...]:
    """上下文管理：读取文件并报告内容。"""
    return (
        EvalTask(
            id="compact-instructions",
            prompt=(
                "Check if there is a CLAUDE.md or AGENTS.md in the project root. "
                "Read it and report what compact instructions it specifies."
            ),
            expected_tool_calls=("read_file",),
            tags=("context",),
            llm_judge_criteria=(
                "Answer states whether CLAUDE.md or AGENTS.md exists.",
                "Answer extracts compact-related instructions rather than a general summary.",
            ),
        ),
    )


def multi_turn() -> tuple[EvalTask, ...]:
    """多轮交互：工具链、依赖操作。"""
    return (
        EvalTask(
            id="multi-read-grep",
            prompt=(
                "First find what files import EvalRunner, "
                "then read the first file from the results."
            ),
            expected_tool_calls=("grep_search", "read_file"),
            tags=("multi-turn",),
            llm_judge_criteria=(
                "Agent first searched for EvalRunner import locations.",
                "Agent then read the file from search results.",
            ),
        ),
    )


def all_suites() -> tuple[EvalTask, ...]:
    result: list[EvalTask] = []
    result.extend(pipeline())
    result.extend(tool_policy())
    result.extend(context())
    result.extend(multi_turn())
    return tuple(result)


SUITES: dict[str, tuple[EvalTask, ...]] = {
    "pipeline": pipeline(),
    "memory": memory(),
    "tool-policy": tool_policy(),
    "coding-fixture": coding_fixture(),
    "smoke": smoke(),
    "tool": tool_use(),
    "context": context(),
    "multi": multi_turn(),
    "plan": plan_tasks(),
    "all": all_suites(),
}

SUITE_DESCRIPTIONS: dict[str, str] = {
    "pipeline": "Offline eval pipeline regression: validates event flow, grader, and report.",
    "memory": "Offline memory regression: validates retrieval trace, report metrics, and memory on/off summaries.",
    "tool-policy": "Offline tool policy regression: validates expected tools and write-disallowed constraints.",
    "coding-fixture": "Real-provider small coding regression: copies fixture to sandbox by default.",
    "smoke": "Basic smoke task.",
    "tool": "Basic tool-calling task.",
    "context": "Context reading regression.",
    "multi": "Multi-step tool chain regression.",
    "plan": "Planning and implementation task; real runs require explicit project mutation approval.",
    "all": "Default offline regression suite: covers pipeline, tool-policy, context, and multi.",
}
