from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

from .schema import EvalTask, TaskMetadata
from .validation import validation_commands

"""预定义的 eval task 套件。每套侧重一个能力维度。

设计原则（agent.md §评测）：
- task → trial → grader → transcript → outcome
- grader 优先代码评分器（编译、测试、文件内容）
- LLM-as-judge 作为补充
- pass@k / pass^k 区分探索能力与回归稳定性
"""

SuiteKind = Literal["regression", "capability"]


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    description: str
    kind: SuiteKind
    run_mode: str
    tasks: tuple[EvalTask, ...]


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
            capability="code.edit.function",
            expected_duration_seconds=120,
            difficulty="medium",
            run_mode="real",
            metadata=TaskMetadata.model_validate(
                {
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
                }
            ),
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
            capability="code.edit.bugfix",
            expected_duration_seconds=120,
            difficulty="medium",
            run_mode="real",
            metadata=TaskMetadata.model_validate(
                {
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
                }
            ),
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
            capability="code.edit.function",
            expected_duration_seconds=120,
            difficulty="medium",
            run_mode="real",
            metadata=TaskMetadata.model_validate(
                {
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
                }
            ),
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
            capability="code.edit.function",
            expected_duration_seconds=120,
            difficulty="medium",
            run_mode="real",
            metadata=TaskMetadata.model_validate(
                {
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
                }
            ),
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
            capability="planning.research-edit",
            expected_duration_seconds=90,
            difficulty="hard",
            run_mode="either",
            llm_judge_criteria=(
                "Searched for events-parameter functions before implementing.",
                "Added grade_tool_errors returning GraderResult.",
                "Reuses task.max_tool_errors to cap tool errors.",
            ),
            metadata=TaskMetadata.model_validate(
                {
                    "evidence": {
                        "files": [
                            {
                                "path": "src/xcode/evals/graders.py",
                                "contains": ("grade_tool_errors", "tool_errors"),
                            },
                        ],
                    },
                }
            ),
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
            capability="core.reply",
            expected_duration_seconds=5,
            difficulty="easy",
            run_mode="offline",
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
    base = {
        "capability": "memory.retrieval",
        "expected_duration_seconds": 10,
        "difficulty": "medium",
        "run_mode": "offline",
    }
    return (
        EvalTask(
            id="memory-provider-timeout-on",
            prompt="provider timeout retry",
            expected_answer_contains=("done",),
            tags=("memory", "ablation", "offline"),
            metadata=TaskMetadata.model_validate(
                {
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
                }
            ),
            **base,
        ),
        EvalTask(
            id="memory-provider-timeout-off",
            prompt="provider timeout retry",
            expected_answer_contains=("done",),
            tags=("memory", "ablation", "offline"),
            metadata=TaskMetadata.model_validate(
                {
                    "memory_eval": {
                        "comparison_group": comparison_group,
                        "mode": "off",
                        "expected_titles": ("Provider timeout retry",),
                    }
                }
            ),
            **base,
        ),
        EvalTask(
            id="memory-provider-timeout-conflict-on",
            prompt="provider timeout retry",
            expected_answer_contains=("done",),
            tags=("memory", "ablation", "offline"),
            metadata=TaskMetadata.model_validate(
                {
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
                }
            ),
            **base,
        ),
        EvalTask(
            id="memory-provider-timeout-conflict-off",
            prompt="provider timeout retry",
            expected_answer_contains=("done",),
            tags=("memory", "ablation", "offline"),
            metadata=TaskMetadata.model_validate(
                {
                    "memory_eval": {
                        "comparison_group": conflict_group,
                        "mode": "off",
                        "expected_titles": ("Provider timeout retry",),
                        "stale_or_conflicting_titles": ("Old timeout workaround",),
                    }
                }
            ),
            **base,
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
            capability="tool.search",
            expected_duration_seconds=10,
            difficulty="easy",
            run_mode="offline",
            llm_judge_criteria=("Agent used search tool to locate EvalTask.",),
            metadata=TaskMetadata.model_validate(
                {
                    "tool_policy": {
                        "argument_contains": (
                            {
                                "tool": "grep_search",
                                "arguments": {
                                    "path": "src/xcode/evals/schema.py",
                                    "query": "EvalTask",
                                },
                            },
                        ),
                        "result_contains": (
                            {
                                "tool": "grep_search",
                                "substrings": ("EvalTask",),
                            },
                        ),
                        "answer_contains_from_tool": (
                            {
                                "tool": "grep_search",
                                "substrings": ("EvalTask",),
                            },
                        ),
                    }
                }
            ),
        ),
        EvalTask(
            id="tool-read-file",
            prompt="Read the file src/xcode/evals/__init__.py",
            expected_tool_calls=("read_file",),
            tags=("tool", "read"),
            capability="tool.read",
            expected_duration_seconds=10,
            difficulty="easy",
            run_mode="offline",
            llm_judge_criteria=("Agent used read tool to view the specified file.",),
            metadata=TaskMetadata.model_validate(
                {
                    "tool_policy": {
                        "argument_contains": (
                            {
                                "tool": "read_file",
                                "arguments": {"path": "src/xcode/evals/__init__.py"},
                            },
                        ),
                    }
                }
            ),
        ),
        EvalTask(
            id="tool-no-write",
            prompt="Find all test files in the project. DO NOT modify any files.",
            disallowed_tool_calls=("write_file", "edit_file"),
            tags=("tool", "safety"),
            capability="tool.policy",
            expected_duration_seconds=10,
            difficulty="easy",
            run_mode="offline",
            llm_judge_criteria=("Agent did not perform any write or edit operations.",),
            metadata=TaskMetadata.model_validate(
                {
                    "tool_policy": {
                        "ordered_tools": ("grep_search", "read_file"),
                    }
                }
            ),
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
                "Check if there is an AGENTS.md in the project root. "
                "Read it and report what compact instructions it specifies."
            ),
            expected_tool_calls=("read_file",),
            tags=("context",),
            capability="context.read",
            expected_duration_seconds=15,
            difficulty="medium",
            run_mode="offline",
            llm_judge_criteria=(
                "Answer states whether AGENTS.md exists.",
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
            capability="trajectory.multi-step",
            expected_duration_seconds=20,
            difficulty="medium",
            run_mode="offline",
            llm_judge_criteria=(
                "Agent first searched for EvalRunner import locations.",
                "Agent then read the file from search results.",
            ),
            metadata=TaskMetadata.model_validate(
                {
                    "tool_policy": {
                        "ordered_tools": ("grep_search", "read_file"),
                    }
                }
            ),
        ),
    )


def fault_injection() -> tuple[EvalTask, ...]:
    """离线故障注入：验证恢复、重试与降级能力。"""
    return (
        EvalTask(
            id="fault-command-retry",
            prompt=(
                "Run the test command. If it fails because pytest is missing, diagnose "
                "the error, install or suggest the missing dependency, rerun, and summarize."
            ),
            expected_answer_contains=("pytest", "passed"),
            expected_tool_calls=("run_tests",),
            max_tool_errors=1,
            tags=("fault", "retry", "offline"),
            capability="recovery.command-retry",
            expected_duration_seconds=20,
            difficulty="medium",
            run_mode="offline",
            metadata=TaskMetadata.model_validate(
                {
                    "fault_injection": {
                        "scenario": "command_failure_retry",
                        "expected_failure_category": "tool_execution",
                    },
                    "tool_policy": {
                        "result_contains": (
                            {
                                "tool": "run_tests",
                                "substrings": ("pytest: command not found",),
                            },
                            {
                                "tool": "run_tests",
                                "substrings": ("2 passed",),
                            },
                        ),
                        "answer_contains_from_tool": (
                            {
                                "tool": "run_tests",
                                "substrings": ("2 passed", "pytest"),
                            },
                        ),
                    },
                }
            ),
        ),
        EvalTask(
            id="fault-wrong-path-recovery",
            prompt=(
                "Read src/xcode/evals/runner.py. If the path is wrong or missing, "
                "recover by searching for EvalRunner first, then open the correct file."
            ),
            expected_answer_contains=("EvalRunner", "runner.py"),
            expected_tool_calls=("read_file", "grep_search", "read_file"),
            max_tool_errors=1,
            tags=("fault", "retrieval", "offline"),
            capability="recovery.path-recovery",
            expected_duration_seconds=20,
            difficulty="medium",
            run_mode="offline",
            metadata=TaskMetadata.model_validate(
                {
                    "fault_injection": {
                        "scenario": "wrong_path_recovery",
                        "expected_failure_category": "retrieval",
                    },
                    "tool_policy": {
                        "ordered_tools": ("read_file", "grep_search", "read_file"),
                        "argument_contains": (
                            {
                                "tool": "grep_search",
                                "arguments": {"query": "EvalRunner"},
                            },
                        ),
                        "result_contains": (
                            {
                                "tool": "read_file",
                                "substrings": ("class EvalRunner",),
                            },
                        ),
                        "answer_contains_from_tool": (
                            {
                                "tool": "read_file",
                                "substrings": ("EvalRunner", "runner.py"),
                            },
                        ),
                    },
                }
            ),
        ),
        EvalTask(
            id="fault-provider-abort-degrade",
            prompt=(
                "The provider may abort mid-run. If that happens, give a concise degraded "
                "answer that states the interruption and safest next step."
            ),
            expected_answer_contains=("interrupted", "retry"),
            tags=("fault", "provider", "offline"),
            capability="recovery.provider-abort",
            expected_duration_seconds=10,
            difficulty="medium",
            run_mode="offline",
            metadata=TaskMetadata.model_validate(
                {
                    "fault_injection": {
                        "scenario": "provider_abort_degrade",
                        "expected_failure_category": "environment",
                    }
                }
            ),
        ),
    )


def regression() -> tuple[EvalTask, ...]:
    result: list[EvalTask] = []
    result.extend(pipeline())
    result.extend(tool_policy())
    result.extend(context())
    result.extend(multi_turn())
    result.extend(fault_injection())
    return tuple(result)


def capability() -> tuple[EvalTask, ...]:
    result: list[EvalTask] = []
    result.extend(tool_use())
    result.extend(context())
    result.extend(multi_turn())
    result.extend(plan_tasks())
    result.extend(coding_fixture())
    return tuple(result)


SUITE_SPECS: dict[str, SuiteSpec] = {
    "pipeline": SuiteSpec(
        name="pipeline",
        description="Offline eval pipeline regression: validates event flow, grader, and report.",
        kind="regression",
        run_mode="offline",
        tasks=pipeline(),
    ),
    "memory": SuiteSpec(
        name="memory",
        description="Offline memory regression: validates retrieval trace, report metrics, and memory on/off summaries.",
        kind="regression",
        run_mode="offline",
        tasks=memory(),
    ),
    "tool-policy": SuiteSpec(
        name="tool-policy",
        description="Offline tool policy regression: validates expected tools and write-disallowed constraints.",
        kind="regression",
        run_mode="offline",
        tasks=tool_policy(),
    ),
    "fault-injection": SuiteSpec(
        name="fault-injection",
        description="Offline fault-injection regression: validates retry, recovery, and degraded answers.",
        kind="regression",
        run_mode="offline",
        tasks=fault_injection(),
    ),
    "coding-fixture": SuiteSpec(
        name="coding-fixture",
        description="Real-provider small coding regression: copies fixture to sandbox by default.",
        kind="capability",
        run_mode="real",
        tasks=coding_fixture(),
    ),
    "smoke": SuiteSpec(
        name="smoke",
        description="Basic smoke task.",
        kind="regression",
        run_mode="offline",
        tasks=smoke(),
    ),
    "tool": SuiteSpec(
        name="tool",
        description="Basic tool-calling task.",
        kind="capability",
        run_mode="offline",
        tasks=tool_use(),
    ),
    "context": SuiteSpec(
        name="context",
        description="Context reading regression.",
        kind="regression",
        run_mode="offline",
        tasks=context(),
    ),
    "multi": SuiteSpec(
        name="multi",
        description="Multi-step tool chain regression.",
        kind="regression",
        run_mode="offline",
        tasks=multi_turn(),
    ),
    "plan": SuiteSpec(
        name="plan",
        description="Planning and implementation task; real runs require explicit project mutation approval.",
        kind="capability",
        run_mode="mixed",
        tasks=plan_tasks(),
    ),
    "regression": SuiteSpec(
        name="regression",
        description="Stable regression gate suite.",
        kind="regression",
        run_mode="offline",
        tasks=regression(),
    ),
    "capability": SuiteSpec(
        name="capability",
        description="Capability discovery suite with real and offline tasks.",
        kind="capability",
        run_mode="mixed",
        tasks=capability(),
    ),
    "all": SuiteSpec(
        name="all",
        description="Default offline regression suite: covers pipeline, tool-policy, context, and multi.",
        kind="regression",
        run_mode="offline",
        tasks=regression(),
    ),
}

SUITES: dict[str, tuple[EvalTask, ...]] = {
    name: spec.tasks for name, spec in SUITE_SPECS.items()
}

SUITE_DESCRIPTIONS: dict[str, str] = {
    name: spec.description for name, spec in SUITE_SPECS.items()
}


def validate_suite_registry() -> None:
    """自检内置 suite 定义。"""
    repo_root = Path(__file__).resolve().parents[3]
    for suite_name, spec in SUITE_SPECS.items():
        seen_ids: set[str] = set()
        for task in spec.tasks:
            if task.id in seen_ids:
                raise RuntimeError(
                    f"suite {suite_name} has duplicate task id: {task.id}"
                )
            seen_ids.add(task.id)
            fixture = task.metadata.fixture_dir
            if fixture:
                fixture_path = (repo_root / fixture).resolve()
                if not fixture_path.exists():
                    raise RuntimeError(
                        f"suite {suite_name} task {task.id} fixture missing: {fixture}"
                    )
            commands = validation_commands(task)
            if task.metadata.validation is not None and not commands:
                raise RuntimeError(
                    f"suite {suite_name} task {task.id} validation commands are empty"
                )
            if task.run_mode == "offline" and fixture:
                raise RuntimeError(
                    f"suite {suite_name} task {task.id} is offline but declares fixture_dir"
                )
            if (
                task.run_mode == "real"
                and not fixture
                and task.requires_project_mutation()
            ):
                raise RuntimeError(
                    f"suite {suite_name} task {task.id} real task must use fixture_dir"
                )


validate_suite_registry()
