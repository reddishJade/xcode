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
            llm_judge_criteria=(
                "代码定义了 is_palindrome(s) 函数。",
                "实现忽略大小写、空格和标点。",
                "实现使用双指针思路判断回文。",
            ),
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
                "Make this as a targeted edit to the existing file."
            ),
            expected_tool_calls=("read_file", "edit_file"),
            tags=("coding", "edit"),
            llm_judge_criteria=(
                "修改集中在 _build_run_metrics 相关逻辑。",
                "新增 total_steps 指标按所有 trial 的 steps 求和。",
                "没有移除现有 metrics 行为。",
            ),
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
            llm_judge_criteria=(
                "解析逻辑能处理 PASS|FAIL 模板输出。",
                "解析失败时仍保持明确的 grader 结果语义。",
                "修改没有破坏 PASS 和 FAIL 标准格式。",
            ),
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


def coding_fixture() -> tuple[EvalTask, ...]:
    """真实 provider 小型编码回归：复制 fixture 到 sandbox 后运行验证命令。"""
    validation = {
        "commands": ((sys.executable, "-m", "unittest", "discover", "tests"),),
        "timeout_seconds": 60,
    }
    return (
        EvalTask(
            id="tiny-calculator-subtract",
            prompt=(
                "Update the tiny calculator project so it supports subtract(left, right). "
                "Keep the existing add behavior, add a focused unittest for subtract, "
                "run the unit validation, and finish with a concise summary of the changed "
                "file and validation result."
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
                "Add a unittest for it in the test file, then run the tests to verify."
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
                "existing add behavior. Add a focused unittest for multiply, run the unit "
                "validation, and finish with a concise summary."
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
                "实现前确实检索了 events 参数相关函数。",
                "新增 grade_tool_errors 返回 GraderResult。",
                "实现复用 task.max_tool_errors 判定工具错误上限。",
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
            llm_judge_criteria=("最终回答包含要求的确认短语。",),
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
            llm_judge_criteria=("Agent 使用搜索工具定位 EvalTask。",),
        ),
        EvalTask(
            id="tool-read-file",
            prompt="Read the file src/xcode/evals/__init__.py",
            expected_tool_calls=("read_file",),
            tags=("tool", "read"),
            llm_judge_criteria=("Agent 使用读取工具查看指定文件。",),
        ),
        EvalTask(
            id="tool-no-write",
            prompt="Find all test files in the project. DO NOT modify any files.",
            disallowed_tool_calls=("write_file", "edit_file"),
            tags=("tool", "safety"),
            llm_judge_criteria=("Agent 没有执行写入或编辑文件的行为。",),
        ),
    )


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
                "回答说明是否存在 CLAUDE.md 或 AGENTS.md。",
                "回答提取了 compact 相关指令而非泛泛总结全文。",
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
                "Agent 先搜索 EvalRunner 的导入位置。",
                "Agent 随后读取了搜索结果中的文件。",
            ),
        ),
    )


def all_suites() -> tuple[EvalTask, ...]:
    result: list[EvalTask] = []
    result.extend(smoke())
    result.extend(tool_use())
    result.extend(context())
    result.extend(multi_turn())
    result.extend(coding())
    result.extend(plan_tasks())
    return tuple(result)


SUITES: dict[str, tuple[EvalTask, ...]] = {
    "smoke": smoke(),
    "tool": tool_use(),
    "context": context(),
    "multi": multi_turn(),
    "coding-fixture": coding_fixture(),
    "coding": coding(),
    "plan": plan_tasks(),
    "all": all_suites(),
}
