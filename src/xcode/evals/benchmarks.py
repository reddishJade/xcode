from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Literal

from .schema import EvalTask

BenchmarkName = Literal[
    "humaneval",
    "swebench-lite",
    "evalplus-humaneval",
    "evalplus-mbpp",
]


def load_benchmark(
    name: str,
    path: Path,
    fixture_root: Path | None = None,
) -> tuple[EvalTask, ...]:
    """从本地 benchmark 数据文件加载 eval 任务。"""
    if name == "humaneval":
        return load_humaneval(path)
    if name == "swebench-lite":
        return load_swebench_lite(path)
    if name == "evalplus-humaneval":
        return load_evalplus_humaneval(path, fixture_root=fixture_root)
    if name == "evalplus-mbpp":
        return load_evalplus_mbpp(path, fixture_root=fixture_root)
    raise ValueError(f"unsupported benchmark: {name}")


def load_humaneval(path: Path) -> tuple[EvalTask, ...]:
    """加载 HumanEval JSON/JSONL 数据为代码补全任务。"""
    tasks: list[EvalTask] = []
    for item in _load_items(path):
        task_id = str(item.get("task_id") or item.get("id") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not task_id or not prompt:
            continue
        entry_point = str(item.get("entry_point") or "").strip()
        canonical = str(item.get("canonical_solution") or "").strip()
        tests = str(item.get("test") or "").strip()
        criteria = [
            "最终回答包含可直接用于补全的 Python 代码。",
            "实现符合题目描述的函数签名和行为。",
        ]
        if tests:
            criteria.append("实现应能通过 benchmark 提供的测试断言。")
        tasks.append(
            EvalTask(
                id=_normalize_task_id("humaneval", task_id),
                prompt=_humaneval_prompt(prompt, entry_point, tests),
                mode="act",
                expected_answer_contains=(entry_point,) if entry_point else (),
                tags=("benchmark", "humaneval", "coding"),
                metadata={
                    "benchmark": {
                        "name": "humaneval",
                        "task_id": task_id,
                        "entry_point": entry_point,
                        "canonical_solution": canonical,
                        "test": tests,
                    }
                },
                llm_judge_criteria=tuple(criteria),
            )
        )
    return tuple(tasks)


def load_swebench_lite(path: Path) -> tuple[EvalTask, ...]:
    """加载 SWE-bench Lite JSON/JSONL 数据为任务级修复任务。"""
    tasks: list[EvalTask] = []
    for item in _load_items(path):
        instance_id = str(item.get("instance_id") or item.get("id") or "").strip()
        problem = str(item.get("problem_statement") or item.get("prompt") or "").strip()
        if not instance_id or not problem:
            continue
        repo = str(item.get("repo") or "").strip()
        base_commit = str(item.get("base_commit") or "").strip()
        tests = str(item.get("test_patch") or "").strip()
        tasks.append(
            EvalTask(
                id=_normalize_task_id("swebench-lite", instance_id),
                prompt=_swebench_prompt(problem, repo, base_commit, tests),
                mode="act",
                tags=("benchmark", "swebench-lite", "coding", "repair"),
                metadata={
                    "benchmark": {
                        "name": "swebench-lite",
                        "instance_id": instance_id,
                        "repo": repo,
                        "base_commit": base_commit,
                        "test_patch": tests,
                    }
                },
                llm_judge_criteria=(
                    "修改应直接解决问题陈述中的缺陷。",
                    "实现应保持补丁范围集中，避免无关重构。",
                    "实现应满足 benchmark 附带的测试意图。",
                ),
            )
        )
    return tuple(tasks)


def load_evalplus_humaneval(
    path: Path,
    fixture_root: Path | None = None,
) -> tuple[EvalTask, ...]:
    """加载 HumanEval+ 风格数据为 sandbox 编码任务。"""
    root = _benchmark_fixture_root(path, fixture_root, "evalplus-humaneval")
    tasks: list[EvalTask] = []
    for item in _load_items(path):
        task_id = str(item.get("task_id") or item.get("id") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        entry_point = str(item.get("entry_point") or "").strip()
        tests = _benchmark_tests(item)
        if not task_id or not prompt or not entry_point or not tests:
            continue
        fixture_dir = _write_evalplus_fixture(
            root=root,
            benchmark_name="evalplus-humaneval",
            raw_id=task_id,
            prompt=prompt,
            entry_point=entry_point,
            tests=tests,
        )
        tasks.append(
            _evalplus_task(
                benchmark_name="evalplus-humaneval",
                raw_id=task_id,
                prompt=prompt,
                entry_point=entry_point,
                fixture_dir=fixture_dir,
                tests=tests,
            )
        )
    return tuple(tasks)


def load_evalplus_mbpp(
    path: Path,
    fixture_root: Path | None = None,
) -> tuple[EvalTask, ...]:
    """加载 MBPP+ 风格数据为 sandbox 编码任务。"""
    root = _benchmark_fixture_root(path, fixture_root, "evalplus-mbpp")
    tasks: list[EvalTask] = []
    for item in _load_items(path):
        task_id = str(item.get("task_id") or item.get("id") or "").strip()
        prompt = str(item.get("prompt") or item.get("text") or "").strip()
        tests = _benchmark_tests(item)
        entry_point = str(item.get("entry_point") or "").strip()
        if not entry_point:
            entry_point = _infer_entry_point(tests)
        if not task_id or not prompt or not entry_point or not tests:
            continue
        fixture_dir = _write_evalplus_fixture(
            root=root,
            benchmark_name="evalplus-mbpp",
            raw_id=task_id,
            prompt=prompt,
            entry_point=entry_point,
            tests=tests,
        )
        tasks.append(
            _evalplus_task(
                benchmark_name="evalplus-mbpp",
                raw_id=task_id,
                prompt=prompt,
                entry_point=entry_point,
                fixture_dir=fixture_dir,
                tests=tests,
            )
        )
    return tuple(tasks)


def _load_items(path: Path) -> tuple[dict[str, Any], ...]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ()
    if path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        raw_items = raw if isinstance(raw, list) else [raw]
    return tuple(item for item in raw_items if isinstance(item, dict))


def _benchmark_fixture_root(
    path: Path,
    fixture_root: Path | None,
    benchmark_name: str,
) -> Path:
    root = fixture_root or path.parent / ".xcode_eval_fixtures"
    return root / benchmark_name


def _benchmark_tests(item: dict[str, Any]) -> str:
    tests = item.get("test")
    if isinstance(tests, str) and tests.strip():
        return tests.strip()
    test_list = item.get("test_list")
    if isinstance(test_list, list | tuple):
        lines = [str(test).strip() for test in test_list if str(test).strip()]
        return "\n".join(lines)
    return ""


def _write_evalplus_fixture(
    *,
    root: Path,
    benchmark_name: str,
    raw_id: str,
    prompt: str,
    entry_point: str,
    tests: str,
) -> Path:
    fixture_dir = root / _normalize_task_id(benchmark_name, raw_id)
    tests_dir = fixture_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "solution.py").write_text(
        _solution_template(prompt),
        encoding="utf-8",
    )
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_solution.py").write_text(
        _evalplus_test_file(entry_point, tests),
        encoding="utf-8",
    )
    return fixture_dir.resolve()


def _solution_template(prompt: str) -> str:
    return f'"""{prompt}"""\n\n# 在下方实现要求的函数。\n'


def _evalplus_test_file(entry_point: str, tests: str) -> str:
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import unittest",
            "",
            "from solution import *",
            "",
            tests,
            "",
            "",
            "class EvalPlusTests(unittest.TestCase):",
            "    def test_benchmark(self) -> None:",
            "        check_fn = globals().get('check')",
            "        if check_fn is not None:",
            f"            check_fn({entry_point})",
            "",
            "",
            "if __name__ == '__main__':",
            "    unittest.main()",
            "",
        ]
    )


def _infer_entry_point(tests: str) -> str:
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", tests)
    return match.group(1) if match else ""


def _evalplus_task(
    *,
    benchmark_name: str,
    raw_id: str,
    prompt: str,
    entry_point: str,
    fixture_dir: Path,
    tests: str,
) -> EvalTask:
    task_id = _normalize_task_id(benchmark_name, raw_id)
    return EvalTask(
        id=task_id,
        prompt=_evalplus_prompt(prompt, entry_point),
        mode="act",
        expected_tool_calls=("read_file", "bash"),
        tags=("benchmark", benchmark_name, "coding", "function"),
        metadata={
            "fixture_dir": str(fixture_dir),
            "validation": {
                "commands": ((sys.executable, "-m", "unittest", "discover", "tests"),),
                "timeout_seconds": 30,
            },
            "evidence": {
                "files": [
                    {
                        "path": "solution.py",
                        "changed": True,
                        "contains": (entry_point,),
                    }
                ],
            },
            "benchmark": {
                "name": benchmark_name,
                "task_id": raw_id,
                "entry_point": entry_point,
                "test": tests,
            },
        },
    )


def _evalplus_prompt(prompt: str, entry_point: str) -> str:
    return "\n".join(
        [
            "Edit solution.py to implement the requested Python function.",
            "Run the unit tests with python -m unittest discover tests.",
            f"Entry point: {entry_point}",
            "",
            prompt,
        ]
    )


def _humaneval_prompt(prompt: str, entry_point: str, tests: str) -> str:
    parts = [
        "Complete the following Python function. Return the final code only.",
        "",
        prompt,
    ]
    if entry_point:
        parts.extend(["", f"Entry point: {entry_point}"])
    if tests:
        parts.extend(["", "Reference tests:", tests])
    return "\n".join(parts)


def _swebench_prompt(
    problem: str,
    repo: str,
    base_commit: str,
    tests: str,
) -> str:
    parts = [
        "Fix the repository issue described below. Make a targeted code change.",
        "",
    ]
    if repo:
        parts.append(f"Repository: {repo}")
    if base_commit:
        parts.append(f"Base commit: {base_commit}")
    parts.extend(["", "Problem statement:", problem])
    if tests:
        parts.extend(["", "Test patch:", tests])
    return "\n".join(parts)


def _normalize_task_id(prefix: str, raw_id: str) -> str:
    normalized = raw_id.replace("/", "-").replace(" ", "-")
    return f"{prefix}-{normalized}"
