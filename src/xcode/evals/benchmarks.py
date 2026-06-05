from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .schema import EvalTask

BenchmarkName = Literal["humaneval", "swebench-lite"]


def load_benchmark(name: str, path: Path) -> tuple[EvalTask, ...]:
    """从本地 benchmark 数据文件加载 eval 任务。"""
    if name == "humaneval":
        return load_humaneval(path)
    if name == "swebench-lite":
        return load_swebench_lite(path)
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
