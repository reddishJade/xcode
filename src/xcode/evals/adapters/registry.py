from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BenchmarkStatus = Literal["integrated", "export-only", "catalog-only"]


@dataclass(frozen=True)
class BenchmarkAdapterSpec:
    """外部 benchmark adapter 的声明信息。"""

    name: str
    display_name: str
    purpose: str
    harness: str
    xcode_role: str
    upstream_url: str
    status: BenchmarkStatus


BENCHMARK_ADAPTERS: dict[str, BenchmarkAdapterSpec] = {
    "humaneval": BenchmarkAdapterSpec(
        name="humaneval",
        display_name="HumanEval",
        purpose="Python single-function completion benchmark.",
        harness="Prompt-only benchmark loader with LLM/outcome grading",
        xcode_role="Load completion tasks and score final answers with deterministic and judge graders.",
        upstream_url="https://github.com/openai/human-eval",
        status="integrated",
    ),
    "evalplus-humaneval": BenchmarkAdapterSpec(
        name="evalplus-humaneval",
        display_name="EvalPlus HumanEval+",
        purpose="Python single-function generation benchmark.",
        harness="EvalPlus-style pytest fixture",
        xcode_role="Edit solution.py in sandbox; validation command runs tests.",
        upstream_url="https://evalplus.github.io/",
        status="integrated",
    ),
    "evalplus-mbpp": BenchmarkAdapterSpec(
        name="evalplus-mbpp",
        display_name="EvalPlus MBPP+",
        purpose="Python beginner programming task benchmark.",
        harness="EvalPlus-style pytest fixture",
        xcode_role="Edit solution.py in sandbox; validation command runs tests.",
        upstream_url="https://evalplus.github.io/",
        status="integrated",
    ),
    "swebench-lite": BenchmarkAdapterSpec(
        name="swebench-lite",
        display_name="SWE-bench Lite",
        purpose="Repository-level bugfix capability benchmark.",
        harness="SWE-bench official harness",
        xcode_role="Generate candidate patches; harness checks out repo, runs tests, and scores.",
        upstream_url="https://www.swebench.com/SWE-bench/",
        status="integrated",
    ),
    "swebench-verified": BenchmarkAdapterSpec(
        name="swebench-verified",
        display_name="SWE-bench Verified",
        purpose="Expert-verified repository-level bugfix benchmark.",
        harness="SWE-bench official harness",
        xcode_role="Generate candidate patches; harness checks out repo, runs tests, and scores.",
        upstream_url="https://www.swebench.com/SWE-bench/",
        status="export-only",
    ),
    "terminal-bench": BenchmarkAdapterSpec(
        name="terminal-bench",
        display_name="Terminal-Bench",
        purpose="Terminal tasks, shell debugging, and engineering operations benchmark.",
        harness="Terminal-Bench harness",
        xcode_role="Execute tasks as a terminal agent; harness produces scores.",
        upstream_url="https://terminalbench.lol/",
        status="catalog-only",
    ),
    "aider-polyglot": BenchmarkAdapterSpec(
        name="aider-polyglot",
        display_name="Aider Polyglot",
        purpose="Multilingual code editing benchmark.",
        harness="Aider Polyglot benchmark",
        xcode_role="Edit code in the task workspace; language test commands produce scores.",
        upstream_url="https://aider.chat/docs/leaderboards/",
        status="catalog-only",
    ),
}

INTEGRATED_BENCHMARKS: tuple[str, ...] = tuple(
    sorted(name for name, spec in BENCHMARK_ADAPTERS.items() if spec.status == "integrated")
)
