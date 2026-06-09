from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

EvalMode = Literal["act", "plan", "review"]


@dataclass(frozen=True)
class EvalTask:
    """单个 eval 用例的声明式规格。"""

    id: str
    prompt: str
    mode: EvalMode = "act"
    expected_answer_contains: tuple[str, ...] = ()
    expected_tool_calls: tuple[str, ...] = ()
    disallowed_tool_calls: tuple[str, ...] = ()
    max_tool_errors: int = 0
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    # LLM-as-judge 评判标准：每条是一个描述性句子，如 "代码编译通过" 或 "方案考虑了边界情况"
    llm_judge_criteria: tuple[str, ...] = ()

    def requires_project_mutation(self) -> bool:
        """判断任务是否需要直接使用调用方项目根目录。"""
        return "fixture_dir" not in self.metadata


@dataclass(frozen=True)
class GraderResult:
    name: str
    passed: bool
    details: str = ""


@dataclass(frozen=True)
class TrialResult:
    task_id: str
    trial_id: str
    success: bool
    answer: str
    trace_path: Path
    graders: tuple[GraderResult, ...]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    success: bool
    output_dir: Path
    trials: tuple[TrialResult, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
