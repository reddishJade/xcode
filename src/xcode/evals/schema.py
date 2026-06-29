from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

EvalMode = Literal["act", "plan", "build"]
EvalRunMode = Literal["offline", "real", "either"]
EvalDifficulty = Literal["easy", "medium", "hard"]

REPORT_SCHEMA_VERSION = 2
TRACE_SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA_VERSION = 1


class EvalTaskSchemaError(ValueError):
    """Eval task schema 校验失败。"""


class ValidationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commands: tuple[str | tuple[str, ...], ...] = ()
    timeout_seconds: float = 60.0


class EvidenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: tuple[dict[str, Any], ...] = ()


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: dict[str, Any] = Field(default_factory=dict)


class MemoryEvalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: dict[str, Any] = Field(default_factory=dict)


class ToolPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: dict[str, Any] = Field(default_factory=dict)


class FaultInjectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: dict[str, Any] = Field(default_factory=dict)


_SENTINEL = object()


class TaskMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_dir: str | None = None
    evidence: EvidenceSpec | None = None
    validation: ValidationSpec | None = None
    benchmark: BenchmarkSpec | None = None
    memory_eval: MemoryEvalSpec | None = None
    tool_policy: ToolPolicySpec | None = None
    fault_injection: FaultInjectionSpec | None = None

    @field_validator(
        "benchmark", "memory_eval", "tool_policy", "fault_injection", mode="before"
    )
    @classmethod
    def _wrap_passthrough(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {"data": v}
        return v

    def get(self, key: str, default: Any = None) -> Any:
        value = getattr(self, key, _SENTINEL)
        if value is _SENTINEL:
            return default
        if value is None:
            return default
        if isinstance(
            value, (BenchmarkSpec, MemoryEvalSpec, ToolPolicySpec, FaultInjectionSpec)
        ):
            return dict(value.data)
        if isinstance(value, BaseModel):
            return value.model_dump(exclude_none=True)
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        value = getattr(self, key, _SENTINEL)
        if value is _SENTINEL:
            return False
        return value is not None

    def __getitem__(self, key: str) -> Any:
        value = getattr(self, key, _SENTINEL)
        if value is _SENTINEL:
            raise KeyError(key)
        if value is None:
            raise KeyError(key)
        if isinstance(
            value, (BenchmarkSpec, MemoryEvalSpec, ToolPolicySpec, FaultInjectionSpec)
        ):
            return dict(value.data)
        if isinstance(value, BaseModel):
            return value.model_dump(exclude_none=True)
        return value


class EvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    mode: EvalMode = "act"
    expected_answer_contains: tuple[str, ...] = ()
    expected_tool_calls: tuple[str, ...] = ()
    disallowed_tool_calls: tuple[str, ...] = ()
    max_tool_errors: int = 0
    tags: tuple[str, ...] = ()
    metadata: TaskMetadata | dict[str, Any] = Field(default_factory=TaskMetadata)
    llm_judge_criteria: tuple[str, ...] = ()
    llm_judge_required: StrictBool = False
    version: str = "1"
    owner: str = "xcode"
    capability: str = "general"
    expected_duration_seconds: int | None = None
    difficulty: EvalDifficulty = "medium"
    run_mode: EvalRunMode = "offline"

    @field_validator("metadata", mode="before")
    @classmethod
    def coerce_metadata(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return TaskMetadata.model_validate(v)
        if v is None:
            return TaskMetadata()
        return v

    def requires_project_mutation(self) -> bool:
        """判断任务是否需要直接使用调用方项目根目录。"""
        meta = self.metadata
        if isinstance(meta, dict):
            return True
        return meta.fixture_dir is None


class GraderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    details: str = ""
    skipped: bool = False
    score: float = 1.0
    required: bool = True
    weight: float = 1.0
    evidence: dict[str, Any] | None = None
    failure_category: str | None = None


class TrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    trial_id: str
    success: bool
    answer: str
    trace_path: Path
    graders: tuple[GraderResult, ...]
    metrics: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    success: bool
    output_dir: Path
    trials: tuple[TrialResult, ...]
    tasks: tuple[EvalTask, ...] = ()
    metrics: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = REPORT_SCHEMA_VERSION
