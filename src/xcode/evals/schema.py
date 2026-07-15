"""Xcode Eval 的最小领域模型。

本模块只描述真实能力评测的数据边界，不依赖 provider、agent pipeline 或报告事件。
VerifierSpec 故意不属于 Task，防止隐藏判分材料进入 Agent 可见序列化结果。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = 1
Identifier = Annotated[
    str, Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
]
PositiveInt = Annotated[int, Field(gt=0)]


class EvalModel(BaseModel):
    """所有 Eval 契约采用严格字段和不可变值。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorCategory(StrEnum):
    """Trial 未形成有效能力结果时的互斥分类。"""

    AGENT_FAILURE = "agent_failure"
    PROVIDER_FAILURE = "provider_failure"
    TOOL_FAILURE = "tool_failure"
    BUDGET_EXCEEDED = "budget_exceeded"
    VERIFIER_FAILURE = "verifier_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    INVALID_TASK = "invalid_task"


class TaskSource(EvalModel):
    """任务来源、许可和精确初始版本。"""

    kind: str
    repository: str
    revision: str
    license: str
    upstream_id: str | None = None


class ResourceBudget(EvalModel):
    """Agent 运行前声明的资源上限。"""

    wall_time_seconds: PositiveInt
    model_calls: PositiveInt
    tool_calls: PositiveInt
    input_tokens: PositiveInt | None = None
    output_tokens: PositiveInt | None = None


class Task(EvalModel):
    """可提供给 Agent 的版本化任务描述。

    verifier_id 是控制面解析用的不透明标识，不包含命令、隐藏路径或参考答案。
    """

    schema_version: int = SCHEMA_VERSION
    task_id: Identifier
    dataset_version: Identifier
    prompt: Annotated[str, Field(min_length=1)]
    source: TaskSource
    verifier_id: Identifier
    allowed_paths: tuple[str, ...]
    ignored_paths: tuple[str, ...] = ()
    guides: tuple[str, ...] = ()
    sensors: tuple[str, ...] = ()
    tags: tuple[Identifier, ...] = ()
    budget: ResourceBudget
    known_limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_agent_visible_paths(self) -> Self:
        """拒绝绝对路径、父目录穿越和空修改范围。"""
        if not self.allowed_paths:
            raise ValueError("allowed_paths must not be empty")
        for value in (*self.allowed_paths, *self.ignored_paths, *self.guides):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"agent-visible path must be workspace-relative: {value}"
                )
        return self


class Variant(EvalModel):
    """可审计的 harness 配置快照。"""

    variant_id: Identifier
    harness_revision: str
    capabilities: dict[str, bool]
    config: dict[str, JsonValue] = Field(default_factory=dict)


class ModelConfig(EvalModel):
    """影响模型输出的受控实验变量。"""

    provider: str
    model: str
    temperature: float | None = None
    seed: int | None = None
    options: dict[str, JsonValue] = Field(default_factory=dict)


class Trial(EvalModel):
    """Task、Variant、模型、预算和重复序号的运行声明。"""

    schema_version: int = SCHEMA_VERSION
    trial_id: Identifier
    experiment_id: Identifier
    task_id: Identifier
    dataset_version: Identifier
    variant: Variant
    model: ModelConfig
    budget: ResourceBudget
    repetition: Annotated[int, Field(ge=0)]
    workspace_revision: str
    command: tuple[str, ...]


class Experiment(EvalModel):
    """一组保持任务、模型、预算和重复策略可比较的 Trial 声明。"""

    schema_version: int = SCHEMA_VERSION
    experiment_id: Identifier
    dataset_version: Identifier
    task_ids: tuple[Identifier, ...]
    variants: tuple[Variant, ...]
    model: ModelConfig
    repetitions: PositiveInt
    command: tuple[str, ...]

    @model_validator(mode="after")
    def validate_comparison_axes(self) -> Self:
        """拒绝空任务、空命令和会破坏配对关系的重复标识。"""
        if not self.task_ids:
            raise ValueError("experiment must select at least one task")
        if not self.variants:
            raise ValueError("experiment must declare at least one variant")
        if not self.command:
            raise ValueError("experiment command must not be empty")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("experiment task_ids must be unique")
        variant_ids = [variant.variant_id for variant in self.variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("experiment variant ids must be unique")
        return self


class VerifierSpec(EvalModel):
    """仅 Eval 控制面可读取的隐藏 verifier 定义。"""

    verifier_id: Identifier
    version: Identifier
    command: tuple[str, ...]
    hidden_root: str
    result_file: str = "verifier-result.json"
    timeout_seconds: PositiveInt

    @model_validator(mode="after")
    def validate_command(self) -> Self:
        """Verifier 必须拥有实际命令且隐藏根目录不能是工作区相对路径。"""
        if not self.command:
            raise ValueError("verifier command must not be empty")
        if not PurePosixPath(self.hidden_root).is_absolute():
            raise ValueError("hidden_root must be outside the relative Agent workspace")
        result_path = PurePosixPath(self.result_file)
        if result_path.is_absolute() or ".." in result_path.parts:
            raise ValueError("result_file must stay inside hidden_root")
        return self


class ResourceUsage(EvalModel):
    """Trial 的未筛选原始资源消耗。"""

    wall_time_seconds: Annotated[float, Field(ge=0)]
    model_calls: Annotated[int, Field(ge=0)]
    tool_calls: Annotated[int, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None


class VerifierResult(EvalModel):
    """Agent 结束后由独立边界产生的四项判分。"""

    verifier_id: Identifier
    verifier_version: Identifier
    completed: bool
    resolved: bool = False
    regression_free: bool = False
    policy_clean: bool = False
    log_artifact: str
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_scores_from_failed_verifier(self) -> Self:
        """未完成的 verifier 不能携带伪造的通过项。"""
        if not self.completed and (
            self.resolved or self.regression_free or self.policy_clean
        ):
            raise ValueError("incomplete verifier cannot report passing checks")
        return self


class ArtifactManifest(EvalModel):
    """离线重建一个 Trial 所需的证据索引。"""

    task: str
    trial: str
    trace: str
    patch: str
    stdout: str
    stderr: str
    verifier_log: str
    result: str
    environment: str
    checksums: str = "checksums.json"


class TrialResult(EvalModel):
    """能力结果与基础设施有效性分离后的最终记录。"""

    schema_version: int = SCHEMA_VERSION
    trial_id: Identifier
    started_at: datetime
    finished_at: datetime
    agent_completed: bool
    valid_trial: bool
    verifier: VerifierResult | None
    error_category: ErrorCategory | None = None
    error_message: str | None = None
    termination_reason: str
    usage: ResourceUsage
    artifacts: ArtifactManifest

    @property
    def success(self) -> bool:
        """成功必须同时满足有效、完成、目标、回归和约束条件。"""
        verifier = self.verifier
        return bool(
            self.valid_trial
            and self.agent_completed
            and verifier is not None
            and verifier.completed
            and verifier.resolved
            and verifier.regression_free
            and verifier.policy_clean
        )

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """阻止无效 Trial 混入成功率分母或丢失排除原因。"""
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.valid_trial:
            if self.error_category is not None:
                raise ValueError("valid trial cannot have an error category")
            if self.verifier is None or not self.verifier.completed:
                raise ValueError("valid trial requires a completed verifier")
        elif self.error_category is None:
            raise ValueError("invalid trial requires an error category")
        return self


class TrialMetric(EvalModel):
    """报告中的单 Trial 结果与投入联合点。"""

    trial_id: Identifier
    task_id: Identifier
    variant_id: Identifier
    repetition: Annotated[int, Field(ge=0)]
    valid_trial: bool
    success: bool
    error_category: ErrorCategory | None
    resolved: bool | None
    regression_free: bool | None
    policy_clean: bool | None
    usage: ResourceUsage


class UsageAggregate(EvalModel):
    """包含失败和无效 Trial 的总投入与成功单位成本。"""

    wall_time_seconds: float
    model_calls: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    tokens_per_success: float | None
    tool_calls_per_success: float | None
    time_per_success: float | None
    cost_per_success: float | None


class VariantSummary(EvalModel):
    """单 Variant 的结果、排除项、重复统计和成本。"""

    variant_id: Identifier
    declared_trials: int
    observed_trials: int
    missing_trials: int
    valid_trials: int
    excluded_trials: int
    successes: int
    success_rate: float | None
    pass_k: PositiveInt
    pass_at_k: float | None
    pass_power_k: float | None
    pass_k_eligible_tasks: int
    resolved_rate: float | None
    regression_free_rate: float | None
    policy_clean_rate: float | None
    exclusions: dict[ErrorCategory, int]
    usage: UsageAggregate


class VariantComparison(EvalModel):
    """同 task/repetition 上两个 Variant 的配对结果与成本差。"""

    candidate_variant_id: Identifier
    control_variant_id: Identifier
    declared_pairs: int
    observed_pairs: int
    missing_pairs: int
    valid_pairs: int
    invalid_pairs: int
    candidate_successes: int
    control_successes: int
    candidate_wins: int
    control_wins: int
    ties: int
    harness_gain: float | None
    input_tokens_delta: int | None
    tool_calls_delta: int
    wall_time_seconds_delta: float


class ExperimentSummary(EvalModel):
    """可以由 Trial artifact 完全离线重建的 Experiment 摘要。"""

    schema_version: int = SCHEMA_VERSION
    experiment_id: Identifier
    dataset_version: Identifier
    task_ids: tuple[Identifier, ...]
    repetitions: PositiveInt
    variants: tuple[VariantSummary, ...]
    comparisons: tuple[VariantComparison, ...]
    efficient_variant_ids: tuple[Identifier, ...]
    trials: tuple[TrialMetric, ...]
    formulas: dict[str, str]
