from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

EvalMode = Literal["act", "plan", "build"]
EvalRunMode = Literal["offline", "real", "either"]
EvalDifficulty = Literal["easy", "medium", "hard"]

REPORT_SCHEMA_VERSION = 2
TRACE_SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA_VERSION = 1


class EvalTaskSchemaError(ValueError):
    """Eval task schema 校验失败。"""


@dataclass(frozen=True)
class ValidationSpec:
    """验证命令配置。"""

    commands: tuple[str | tuple[str, ...], ...] = ()
    timeout_seconds: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "commands": self.commands,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class EvidenceSpec:
    """文件证据配置。"""

    files: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"files": self.files}


@dataclass(frozen=True)
class BenchmarkSpec:
    """benchmark 元数据。"""

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class MemoryEvalSpec:
    """memory eval 元数据。"""

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class ToolPolicySpec:
    """工具策略检查元数据。"""

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class FaultInjectionSpec:
    """故障注入任务元数据。"""

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class TaskMetadata:
    """强类型 eval metadata。"""

    fixture_dir: str | None = None
    evidence: EvidenceSpec | None = None
    validation: ValidationSpec | None = None
    benchmark: BenchmarkSpec | None = None
    memory_eval: MemoryEvalSpec | None = None
    tool_policy: ToolPolicySpec | None = None
    fault_injection: FaultInjectionSpec | None = None

    @classmethod
    def from_object(
        cls,
        value: TaskMetadata | Mapping[str, Any] | None,
        *,
        path: str = "metadata",
    ) -> TaskMetadata:
        if value is None:
            return cls()
        if isinstance(value, TaskMetadata):
            return value
        if not isinstance(value, Mapping):
            raise EvalTaskSchemaError(f"{path}: expected object")
        allowed = {
            "fixture_dir",
            "evidence",
            "validation",
            "benchmark",
            "memory_eval",
            "tool_policy",
            "fault_injection",
        }
        _reject_unknown_fields(value, allowed, path=path)
        fixture_dir = _optional_string(value.get("fixture_dir"), path=f"{path}.fixture_dir")
        evidence = _parse_evidence(value.get("evidence"), path=f"{path}.evidence")
        validation = _parse_validation(
            value.get("validation"),
            path=f"{path}.validation",
        )
        benchmark = _parse_passthrough_dict(
            value.get("benchmark"),
            cls_=BenchmarkSpec,
            path=f"{path}.benchmark",
        )
        memory_eval = _parse_passthrough_dict(
            value.get("memory_eval"),
            cls_=MemoryEvalSpec,
            path=f"{path}.memory_eval",
        )
        tool_policy = _parse_passthrough_dict(
            value.get("tool_policy"),
            cls_=ToolPolicySpec,
            path=f"{path}.tool_policy",
        )
        fault_injection = _parse_passthrough_dict(
            value.get("fault_injection"),
            cls_=FaultInjectionSpec,
            path=f"{path}.fault_injection",
        )
        return cls(
            fixture_dir=fixture_dir,
            evidence=evidence,
            validation=validation,
            benchmark=benchmark,
            memory_eval=memory_eval,
            tool_policy=tool_policy,
            fault_injection=fault_injection,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.fixture_dir is not None:
            data["fixture_dir"] = self.fixture_dir
        if self.evidence is not None:
            data["evidence"] = self.evidence.to_dict()
        if self.validation is not None:
            data["validation"] = self.validation.to_dict()
        if self.benchmark is not None:
            data["benchmark"] = self.benchmark.to_dict()
        if self.memory_eval is not None:
            data["memory_eval"] = self.memory_eval.to_dict()
        if self.tool_policy is not None:
            data["tool_policy"] = self.tool_policy.to_dict()
        if self.fault_injection is not None:
            data["fault_injection"] = self.fault_injection.to_dict()
        return data

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.to_dict()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


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
    metadata: TaskMetadata | Mapping[str, Any] | None = None
    llm_judge_criteria: tuple[str, ...] = ()
    llm_judge_required: bool = False
    version: str = "1"
    owner: str = "xcode"
    capability: str = "general"
    expected_duration_seconds: int | None = None
    difficulty: EvalDifficulty = "medium"
    run_mode: EvalRunMode = "offline"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            TaskMetadata.from_object(self.metadata, path="metadata"),
        )

    def requires_project_mutation(self) -> bool:
        """判断任务是否需要直接使用调用方项目根目录。"""
        return self.metadata.fixture_dir is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "mode": self.mode,
            "expected_answer_contains": self.expected_answer_contains,
            "expected_tool_calls": self.expected_tool_calls,
            "disallowed_tool_calls": self.disallowed_tool_calls,
            "max_tool_errors": self.max_tool_errors,
            "tags": self.tags,
            "metadata": self.metadata.to_dict(),
            "llm_judge_criteria": self.llm_judge_criteria,
            "llm_judge_required": self.llm_judge_required,
            "version": self.version,
            "owner": self.owner,
            "capability": self.capability,
            "expected_duration_seconds": self.expected_duration_seconds,
            "difficulty": self.difficulty,
            "run_mode": self.run_mode,
        }


@dataclass(frozen=True)
class GraderResult:
    """单个 grader 的评判结果。"""

    name: str
    passed: bool
    details: str = ""
    skipped: bool = False
    score: float = 1.0
    required: bool = True
    weight: float = 1.0
    evidence: dict[str, Any] | None = None
    failure_category: str | None = None


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
    tasks: tuple[EvalTask, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: int = REPORT_SCHEMA_VERSION


def parse_eval_task(
    item: Mapping[str, Any],
    *,
    path: str = "task",
) -> EvalTask:
    """严格解析外部 EvalTask 定义。"""
    if not isinstance(item, Mapping):
        raise EvalTaskSchemaError(f"{path}: expected object")
    allowed = {
        "id",
        "prompt",
        "mode",
        "expected_answer_contains",
        "expected_tool_calls",
        "disallowed_tool_calls",
        "max_tool_errors",
        "tags",
        "metadata",
        "llm_judge_criteria",
        "llm_judge_required",
        "version",
        "owner",
        "capability",
        "expected_duration_seconds",
        "difficulty",
        "run_mode",
    }
    _reject_unknown_fields(item, allowed, path=path)
    task_id = _required_string(item.get("id"), path=f"{path}.id")
    prompt = _required_string(item.get("prompt"), path=f"{path}.prompt")
    mode = _literal_value(
        item.get("mode", "act"),
        allowed={"act", "plan", "build"},
        path=f"{path}.mode",
    )
    difficulty = _literal_value(
        item.get("difficulty", "medium"),
        allowed={"easy", "medium", "hard"},
        path=f"{path}.difficulty",
    )
    run_mode = _literal_value(
        item.get("run_mode", "offline"),
        allowed={"offline", "real", "either"},
        path=f"{path}.run_mode",
    )
    max_tool_errors = _non_negative_int(
        item.get("max_tool_errors", 0),
        path=f"{path}.max_tool_errors",
    )
    expected_duration_seconds = _optional_non_negative_int(
        item.get("expected_duration_seconds"),
        path=f"{path}.expected_duration_seconds",
    )
    return EvalTask(
        id=task_id,
        prompt=prompt,
        mode=mode,
        expected_answer_contains=_string_tuple(
            item.get("expected_answer_contains", ()),
            path=f"{path}.expected_answer_contains",
        ),
        expected_tool_calls=_string_tuple(
            item.get("expected_tool_calls", ()),
            path=f"{path}.expected_tool_calls",
        ),
        disallowed_tool_calls=_string_tuple(
            item.get("disallowed_tool_calls", ()),
            path=f"{path}.disallowed_tool_calls",
        ),
        max_tool_errors=max_tool_errors,
        tags=_string_tuple(item.get("tags", ()), path=f"{path}.tags"),
        metadata=TaskMetadata.from_object(item.get("metadata"), path=f"{path}.metadata"),
        llm_judge_criteria=_string_tuple(
            item.get("llm_judge_criteria", ()),
            path=f"{path}.llm_judge_criteria",
        ),
        llm_judge_required=_bool_value(
            item.get("llm_judge_required", False),
            path=f"{path}.llm_judge_required",
        ),
        version=_required_string(item.get("version", "1"), path=f"{path}.version"),
        owner=_required_string(item.get("owner", "xcode"), path=f"{path}.owner"),
        capability=_required_string(
            item.get("capability", "general"),
            path=f"{path}.capability",
        ),
        expected_duration_seconds=expected_duration_seconds,
        difficulty=difficulty,
        run_mode=run_mode,
    )


def _parse_evidence(value: object, *, path: str) -> EvidenceSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EvalTaskSchemaError(f"{path}: expected object")
    allowed = {"files"}
    _reject_unknown_fields(value, allowed, path=path)
    files_obj = value.get("files", ())
    if not isinstance(files_obj, list | tuple):
        raise EvalTaskSchemaError(f"{path}.files: expected array")
    files: list[dict[str, Any]] = []
    for index, item in enumerate(files_obj):
        if not isinstance(item, Mapping):
            raise EvalTaskSchemaError(f"{path}.files[{index}]: expected object")
        allowed_item = {"path", "exists", "changed", "contains", "not_contains"}
        _reject_unknown_fields(item, allowed_item, path=f"{path}.files[{index}]")
        rel_path = _required_string(item.get("path"), path=f"{path}.files[{index}].path")
        parsed: dict[str, Any] = {"path": rel_path}
        if "exists" in item:
            if not isinstance(item["exists"], bool):
                raise EvalTaskSchemaError(
                    f"{path}.files[{index}].exists: expected boolean"
                )
            parsed["exists"] = item["exists"]
        if "changed" in item:
            if not isinstance(item["changed"], bool):
                raise EvalTaskSchemaError(
                    f"{path}.files[{index}].changed: expected boolean"
                )
            parsed["changed"] = item["changed"]
        if "contains" in item:
            parsed["contains"] = _string_tuple(
                item["contains"],
                path=f"{path}.files[{index}].contains",
            )
        if "not_contains" in item:
            parsed["not_contains"] = _string_tuple(
                item["not_contains"],
                path=f"{path}.files[{index}].not_contains",
            )
        files.append(parsed)
    return EvidenceSpec(files=tuple(files))


def _parse_validation(value: object, *, path: str) -> ValidationSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EvalTaskSchemaError(f"{path}: expected object")
    allowed = {"commands", "timeout_seconds"}
    _reject_unknown_fields(value, allowed, path=path)
    commands_obj = value.get("commands", ())
    if not isinstance(commands_obj, list | tuple):
        raise EvalTaskSchemaError(f"{path}.commands: expected array")
    commands: list[str | tuple[str, ...]] = []
    for index, command in enumerate(commands_obj):
        cmd_path = f"{path}.commands[{index}]"
        if isinstance(command, str):
            text = command.strip()
            if not text:
                raise EvalTaskSchemaError(f"{cmd_path}: command cannot be empty")
            commands.append(text)
            continue
        if isinstance(command, list | tuple):
            parts = [str(part).strip() for part in command]
            if not parts or any(not part for part in parts):
                raise EvalTaskSchemaError(f"{cmd_path}: argv entries cannot be empty")
            commands.append(tuple(parts))
            continue
        raise EvalTaskSchemaError(f"{cmd_path}: expected string or argv array")
    timeout_seconds = value.get("timeout_seconds", 60.0)
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise EvalTaskSchemaError(
            f"{path}.timeout_seconds: expected number"
        ) from exc
    if timeout <= 0:
        raise EvalTaskSchemaError(f"{path}.timeout_seconds: must be > 0")
    return ValidationSpec(commands=tuple(commands), timeout_seconds=timeout)


def _parse_passthrough_dict(
    value: object,
    *,
    cls_: (
        type[BenchmarkSpec]
        | type[MemoryEvalSpec]
        | type[ToolPolicySpec]
        | type[FaultInjectionSpec]
    ),
    path: str,
) -> BenchmarkSpec | MemoryEvalSpec | ToolPolicySpec | FaultInjectionSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EvalTaskSchemaError(f"{path}: expected object")
    return cls_(data=dict(value))


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    *,
    path: str,
) -> None:
    for key in value:
        if key not in allowed:
            raise EvalTaskSchemaError(f"{path}.{key}: unknown field")


def _required_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalTaskSchemaError(f"{path}: expected non-empty string")
    return value.strip()


def _optional_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvalTaskSchemaError(f"{path}: expected non-empty string")
    return value.strip()


def _string_tuple(value: object, *, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise EvalTaskSchemaError(f"{path}: expected non-empty string")
        return (text,)
    if not isinstance(value, list | tuple):
        raise EvalTaskSchemaError(f"{path}: expected array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise EvalTaskSchemaError(f"{path}[{index}]: expected non-empty string")
        result.append(item.strip())
    return tuple(result)


def _literal_value(
    value: object,
    *,
    allowed: set[str],
    path: str,
) -> str:
    if not isinstance(value, str) or value not in allowed:
        values = ", ".join(sorted(allowed))
        raise EvalTaskSchemaError(f"{path}: expected one of {values}")
    return value


def _non_negative_int(value: object, *, path: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise EvalTaskSchemaError(f"{path}: expected integer >= 0")
    return value


def _optional_non_negative_int(value: object, *, path: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, path=path)


def _bool_value(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise EvalTaskSchemaError(f"{path}: expected boolean")
    return value
