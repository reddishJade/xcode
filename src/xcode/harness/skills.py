from __future__ import annotations


from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

from ..agent.types import ToolExecutionMode
from .observability import (
    HITLResult,
    PermissionPolicy,
    check_tool_permission,
    redact_text,
)

"""工具注册表与 HITL 执行门禁。

`ToolSpec` 描述工具能力，dispatch map 根据工具名找到 handler。HITL 在执行
handler 前根据 risk 字段和 permission policy 决定是否需要 approval callback。
"""

ToolInput = dict[str, Any]
ActionHandler = Callable[[ToolInput], str]
RiskEvaluator = Callable[[ToolInput], str]
ApprovalCallback = Callable[["ToolSpec", ToolInput], HITLResult]


@dataclass(frozen=True)
class ToolSpec:
    """工具的可复用描述。

    name/description/input_hint 进入 prompt，handler 负责执行，risk 决定是否
    需要人工确认。
    """

    name: str
    description: str
    input_hint: str
    handler: ActionHandler
    risk: str = "low"
    schema: dict[str, Any] | None = None
    read_only: bool = False
    concurrency_safe: bool = False
    risk_evaluator: RiskEvaluator | None = None
    group: str = "core"
    execution_mode: ToolExecutionMode | None = None
    counts_as_progress: bool | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)


ToolExecutionStatus = Literal["ok", "denied", "error", "approval_required"]

STATUS_OK: ToolExecutionStatus = "ok"
STATUS_DENIED: ToolExecutionStatus = "denied"
STATUS_ERROR: ToolExecutionStatus = "error"
STATUS_APPROVAL_REQUIRED: ToolExecutionStatus = "approval_required"

RISK_LOW = "low"
RISK_HIGH = "high"


@dataclass(frozen=True)
class ToolExecutionResult:
    status: ToolExecutionStatus
    content: str
    metadata: dict[str, Any] | None = None


def resolve_project_path(project_root: Path, raw_path: str) -> Path:
    relative_path = Path(raw_path.strip().strip("\"'") or ".")
    if relative_path.is_absolute():
        raise ValueError("absolute paths are not allowed")
    if ".." in relative_path.parts:
        raise ValueError("parent-directory paths are not allowed")

    root = project_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes project root")
    return candidate


def build_tool_prompt(registry: tuple[ToolSpec, ...]) -> str:
    lines = []
    for index, tool in enumerate(registry, 1):
        lines.extend(
            [
                f"{index}. {tool.name}",
                f"   Description: {tool.description}",
                f"   Action Input: {tool.input_hint}",
                f"   Risk: {tool.risk}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def run_tool(
    registry: dict[str, ToolSpec],
    action: str,
    action_input: ToolInput,
    approval_callback: ApprovalCallback | None = None,
    permission_policy: PermissionPolicy | None = None,
) -> str:
    return run_tool_result(
        registry,
        action,
        action_input,
        approval_callback,
        permission_policy,
    ).content


def run_tool_result(
    registry: dict[str, ToolSpec],
    action: str,
    action_input: ToolInput,
    approval_callback: ApprovalCallback | None = None,
    permission_policy: PermissionPolicy | None = None,
) -> ToolExecutionResult:
    """执行一个工具，并在高风险工具前触发 HITL。

    审批逻辑靠近 dispatch，而不是散落在各个工具内部。这样新增高风险工具
    时，只需设置 `risk="high"`，Harness 行为保持一致。
    """

    tool = registry.get(action)
    if tool is None:
        return ToolExecutionResult(
            STATUS_ERROR,
            f"unknown tool: {action}. available tools: {', '.join(sorted(registry))}",
        )
    action_input_text = stringify_tool_input(action_input)
    perm_result = check_tool_permission(
        action,
        action_input_text,
        permission_policy=permission_policy,
        approval_callback=approval_callback,
        tool_spec=tool,
        tool_input=action_input,
        high_risk_requires_approval=True,
    )
    if perm_result.blocked:
        status = (
            STATUS_APPROVAL_REQUIRED if perm_result.decision == "ask" else STATUS_DENIED
        )
        return ToolExecutionResult(
            status, perm_result.reason, metadata=perm_result.metadata
        )
    try:
        content = redact_text(tool.handler(action_input))
        return ToolExecutionResult(STATUS_OK, content, metadata=perm_result.metadata)
    except Exception as exc:
        meta = {"error": str(exc)}
        if perm_result.metadata:
            meta.update(perm_result.metadata)
        return ToolExecutionResult(STATUS_ERROR, f"tool error: {exc}", meta)


def stringify_tool_input(action_input: ToolInput) -> str:
    return json.dumps(action_input, ensure_ascii=False, sort_keys=True)


BASE_REGISTRY: tuple[ToolSpec, ...] = ()
