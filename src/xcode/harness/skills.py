"""工具注册表与 HITL 执行门禁。

ToolSpec 描述工具能力，dispatch map 根据工具名找到 handler。HITL 在执行
handler 前根据 risk 字段和 permission policy 决定是否需要 approval callback。"""

from __future__ import annotations


from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

from ..agent.protocols import ToolExecutionMode
from ..agent.types import ShellCallOutputContent
from .observability import (
    HITLResult,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
    PermissionRiskEvaluator,
    redact_text,
)
from .session import JsonValue

ToolInput = dict[str, Any]
type ToolMetadataValue = JsonValue | list[ShellCallOutputContent]
type ToolMetadata = dict[str, ToolMetadataValue]
ActionHandler = Callable[[ToolInput], str]
ApprovalCallback = Callable[["ToolSpec", ToolInput], HITLResult]
AGENT_CONTENT_BLOCKS_METADATA_KEY = "agent_content_blocks"


class ToolOutput(str):
    """带结构化元数据的工具输出文本。"""

    metadata: ToolMetadata

    def __new__(
        cls,
        content: str,
        metadata: Mapping[str, object] | None = None,
    ) -> "ToolOutput":
        output = str.__new__(cls, content)
        output.metadata = _tool_metadata(metadata)
        return output


@dataclass(frozen=True)
class ToolSpec:
    """工具的可复用描述。

    prompt_snippet/prompt_guidelines 进入 system prompt，description/input_hint
    保留为工具协议说明，handler 负责执行，risk 决定是否需要人工确认。
    """

    name: str
    description: str
    input_hint: str
    handler: ActionHandler
    risk: str = "low"
    schema: dict[str, Any] | None = None
    read_only: bool = False
    concurrency_safe: bool = False
    risk_evaluator: PermissionRiskEvaluator | None = None
    group: str = "core"
    execution_mode: ToolExecutionMode | None = None
    counts_as_progress: bool | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()
    builtin: dict[str, Any] | None = None


ToolExecutionStatus = Literal["ok", "denied", "error", "approval_required"]

_STATUS_OK: ToolExecutionStatus = "ok"
_STATUS_DENIED: ToolExecutionStatus = "denied"
_STATUS_ERROR: ToolExecutionStatus = "error"
_STATUS_APPROVAL_REQUIRED: ToolExecutionStatus = "approval_required"

_RISK_LOW = "low"
_RISK_HIGH = "high"


@dataclass(frozen=True)
class ToolExecutionResult:
    status: ToolExecutionStatus
    content: str
    metadata: ToolMetadata | None = None


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
    for tool in registry:
        snippet = tool.prompt_snippet or tool.description
        if snippet.strip():
            lines.append(f"- {tool.name}: {snippet.strip()}")
    return "\n".join(lines) if lines else "(none)"


def build_tool_guidelines(registry: tuple[ToolSpec, ...]) -> str:
    guidelines: list[str] = []
    seen: set[str] = set()
    for tool in registry:
        for guideline in tool.prompt_guidelines:
            normalized = guideline.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                guidelines.append(f"- {normalized}")
    return "\n".join(guidelines)


def run_tool_result(
    registry: dict[str, ToolSpec],
    action: str,
    action_input: ToolInput,
    approval_callback: ApprovalCallback | None = None,
    permission_policy: PermissionPolicy | None = None,
    restricted_dirs: tuple[str, ...] = (),
    allowlist_mode: bool = False,
) -> ToolExecutionResult:
    """执行一个工具，并在高风险工具前触发 HITL。

    审批逻辑靠近 dispatch，而不是散落在各个工具内部。这样新增高风险工具
    时，只需设置 `risk="high"`，Harness 行为保持一致。
    """

    tool = registry.get(action)
    if tool is None:
        return ToolExecutionResult(
            _STATUS_ERROR,
            f"unknown tool: {action}. available tools: {', '.join(sorted(registry))}",
        )
    action_input_text = stringify_tool_input(action_input)
    engine = PermissionEngine(
        PermissionEngineConfig(
            static_policy=permission_policy,
            restricted_dirs=restricted_dirs,
            allowlist_mode=allowlist_mode,
            high_risk_requires_approval=True,
        )
    )
    perm_result = engine.decide(
        action,
        action_input_text,
        tool_spec=tool,
        tool_input=action_input,
        approval_callback=approval_callback,
    )
    if perm_result.blocked:
        status = (
            _STATUS_APPROVAL_REQUIRED
            if perm_result.decision == "ask"
            else _STATUS_DENIED
        )
        return ToolExecutionResult(
            status, perm_result.reason, metadata=_merge_metadata(perm_result.metadata)
        )
    try:
        raw_content = tool.handler(action_input)
        content = redact_text(str(raw_content))
        metadata = _merge_metadata(
            _tool_output_metadata(raw_content),
            perm_result.metadata,
        )
        return ToolExecutionResult(_STATUS_OK, content, metadata=metadata)
    except Exception as exc:
        meta = _merge_metadata({"error": str(exc)}, perm_result.metadata)
        return ToolExecutionResult(_STATUS_ERROR, f"tool error: {exc}", meta)


def _tool_output_metadata(output: str) -> ToolMetadata | None:
    if isinstance(output, ToolOutput) and output.metadata:
        return output.metadata
    return None


def _merge_metadata(
    *items: Mapping[str, ToolMetadataValue] | None,
) -> ToolMetadata | None:
    merged: ToolMetadata = {}
    for item in items:
        if item:
            merged.update(item)
    return merged or None


def _tool_metadata(value: object) -> ToolMetadata:
    """规范化工具输出元数据，并保留 agent 专用结构化块。"""
    if not isinstance(value, Mapping):
        return {}
    metadata: ToolMetadata = {}
    for key, item in value.items():
        normalized_key = str(key)
        if normalized_key == AGENT_CONTENT_BLOCKS_METADATA_KEY:
            blocks = _agent_content_blocks(item)
            if blocks:
                metadata[normalized_key] = blocks
            continue
        metadata[normalized_key] = _json_value(item)
    return metadata


def _agent_content_blocks(value: object) -> list[ShellCallOutputContent]:
    """提取可传递给 agent loop 的结构化内容块。"""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, ShellCallOutputContent)]


def _json_value(value: object) -> JsonValue:
    """将任意值转换为可 JSON 序列化的元数据值。"""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def stringify_tool_input(action_input: ToolInput) -> str:
    return json.dumps(action_input, ensure_ascii=False, sort_keys=True)
