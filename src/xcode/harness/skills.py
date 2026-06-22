"""工具注册表与 HITL 执行门禁。

ToolSpec 描述工具能力，dispatch map 根据工具名找到 handler。HITL 在执行
handler 前根据 risk 字段和 permission policy 决定是否需要 approval callback。"""

from __future__ import annotations


from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Literal

from ..agent.protocols import ToolExecutionMode
from ..agent.types import FileContent, ImageContent, ShellCallOutputContent
from .observability import HITLResult
from .session import JsonValue

ToolInput = dict[str, Any]
type AgentResultContentBlock = ImageContent | FileContent | ShellCallOutputContent
type ToolMetadataValue = JsonValue | list[AgentResultContentBlock]
type ToolMetadata = dict[str, ToolMetadataValue]
ActionHandler = Callable[[ToolInput], str]
ApprovalCallback = Callable[["ToolSpec", ToolInput], HITLResult]
AGENT_CONTENT_BLOCKS_METADATA_KEY = "agent_content_blocks"
CITATION_SOURCES_METADATA_KEY = "citation_sources"


@dataclass(frozen=True)
class CitationSource:
    """模型可引用的本地证据来源。"""

    kind: Literal["file", "search"]
    path: str
    start_line: int
    end_line: int
    text: str


# ── Governance metadata ──


@dataclass(frozen=True)
class ToolSurfacePolicy:
    """宿主控制的工具展示、调用和委派策略。"""

    exposure: Literal["root", "grouped", "hidden"] = "hidden"
    user_invocable: bool = False
    primary_agent_invocable: bool = False
    subagent_policy: Literal["deny", "explicit_grant", "policy_derived"] = "deny"


# 各类工具的 closed default
SURFACE_POLICY_CORE = ToolSurfacePolicy(
    exposure="root",
    user_invocable=True,
    primary_agent_invocable=True,
    subagent_policy="policy_derived",
)


@dataclass(frozen=True)
class ToolOrigin:
    """工具的传输和来源信息。仅用于审计和展示，不参与权限裁决。"""

    kind: Literal["core", "mcp", "skill"] = "core"
    source: str | None = None


@dataclass(frozen=True)
class ToolActionProfile:
    """宿主控制的动作特征，提供给四轴权限引擎。

    不由 MCP 元数据或工具描述文本推导。action_profile is None 的工具不能进入
    任何 capability envelope 或通过公开 /tool 路径调用。
    """

    capability: str  # read, write, execute, network, credentialed-action
    target_resolver: str
    side_effecting: bool = False
    credentialed: bool = False


@dataclass(frozen=True)
class ToolSelector:
    """用户公开选择器，如 everything.echo。"""

    selector: str


@dataclass(frozen=True)
class RegisteredTool:
    """注册层包装，将 ToolSpec 与宿主治理元数据绑定。"""

    canonical_id: str
    public_selector: ToolSelector
    spec: ToolSpec
    surface_policy: ToolSurfacePolicy
    origin: ToolOrigin
    action_profile: ToolActionProfile | None = None


class ToolOutput(str):
    """带结构化元数据的工具输出文本。"""

    metadata: ToolMetadata
    is_error: bool

    def __new__(
        cls,
        content: str,
        metadata: Mapping[str, object] | None = None,
        is_error: bool = False,
    ) -> "ToolOutput":
        output = str.__new__(cls, content)
        output.metadata = _tool_metadata(metadata)
        output.is_error = is_error
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
    schema: dict[str, Any] | None = None
    read_only: bool = False
    concurrency_safe: bool = False
    group: str = "core"
    execution_mode: ToolExecutionMode | None = None
    counts_as_progress: bool | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()
    builtin: dict[str, Any] | None = None


def _wrap_spec_as_registered(
    spec: ToolSpec,
    origin_kind: str = "core",
    origin_source: str | None = None,
) -> RegisteredTool:
    """将裸 ToolSpec 包装为带 closed default 策略的 RegisteredTool。

    迁移期间使用：所有现有构建站点产生 ToolSpec，通过此函数包装。
    """
    surface = SURFACE_POLICY_CORE if spec.group == "core" else ToolSurfacePolicy()
    return RegisteredTool(
        canonical_id=spec.name,
        public_selector=ToolSelector(spec.name),
        spec=spec,
        surface_policy=surface,
        origin=ToolOrigin(kind=origin_kind, source=origin_source),  # type: ignore[arg-type]
        action_profile=None,
    )


class ToolRegistryState:
    """保存可在运行期间原子替换的工具注册表快照。"""

    def __init__(self, registry: tuple[ToolSpec, ...]) -> None:
        """使用初始工具列表创建线程安全状态。"""
        self._lock = threading.Lock()
        self._registered: tuple[RegisteredTool, ...] = tuple(
            _wrap_spec_as_registered(t) for t in registry
        )

    def snapshot(self) -> tuple[ToolSpec, ...]:
        """返回当前不可变工具快照（兼容旧 API）。"""
        with self._lock:
            return tuple(rt.spec for rt in self._registered)

    def registered_snapshot(self) -> tuple[RegisteredTool, ...]:
        """返回包含治理元数据的当前快照。"""
        with self._lock:
            return self._registered

    def __iter__(self) -> Iterator[ToolSpec]:
        """迭代调用开始时的稳定工具快照。"""
        return iter(self.snapshot())

    def __len__(self) -> int:
        """返回当前工具数量。"""
        return len(self.snapshot())

    def replace(self, registry: tuple[ToolSpec, ...]) -> None:
        """原子替换完整工具注册表。"""
        with self._lock:
            self._registered = tuple(_wrap_spec_as_registered(t) for t in registry)

    def replace_group(
        self,
        group: str,
        tools: tuple[ToolSpec, ...],
    ) -> tuple[ToolSpec, ...]:
        """在原有位置替换指定工具组，并返回新快照。"""
        with self._lock:
            wrapped = tuple(_wrap_spec_as_registered(t, "mcp") for t in tools)
            existing = self._registered
            insertion_index = next(
                (index for index, rt in enumerate(existing) if rt.spec.group == group),
                len(existing),
            )
            retained = tuple(rt for rt in existing if rt.spec.group != group)
            self._registered = (
                retained[:insertion_index] + wrapped + retained[insertion_index:]
            )
            return tuple(rt.spec for rt in self._registered)

    # ── Governance filter API ──

    def tools_visible_to_root(self) -> tuple[RegisteredTool, ...]:
        """Return tools with exposure == 'root' and user_invocable and action_profile present."""
        return tuple(
            rt
            for rt in self.registered_snapshot()
            if rt.surface_policy.exposure == "root"
            and rt.surface_policy.user_invocable
            and rt.action_profile is not None
        )

    def tools_user_invocable(self) -> tuple[RegisteredTool, ...]:
        """Return tools meeting the three /tool selector conditions."""
        return tuple(
            rt
            for rt in self.registered_snapshot()
            if rt.surface_policy.exposure != "hidden"
            and rt.surface_policy.user_invocable
            and rt.action_profile is not None
        )

    def tools_for_completion(self) -> tuple[RegisteredTool, ...]:
        """Return tools visible to tab completion (exposure != 'hidden')."""
        return tuple(
            rt
            for rt in self.registered_snapshot()
            if rt.surface_policy.exposure != "hidden"
        )

    def tools_primary_agent_invocable(self) -> tuple[RegisteredTool, ...]:
        """Return tools eligible for the primary-agent capability envelope."""
        return tuple(
            rt
            for rt in self.registered_snapshot()
            if rt.surface_policy.primary_agent_invocable
            and rt.action_profile is not None
        )

    def tools_for_subagent(self) -> tuple[RegisteredTool, ...]:
        """Return tools with subagent_policy != 'deny' (delegation eligibility ceiling)."""
        return tuple(
            rt
            for rt in self.registered_snapshot()
            if rt.surface_policy.subagent_policy != "deny"
        )

    def resolve_selector(self, selector: str) -> str | None:
        """Map a public selector to canonical id. Returns None if not found."""
        for rt in self.registered_snapshot():
            if rt.public_selector.selector == selector:
                return rt.canonical_id
        return None


def filter_root_visible(
    tools: tuple[RegisteredTool, ...],
) -> tuple[RegisteredTool, ...]:
    return tuple(
        t
        for t in tools
        if t.surface_policy.exposure == "root"
        and t.surface_policy.user_invocable
        and t.action_profile is not None
    )


def filter_user_invocable(
    tools: tuple[RegisteredTool, ...],
) -> tuple[RegisteredTool, ...]:
    return tuple(
        t
        for t in tools
        if t.surface_policy.exposure != "hidden"
        and t.surface_policy.user_invocable
        and t.action_profile is not None
    )


def filter_primary_agent_invocable(
    tools: tuple[RegisteredTool, ...],
) -> tuple[RegisteredTool, ...]:
    return tuple(
        t
        for t in tools
        if t.surface_policy.primary_agent_invocable and t.action_profile is not None
    )


def filter_for_subagent(
    tools: tuple[RegisteredTool, ...],
) -> tuple[RegisteredTool, ...]:
    return tuple(t for t in tools if t.surface_policy.subagent_policy != "deny")


def resolve_public_selector(
    tools: tuple[RegisteredTool, ...],
    selector: str,
) -> str | None:
    for rt in tools:
        if rt.public_selector.selector == selector:
            return rt.canonical_id
    return None


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


def _agent_content_blocks(value: object) -> list[AgentResultContentBlock]:
    """提取可传递给 agent loop 的结构化内容块。"""
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, ImageContent | FileContent | ShellCallOutputContent)
    ]


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
