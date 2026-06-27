"""工具注册表与工具输出。

ToolSpec 描述工具能力，dispatch map 根据工具名找到 handler。"""

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
StreamingActionHandler = Callable[[ToolInput, Callable[[str], None] | None], str]
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
    streaming_handler: StreamingActionHandler | None = None


class ToolRegistryState:
    """保存可在运行期间原子替换的工具注册表快照。"""

    def __init__(self, registry: tuple[ToolSpec, ...]) -> None:
        """使用初始工具列表创建线程安全状态。"""
        self._lock = threading.Lock()
        self._registry: tuple[ToolSpec, ...] = registry

    def snapshot(self) -> tuple[ToolSpec, ...]:
        """返回当前不可变工具快照。"""
        with self._lock:
            return self._registry

    def __iter__(self) -> Iterator[ToolSpec]:
        """迭代调用开始时的稳定工具快照。"""
        return iter(self.snapshot())

    def __len__(self) -> int:
        """返回当前工具数量。"""
        return len(self.snapshot())

    def replace(self, registry: tuple[ToolSpec, ...]) -> None:
        """原子替换完整工具注册表。"""
        with self._lock:
            self._registry = registry

    def replace_group(
        self,
        group: str,
        tools: tuple[ToolSpec, ...],
    ) -> tuple[ToolSpec, ...]:
        """在原有位置替换指定工具组，并返回新快照。"""
        with self._lock:
            existing = self._registry
            insertion_index = next(
                (index for index, rt in enumerate(existing) if rt.group == group),
                len(existing),
            )
            retained = tuple(rt for rt in existing if rt.group != group)
            self._registry = (
                retained[:insertion_index] + tools + retained[insertion_index:]
            )
            return self._registry


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
