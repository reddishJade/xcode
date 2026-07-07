"""会话层共享数据类型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class SessionEntry:
    """会话中的单条记录（树节点）。"""

    id: str
    parent_id: str | None
    type: str
    content: JsonValue
    created_at: str


@dataclass(frozen=True)
class SessionInfo:
    """会话元数据快照。"""

    id: str
    title: str
    summary: str
    project_path: str
    created_at: str
    updated_at: str
    parent_id: str | None = None
    entry_count: int = 0


@dataclass(frozen=True)
class SessionInfoView:
    """带有文件路径的会话元数据，用于 UI 展示。"""

    id: str
    title: str
    summary: str
    updated_at: str
    path: Path
    project_path: str = ""
    parent_id: str | None = None


@dataclass(frozen=True)
class TreeNode:
    """会话树节点，用于 get_tree() 展示。"""

    id: str
    title: str
    depth: int
    is_current: bool
    is_leaf: bool
