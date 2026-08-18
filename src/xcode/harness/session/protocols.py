"""SessionStore / SessionRepo 协议定义。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .types import JsonValue, SessionEntry, SessionInfo, SessionInfoView, TreeNode


class SessionStore(Protocol):
    """会话存储接口：管理当前会话的 entry 级操作。

    与 SessionRepo 分离：SessionRepo 管理多会话生命周期，
    SessionStore 单次打开一个会话后的读写操作。
    """

    @property
    def session_id(self) -> str: ...

    def append(self, record_type: str, content: JsonValue) -> str:
        """追加一条记录，返回 entry id。"""
        ...

    def read_entries(self) -> list[SessionEntry]:
        """读取当前会话的全部 entry。"""
        ...

    def build_branch(self) -> list[SessionEntry]:
        """返回从根到当前叶节点的路径（用于构建 LLM context）。"""
        ...

    def get_forkable_user_messages(self) -> list[SessionEntry]:
        """返回当前分支中可作为 fork 起点的用户消息。"""
        ...

    def fork(self, title: str = "", summary: str = "") -> SessionStore:
        """从当前叶节点分叉，返回新会话的存储。"""
        ...


class SessionRepo(Protocol):
    """会话仓库接口：多会话生命周期管理。"""

    def create(
        self,
        project_root: Path,
        title: str = "",
        summary: str = "",
    ) -> SessionStore:
        """创建新会话并打开。"""
        ...

    def open(self, session_id: str) -> SessionStore:
        """打开已有会话。"""
        ...

    @property
    def current(self) -> SessionStore:
        """当前会话存储。"""
        ...

    def list(self, limit: int | None = 10) -> list[SessionInfo]: ...

    def find_by_id(self, session_id: str) -> SessionInfo | None: ...

    def find_latest_for_project(self, project_root: Path) -> SessionInfoView | None: ...

    def update_summary(self) -> None:
        """更新当前会话摘要。"""
        ...

    def rename(self, title: str) -> None:
        """重命名当前会话。"""
        ...

    def get_tree(self) -> list[TreeNode]:
        """返回会话树的扁平列表（用于 UI 展示）。"""
        ...

    def clear(self) -> None:
        """清空当前会话。"""
        ...
