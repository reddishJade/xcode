from __future__ import annotations

from ..cli.session import (
    FORK_TYPES,
    SessionMetadata,
    SessionMetadataView,
    SessionRecord,
    SessionStore as CliSessionStore,
)

"""Session 存储，委托给 cli/session.py 的完整实现。"""


class SessionStore(CliSessionStore):
    """JSONL 持久化会话存储（含分支支持）。"""


__all__ = [
    "FORK_TYPES",
    "SessionMetadata",
    "SessionMetadataView",
    "SessionRecord",
    "SessionStore",
]
