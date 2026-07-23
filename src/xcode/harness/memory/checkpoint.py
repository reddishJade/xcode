"""长任务 session checkpoint 的持久化。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile

_BOUNDARY_PATTERN = re.compile(r"(?m)^Boundary-Message-ID:[ \t]*(.+?)$")


@dataclass(frozen=True)
class SessionCheckpoint:
    """一次 compact cycle 的可恢复状态。"""

    session_id: str
    boundary_message_id: str
    body: str
    path: Path

    def render_rebuild_prompt(self) -> str:
        """渲染给恢复后模型的上下文种子。"""
        return (
            "<session-checkpoint>\n"
            "This is the durable state from the previous context cycle. "
            "Recent transcript messages follow verbatim.\n\n"
            f"{self.body.strip()}\n"
            "</session-checkpoint>"
        )


def write_session_checkpoint(
    root: Path,
    *,
    session_id: str,
    boundary_message_id: str,
    summary: str,
    read_files: set[str] | None = None,
    modified_files: set[str] | None = None,
) -> SessionCheckpoint:
    """以原子替换写入当前 session 的最新 checkpoint。"""
    path = _checkpoint_path(root, session_id)
    state = summary.removeprefix("[Compressed]").strip()
    sections = [
        "# Session checkpoint",
        f"Boundary-Message-ID: {boundary_message_id}",
        "",
        state,
    ]
    tracked = _render_files(read_files or set(), modified_files or set())
    if tracked:
        sections.extend(["", tracked])
    body = "\n".join(sections).rstrip() + "\n"
    _atomic_write(path, body)
    return SessionCheckpoint(
        session_id=session_id,
        boundary_message_id=boundary_message_id,
        body=body,
        path=path,
    )


def load_session_checkpoint(
    root: Path,
    session_id: str,
) -> SessionCheckpoint | None:
    """读取 session 最新 checkpoint；损坏时安全回退到完整 transcript。"""
    path = _checkpoint_path(root, session_id)
    if not path.is_file():
        return None
    try:
        body = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _BOUNDARY_PATTERN.search(body)
    if match is None:
        return None
    boundary = match.group(1).strip()
    if not boundary:
        return None
    return SessionCheckpoint(
        session_id=session_id,
        boundary_message_id=boundary,
        body=body,
        path=path,
    )


def _checkpoint_path(root: Path, session_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id).strip("._")
    if not safe_id:
        raise ValueError("session_id must contain a safe path character")
    return root / safe_id / "checkpoint.md"


def _render_files(read_files: set[str], modified_files: set[str]) -> str:
    if not read_files and not modified_files:
        return ""
    lines = ["## Files"]
    if read_files:
        lines.append("### Read")
        lines.extend(f"- {path}" for path in sorted(read_files))
    if modified_files:
        lines.append("### Modified")
        lines.extend(f"- {path}" for path in sorted(modified_files))
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".checkpoint.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
