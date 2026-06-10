from __future__ import annotations

import subprocess
import time
from pathlib import Path

"""每轮任务开始前注入的 Git 工作区基线。"""

# Git 输出限制和缓存配置
MAX_SECTION_CHARS = 6_000  # Git 输出截断：避免大 diff 撑爆 system prompt
CACHE_TTL = 5.0  # 缓存有效期（秒）：减少频繁 git 调用开销

_cache: dict[str, tuple[float, str]] = {}


def build_git_preflight(project_root: Path) -> str:
    key = str(project_root.resolve())
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]
    status = _run_git(project_root, "status", "--short")
    if status is None:
        return "<git-preflight>\nstatus: unavailable\n</git-preflight>"

    last_commit = _run_git(project_root, "show", "--stat", "--oneline", "-1")
    lines = ["<git-preflight>"]
    clean = not status.strip()
    lines.append("status:")
    lines.append(status.strip() or "clean")
    if last_commit:
        lines.append("\nlast_commit:")
        lines.append(_truncate(last_commit.strip()))
    if not clean:
        unstaged = _run_git(project_root, "diff", "--stat")
        staged = _run_git(project_root, "diff", "--cached", "--stat")
        if unstaged:
            lines.append("\ndirty_diff_stat:")
            lines.append(_truncate(unstaged.strip()))
        if staged:
            lines.append("\nstaged_diff_stat:")
            lines.append(_truncate(staged.strip()))
        lines.append(
            "\nWorking tree has pre-existing changes. Treat them as user-owned "
            "baseline. Before editing any dirty file, inspect the file and "
            "relevant diff; do not overwrite unrelated changes."
        )
    lines.append("</git-preflight>")
    result = "\n".join(lines)
    _cache[key] = (time.monotonic(), result)
    return result


def _run_git(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=3,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _truncate(text: str) -> str:
    if len(text) <= MAX_SECTION_CHARS:
        return text
    keep = MAX_SECTION_CHARS - 80
    return text[:keep] + f"\n[... truncated {len(text) - keep} chars ...]"
