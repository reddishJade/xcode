from __future__ import annotations

import subprocess
import time
from pathlib import Path

"""每轮任务开始前注入的 Git 工作区基线。"""

# Git 输出限制和缓存配置
MAX_SECTION_CHARS = 6_000  # Git 输出截断：避免大 diff 撑爆 system prompt
CACHE_TTL = 5.0  # 缓存有效期（秒）：减少频繁 git 调用开销

_ttl_cache: dict[str, tuple[float, str]] = {}
_snapshot_cache: dict[tuple[str, str, tuple[tuple[str, int, int], ...]], str] = {}


def build_git_preflight(project_root: Path) -> str:
    key = str(project_root.resolve())
    now = time.monotonic()
    cached = _ttl_cache.get(key)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]
    status = _run_git(project_root, "status", "--short")
    if status is None:
        return "<git-preflight>\nstatus: unavailable\n</git-preflight>"
    initial_snapshot = _git_metadata_snapshot(project_root, status)
    snapshot_key = (key, status, initial_snapshot)
    snapshot_cached = _snapshot_cache.get(snapshot_key)
    if snapshot_cached is not None:
        _ttl_cache[key] = (now, snapshot_cached)
        return snapshot_cached

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
    _ttl_cache[key] = (time.monotonic(), result)
    _snapshot_cache[snapshot_key] = result
    final_snapshot = _git_metadata_snapshot(project_root, status)
    if final_snapshot != initial_snapshot:
        _snapshot_cache[(key, status, final_snapshot)] = result
    return result


def _git_metadata_snapshot(
    project_root: Path, status: str
) -> tuple[tuple[str, int, int], ...]:
    """读取稳定 Git 与脏文件 mtime，用于跨 TTL 复用 preflight。"""
    git_dir = project_root / ".git"
    if git_dir.is_file():
        try:
            text = git_dir.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ()
        prefix = "gitdir:"
        if text.lower().startswith(prefix):
            git_dir = (project_root / text[len(prefix) :].strip()).resolve()
    if not git_dir.exists():
        return ()

    candidates = [git_dir / "HEAD", git_dir / "packed-refs"]
    heads = git_dir / "refs" / "heads"
    if heads.exists():
        candidates.extend(path for path in sorted(heads.rglob("*")) if path.is_file())
    candidates.extend(_status_paths(project_root, status))

    snapshot: list[tuple[str, int, int]] = []
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            name = path.relative_to(git_dir).as_posix()
        except ValueError:
            name = path.name
        snapshot.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(snapshot)


def _status_paths(project_root: Path, status: str) -> list[Path]:
    """从 short status 提取工作区路径。"""
    paths: list[Path] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        raw_path = line[3:].strip()
        if " -> " in raw_path:
            raw_path = raw_path.rsplit(" -> ", 1)[1]
        normalized = raw_path.strip('"')
        if normalized:
            paths.append(project_root / normalized)
    return paths


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
