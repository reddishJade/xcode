"""受限时项目文件索引，供交互式补全复用。"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pathspec

from .path_utils import is_path_blocked


def build_project_file_index(
    project_root: Path,
    *,
    max_files: int = 5000,
    time_budget_seconds: float = 0.075,
) -> tuple[str, ...]:
    """在数量和时间预算内枚举可见、未忽略的项目文件。"""
    root = project_root.resolve()
    deadline = time.perf_counter() + max(0.001, time_budget_seconds)
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []
    files: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if time.perf_counter() >= deadline or len(files) >= max_files:
            break
        directory = Path(dirpath).resolve()
        _append_gitignore_spec(directory, filenames, specs)
        active_specs = [
            (spec_root, spec)
            for spec_root, spec in specs
            if _contains_path(spec_root, directory)
        ]

        kept_dirs: list[str] = []
        for dirname in sorted(dirnames, key=str.casefold):
            child = directory / dirname
            if dirname.startswith(".") or child.is_symlink():
                continue
            if is_path_blocked(root, child):
                continue
            if _is_ignored(child, active_specs, directory=True):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames, key=str.casefold):
            if time.perf_counter() >= deadline or len(files) >= max_files:
                break
            if filename.startswith("."):
                continue
            path = directory / filename
            if path.is_symlink() or not path.is_file():
                continue
            if is_path_blocked(root, path) or _is_ignored(path, active_specs):
                continue
            files.append(path.relative_to(root).as_posix())

    return tuple(files)


def _append_gitignore_spec(
    directory: Path,
    filenames: list[str],
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]],
) -> None:
    """加载当前目录的 .gitignore。"""
    if ".gitignore" not in filenames:
        return
    path = directory / ".gitignore"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    specs.append((directory, pathspec.GitIgnoreSpec.from_lines(lines)))


def _is_ignored(
    path: Path,
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]],
    *,
    directory: bool = False,
) -> bool:
    """按父到子顺序应用最后匹配的 ignore 决策。"""
    ignored = False
    for spec_root, spec in specs:
        try:
            relative = path.resolve().relative_to(spec_root).as_posix()
        except ValueError:
            continue
        if directory:
            relative += "/"
        decision = spec.check_file(relative).include
        if decision is not None:
            ignored = decision
    return ignored


def _contains_path(parent: Path, child: Path) -> bool:
    """判断 child 是否位于 parent 内。"""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
