"""搜索工具共享的枚举、glob 和 gitignore 工具函数。"""

from __future__ import annotations

from collections.abc import Callable
import os
import subprocess
from pathlib import Path

import pathspec

from .tools_manager import get_tool_path


_RG_PATH: str | None = None
_RG_CHECKED = False


def get_rg_path() -> str | None:
    """启动时检查一次 rg 可用性，后续调用缓存结果。"""
    global _RG_PATH, _RG_CHECKED
    if not _RG_CHECKED:
        _RG_PATH = get_tool_path("rg")
        _RG_CHECKED = True
    return _RG_PATH


def enumerate_search_files(
    root: Path,
    base: Path,
    use_ripgrep: bool = True,
) -> list[Path]:
    """枚举可搜索文件，优先使用 ripgrep，回退到 Python walk。"""
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {_display(root, base)}")
    if base.is_file():
        return [] if _is_search_path_excluded(root, base) else [base]
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {_display(root, base)}")

    if use_ripgrep:
        rg = get_rg_path()
        if rg:
            return _enumerate_with_ripgrep(root, base, rg)
    return _enumerate_with_python(root, base)


def _enumerate_with_ripgrep(root: Path, base: Path, rg: str) -> list[Path]:
    command = [
        rg,
        "--files",
        "--color",
        "never",
        "--no-require-git",
        "--no-ignore-dot",
        "--no-ignore-exclude",
        "--no-ignore-global",
        *_rg_exclusion_args(),
        "--",
        str(base),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode not in (0, 1):
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ValueError(f"ripgrep file discovery failed: {detail}")

    files: list[Path] = []
    for line in completed.stdout.splitlines():
        raw_path = line.strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_file() and not _is_search_path_excluded(root, path):
            files.append(path)
    return files


def _enumerate_with_python(root: Path, base: Path) -> list[Path]:
    ignore_specs = _load_gitignore_specs(root)
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        directory = Path(dirpath)
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            child = directory / dirname
            if child.is_symlink() or _is_search_path_excluded(root, child):
                continue
            if _is_gitignored(child, ignore_specs, directory=True):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            path = directory / filename
            if path.is_symlink() or _is_search_path_excluded(root, path):
                continue
            if _is_gitignored(path, ignore_specs):
                continue
            files.append(path.resolve())
    return files


def _load_gitignore_specs(
    root: Path,
) -> tuple[tuple[Path, pathspec.GitIgnoreSpec], ...]:
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in sorted(dirnames)
            if not dirname.startswith(".") and not (directory / dirname).is_symlink()
        ]
        if ".gitignore" not in filenames:
            continue
        ignore_path = directory / ".gitignore"
        try:
            lines = ignore_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError:
            continue
        specs.append((directory.resolve(), pathspec.GitIgnoreSpec.from_lines(lines)))
    return tuple(specs)


def _is_gitignored(
    path: Path,
    specs: tuple[tuple[Path, pathspec.GitIgnoreSpec], ...],
    *,
    directory: bool = False,
) -> bool:
    ignored = False
    resolved = path.resolve()
    for spec_root, spec in specs:
        try:
            relative = resolved.relative_to(spec_root).as_posix()
        except ValueError:
            continue
        if directory:
            relative += "/"
        decision = spec.check_file(relative).include
        if decision is not None:
            ignored = decision
    return ignored


def build_path_matcher(
    pattern: str,
    *,
    recursive_basename: bool,
) -> Callable[[str], bool]:
    normalized = pattern.replace("\\", "/").removeprefix("./")
    if recursive_basename and "/" not in normalized:
        normalized = f"**/{normalized}"
    try:
        spec = pathspec.GitIgnoreSpec.from_lines([f"/{normalized}"])
    except ValueError as exc:
        raise ValueError(f"Invalid glob pattern: {exc}") from exc
    return spec.match_file


def _rg_exclusion_args() -> list[str]:
    patterns = (
        "!**/.git/**",
        "!**/.venv/**",
        "!**/__pycache__/**",
        "!**/.local/chroma_db/**",
        "!**/.env",
        "!**/.env.*",
    )
    args: list[str] = []
    for pattern in patterns:
        args.extend(["--glob", pattern])
    return args


def _is_search_path_excluded(root: Path, path: Path) -> bool:
    from .path_utils import is_path_blocked

    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return True
    return is_path_blocked(root, path) or any(
        part.startswith(".") for part in relative.parts
    )


def _display(root: Path, path: Path) -> str:
    from .path_utils import display_path

    return display_path(root, path)


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0
