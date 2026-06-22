"""供编码 Agent 使用的只读代码搜索工具。"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pathspec

from xcode.harness.skills import (
    CITATION_SOURCES_METADATA_KEY,
    ToolInput,
    ToolOutput,
    ToolSpec,
    resolve_project_path,
)
from .path_utils import display_path, is_path_blocked, truncate_output
from .tools_manager import ensure_tool
from .truncate import GREP_MAX_LINE_LENGTH, truncate_line, truncate_tail


class GrepOperations(Protocol):
    """定义 Python grep fallback 所需的文件读取边界。"""

    def read_file(self, path: Path) -> str:
        """读取文本文件。"""
        ...


class LsOperations(Protocol):
    """定义 ls 工具所需的文件系统操作。"""

    def exists(self, path: Path) -> bool:
        """判断路径是否存在。"""
        ...

    def is_directory(self, path: Path) -> bool:
        """判断路径是否为目录。"""
        ...

    def list_dir(self, path: Path) -> list[Path]:
        """列出目录中的直接子项。"""
        ...


class LocalGrepOperations:
    """提供本地文件系统 grep 操作。"""

    def read_file(self, path: Path) -> str:
        """读取 UTF-8 文本并忽略无法解码的字节。"""
        return path.read_text(encoding="utf-8", errors="ignore")


class LocalLsOperations:
    """提供本地文件系统 ls 操作。"""

    def exists(self, path: Path) -> bool:
        """判断路径是否存在。"""
        return path.exists()

    def is_directory(self, path: Path) -> bool:
        """判断路径是否为目录。"""
        return path.is_dir()

    def list_dir(self, path: Path) -> list[Path]:
        """列出目录中的直接子项。"""
        return list(path.iterdir())


# 输出限制：平衡 LLM 上下文窗口利用率与响应速度
MAX_GREP_RESULTS = 100  # grep 最多返回 100 行匹配
MAX_GLOB_RESULTS = 200  # glob 最多返回 200 个文件
MAX_LS_ENTRIES = 500  # ls 最多列出 500 个条目
_RG_MISSING_HINT_EMITTED = False


def build_code_tools(
    project_root: Path,
    grep_ops: GrepOperations | None = None,
    ls_ops: LsOperations | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[ToolSpec, ...]:
    """构建项目范围内的只读代码搜索工具。"""
    root = project_root.resolve()
    local_grep = grep_ops or LocalGrepOperations()
    local_ls = ls_ops or LocalLsOperations()

    def _cancel_check() -> None:
        """在调用开始前检查取消状态。"""
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Tool cancelled")

    def grep_search(data: ToolInput) -> str:
        """执行内容搜索。"""
        pattern = str(data.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern is required")
        base = _safe_path(root, str(data.get("path", ".")))
        glob = data.get("glob")
        max_results = _validated_int(
            data,
            "max_results",
            MAX_GREP_RESULTS,
            minimum=1,
        )
        ignore_case = bool(data.get("ignore_case", False))
        literal = bool(data.get("literal", False))
        context = _validated_int(data, "context", 0, minimum=0)
        _cancel_check()
        result = _grep(
            root,
            base,
            pattern,
            str(glob) if glob else None,
            max_results,
            local_grep,
            ignore_case=ignore_case,
            literal=literal,
            context=context,
            cancel_event=cancel_event,
        )
        sources = _citation_sources_from_grep_output(result)
        if sources:
            return ToolOutput(result, metadata={CITATION_SOURCES_METADATA_KEY: sources})
        return result

    def glob_files(data: ToolInput) -> str:
        """按项目相对 glob 搜索文件。"""
        _cancel_check()
        pattern = str(data.get("pattern", "*")).strip() or "*"
        base = _safe_path(root, str(data.get("path", ".")))
        max_results = _validated_int(
            data,
            "max_results",
            MAX_GLOB_RESULTS,
            minimum=1,
        )
        return _glob_files(root, base, pattern, max_results, recursive_basename=False)

    def find_files(data: ToolInput) -> str:
        """递归按文件名或路径 glob 搜索文件。"""
        _cancel_check()
        pattern = str(data.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern is required")
        base = _safe_path(root, str(data.get("path", ".")))
        max_results = _validated_int(
            data,
            "max_results",
            MAX_GLOB_RESULTS,
            minimum=1,
        )
        return _glob_files(root, base, pattern, max_results, recursive_basename=True)

    def ls_files(data: ToolInput) -> str:
        """列出目录中的直接子项。"""
        _cancel_check()
        raw_path = str(data.get("path", ".")).strip()
        base = _safe_path(root, raw_path)
        limit = _validated_int(data, "limit", MAX_LS_ENTRIES, minimum=1)
        return _ls(root, base, limit, local_ls)

    return (
        ToolSpec(
            name="glob_files",
            description="Find project files by glob pattern. Use **/*.py for recursive search.",
            input_hint='JSON: {"path": ".", "pattern": "**/*.py", "max_results": 100}',
            handler=glob_files,
            read_only=True,
            concurrency_safe=True,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1},
                },
            },
        ),
        ToolSpec(
            name="find_files",
            description=(
                "Find files recursively by name or path glob. "
                "Uses the same .gitignore-aware discovery engine as glob_files."
            ),
            input_hint='JSON: {"path": ".", "pattern": "*.py", "max_results": 100}',
            handler=find_files,
            read_only=True,
            concurrency_safe=True,
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match, e.g. '*.ts' or 'src/**/*.py'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default: 200)",
                        "minimum": 1,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="grep_search",
            description="Search file contents for a pattern. Uses ripgrep when available, then falls back to Python grep.",
            input_hint='JSON: {"pattern": "ToolSpec", "path": "src/xcode", "glob": "*.py"}',
            handler=grep_search,
            read_only=True,
            concurrency_safe=True,
            schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex or literal string)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter, e.g. '*.py' or '**/*.spec.ts'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max matches (default: 100)",
                        "minimum": 1,
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)",
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Treat pattern as literal string instead of regex (default: false)",
                    },
                    "context": {
                        "type": "integer",
                        "description": "Lines of context before and after each match (default: 0)",
                        "minimum": 0,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="ls",
            description="List directory contents. Entries sorted alphabetically, '/' suffix for directories. Includes dotfiles.",
            input_hint='JSON: {"path": "src/xcode", "limit": 100}',
            handler=ls_files,
            read_only=True,
            concurrency_safe=True,
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum entries (default: 500)",
                        "minimum": 1,
                    },
                },
                "additionalProperties": False,
            },
        ),
    )


def _grep(
    root: Path,
    base: Path,
    pattern: str,
    glob: str | None,
    max_results: int,
    grep_ops: GrepOperations,
    *,
    ignore_case: bool = False,
    literal: bool = False,
    context: int = 0,
    cancel_event: threading.Event | None = None,
) -> str:
    """优先使用 ripgrep 搜索内容，缺失时使用 Python fallback。"""
    global _RG_MISSING_HINT_EMITTED
    rg = ensure_tool("rg", silent=True)
    if rg:
        search_paths = [base]
        if glob:
            matcher = _build_path_matcher(glob, recursive_basename=True)
            search_paths = [
                path
                for path in _enumerate_search_files(root, base)
                if matcher(
                    path.relative_to(base).as_posix() if base.is_dir() else path.name
                )
            ]
            if not search_paths:
                return "No matches found."
            if _command_path_chars(search_paths) > 24_000:
                return _grep_fallback(
                    root,
                    base,
                    pattern,
                    glob,
                    max_results,
                    grep_ops,
                    ignore_case=ignore_case,
                    literal=literal,
                    context=context,
                )
        command = [
            rg,
            "--line-number",
            "--no-heading",
            "--with-filename",
            "--color",
            "never",
            "--no-require-git",
            "--no-ignore-dot",
            "--no-ignore-exclude",
            "--no-ignore-global",
        ]
        if ignore_case:
            command.append("--ignore-case")
        if literal:
            command.append("--fixed-strings")
        if context > 0:
            command.extend(["--context", str(context)])
        command.extend(_rg_exclusion_args())
        command.extend(["--", pattern])
        command.extend(str(path) for path in search_paths)
        proc = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        lines: list[str] = []
        lines_truncated = 0
        stopped_for_limit = False
        stderr = ""
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    proc.kill()
                    raise ValueError("Tool cancelled")
                raw_line = proc.stdout.readline()
                if not raw_line:
                    break
                line = raw_line.rstrip("\n").rstrip("\r")
                truncated, was_truncated = truncate_line(line)
                if was_truncated:
                    lines_truncated += 1
                lines.append(truncated)
                if len(lines) >= max_results:
                    stopped_for_limit = True
                    proc.kill()
                    break
            proc.wait(timeout=5)
            stderr = proc.stderr.read().strip()
        finally:
            proc.stdout.close()
            proc.stderr.close()
        if not stopped_for_limit and proc.returncode not in (0, 1):
            detail = stderr or f"exit code {proc.returncode}"
            raise ValueError(f"ripgrep failed: {detail}")
        if not lines:
            return "No matches found."
        result = "\n".join(lines)
        if lines_truncated > 0:
            result += f"\n[Truncated {lines_truncated} long lines to {GREP_MAX_LINE_LENGTH} chars]"
        tr = truncate_tail(result)
        if tr.truncated:
            result = tr.content
            result += (
                f"\n[Showing {tr.output_lines} of {tr.total_lines} lines "
                f"({tr.max_bytes // 1024}KB limit). "
                f"Use 'max_results=N*2' for more, or refine pattern.]"
            )
        return result

    hint = ""
    if not _RG_MISSING_HINT_EMITTED:
        _RG_MISSING_HINT_EMITTED = True
        hint = "[ripgrep not found; install rg for faster search. Falling back to Python grep.]\n"
    return hint + _grep_fallback(
        root,
        base,
        pattern,
        glob,
        max_results,
        grep_ops,
        ignore_case=ignore_case,
        literal=literal,
        context=context,
    )


def _glob_files(
    root: Path,
    base: Path,
    pattern: str,
    max_results: int,
    *,
    recursive_basename: bool,
) -> str:
    """枚举并按 glob、修改时间和结果上限筛选文件。"""
    files = _enumerate_search_files(root, base)
    matcher = _build_path_matcher(pattern, recursive_basename=recursive_basename)
    matching_files = [
        path
        for path in files
        if matcher(path.relative_to(base).as_posix() if base.is_dir() else path.name)
    ]
    matching_files.sort(key=lambda path: (-_mtime_ns(path), _display(root, path)))
    matches = [_display(root, path) for path in matching_files[:max_results]]
    if not matches:
        return "No files found."
    suffix = "\n... truncated" if len(matching_files) > max_results else ""
    return "\n".join(matches) + suffix


def _ls(
    root: Path,
    base: Path,
    limit: int,
    ls_ops: LsOperations,
) -> str:
    """列出目录内容并应用 blocked path 规则。"""
    if not ls_ops.exists(base):
        raise FileNotFoundError(f"Path not found: {_display(root, base)}")
    if not ls_ops.is_directory(base):
        raise NotADirectoryError(f"Not a directory: {_display(root, base)}")
    try:
        entries = sorted(ls_ops.list_dir(base), key=lambda p: p.name.lower())
    except PermissionError as exc:
        raise PermissionError(f"Permission denied: {_display(root, base)}") from exc

    lines: list[str] = []
    for entry in entries:
        if len(lines) >= limit:
            break
        if _is_blocked(root, entry):
            continue
        name = _display(root, entry)
        if entry.is_dir():
            name += "/"
        lines.append(name)

    if not lines:
        return "(empty directory)"
    result = "\n".join(lines)
    if len(entries) > limit:
        result += f"\n... {len(entries) - limit} more entries omitted"
    return result


def _grep_fallback(
    root: Path,
    base: Path,
    pattern: str,
    glob: str | None,
    max_results: int,
    grep_ops: GrepOperations,
    *,
    ignore_case: bool = False,
    literal: bool = False,
    context: int = 0,
) -> str:
    """使用 Python 正则逐文件执行内容搜索。"""
    files = _enumerate_search_files(root, base, use_ripgrep=False)
    if glob:
        matcher = _build_path_matcher(glob, recursive_basename=True)
        files = [
            path
            for path in files
            if matcher(
                path.relative_to(base).as_posix() if base.is_dir() else path.name
            )
        ]

    re_pattern: re.Pattern[str] | None = None
    if not literal:
        try:
            flags = re.IGNORECASE if ignore_case else 0
            re_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

    matches: list[str] = []
    for path in files:
        if len(matches) >= max_results:
            break
        if not path.is_file() or _is_blocked(root, path):
            continue
        text = grep_ops.read_file(path)
        all_lines = text.splitlines()
        match_indices: list[int] = []
        for line_no, line in enumerate(all_lines, 1):
            if literal:
                search_line = line.casefold() if ignore_case else line
                search_pattern = pattern.casefold() if ignore_case else pattern
                matched = search_pattern in search_line
            else:
                matched = bool(re_pattern.search(line)) if re_pattern else False
            if matched:
                match_indices.append(line_no)
                if len(matches) >= max_results:
                    break
        for line_no in match_indices:
            if len(matches) >= max_results:
                break
            start = max(0, line_no - 1 - context) if context > 0 else line_no - 1
            end = min(len(all_lines), line_no + context) if context > 0 else line_no
            for ctx_line_no in range(start, end):
                prefix = ">" if ctx_line_no + 1 == line_no else "-"
                raw = all_lines[ctx_line_no]
                truncated_raw, _ = truncate_line(raw)
                matches.append(
                    f"{_display(root, path)}:{ctx_line_no + 1}:{prefix} {truncated_raw}"
                )
    return _truncate("\n".join(matches)) if matches else "No matches found."


def _enumerate_search_files(
    root: Path,
    base: Path,
    *,
    use_ripgrep: bool = True,
) -> list[Path]:
    """枚举可搜索文件，并统一应用 ignore、hidden 和 blocked 规则。"""
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {_display(root, base)}")
    if base.is_file():
        return [] if _is_search_path_excluded(root, base) else [base]
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {_display(root, base)}")

    if use_ripgrep:
        rg = ensure_tool("rg", silent=True)
        if rg:
            return _enumerate_with_ripgrep(root, base, rg)
    return _enumerate_with_python(root, base)


def _enumerate_with_ripgrep(root: Path, base: Path, rg: str) -> list[Path]:
    """使用 ripgrep 枚举文件，并将命令错误转换为明确诊断。"""
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
    ]
    command.extend(["--", str(base)])
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
    """使用 Python walker 和 .gitignore 规则枚举文件。"""
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
    """按目录加载项目内 .gitignore 规则。"""
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in sorted(dirnames)
            if not dirname.startswith(".")
            and not _is_blocked(root, directory / dirname)
            and not (directory / dirname).is_symlink()
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
    """按父目录到子目录顺序应用 .gitignore 的最后匹配规则。"""
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


def _build_path_matcher(
    pattern: str,
    *,
    recursive_basename: bool,
) -> Callable[[str], bool]:
    """构建相对路径 glob matcher。"""
    normalized = pattern.replace("\\", "/").removeprefix("./")
    if recursive_basename and "/" not in normalized:
        normalized = f"**/{normalized}"
    try:
        spec = pathspec.GitIgnoreSpec.from_lines([f"/{normalized}"])
    except ValueError as exc:
        raise ValueError(f"Invalid glob pattern: {exc}") from exc
    return spec.match_file


def _rg_exclusion_args() -> list[str]:
    """返回 ripgrep 搜索与文件枚举共用的 blocked path 排除参数。"""
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
    """判断路径是否因 hidden 或 blocked 规则排除。"""
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return True
    return _is_blocked(root, path) or any(
        part.startswith(".") for part in relative.parts
    )


def _mtime_ns(path: Path) -> int:
    """返回文件修改时间；无法读取时按最旧处理。"""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _command_path_chars(paths: list[Path]) -> int:
    """估算文件路径参数占用，避免超过 Windows 命令行限制。"""
    return sum(len(str(path)) + 1 for path in paths)


def _validated_int(
    data: ToolInput,
    key: str,
    default: int,
    *,
    minimum: int,
) -> int:
    """读取并校验工具整数参数。"""
    raw = data.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    if raw < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return raw


def _safe_path(root: Path, raw_path: str) -> Path:
    """解析项目路径并拒绝 hidden 或 blocked 目标。"""
    path = resolve_project_path(root, raw_path)
    if _is_search_path_excluded(root, path):
        raise ValueError(f"path is blocked: {_display(root, path)}")
    return path


def _is_blocked(root: Path, path: Path) -> bool:
    """判断路径是否命中项目 blocked 规则。"""
    return is_path_blocked(root, path)


def _truncate(text: str) -> str:
    """按工具输出预算截断文本。"""
    return truncate_output(text)


def _citation_sources_from_grep_output(
    text: str,
) -> list[dict[str, object]]:
    """从 grep 输出文本解析行级引用来源。

    每行格式: path:line_num:content。
    最多 30 个 per-line 来源，超出后聚合为一个 range source。
    """
    pattern = re.compile(r"^(.+?):(\d+):(.*)")
    sources: list[dict[str, object]] = []
    range_start: int | None = None
    lines = text.splitlines()
    for line in lines:
        m = pattern.match(line)
        if not m:
            continue
        path = m.group(1)
        line_num = int(m.group(2))
        if len(sources) < 30:
            sources.append(
                {
                    "kind": "search",
                    "path": path,
                    "start_line": line_num,
                    "end_line": line_num,
                    "text": line,
                }
            )
        else:
            if range_start is None:
                range_start = line_num
    if range_start is not None:
        sources.append(
            {
                "kind": "search",
                "path": "",
                "start_line": range_start,
                "end_line": len(lines),
                "text": f"[{len(sources) - 30}+ more lines]",
            }
        )
    return sources


def _display(root: Path, path: Path) -> str:
    """返回稳定的项目相对显示路径。"""
    return display_path(root, path)
