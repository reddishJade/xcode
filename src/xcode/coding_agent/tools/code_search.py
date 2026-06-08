from __future__ import annotations

import ast
import re
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Any, Protocol

from xcode.harness.skills import ToolInput, ToolSpec, resolve_project_path
from .path_utils import is_path_blocked, truncate_output, display_path
from .tools_manager import ensure_tool
from .truncate import GREP_MAX_LINE_LENGTH, truncate_line, truncate_tail

"""供编码 Agent 使用的只读代码搜索工具。"""


class GrepOperations(Protocol):
    def is_directory(self, path: Path) -> bool: ...

    def read_file(self, path: Path) -> str: ...


class LsOperations(Protocol):
    def exists(self, path: Path) -> bool: ...

    def is_directory(self, path: Path) -> bool: ...

    def list_dir(self, path: Path) -> list[Path]: ...


class FindOperations(Protocol):
    def exists(self, path: Path) -> bool: ...

    def is_directory(self, path: Path) -> bool: ...


class LocalGrepOperations:
    def is_directory(self, path: Path) -> bool:
        return path.is_dir()

    def read_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")


class LocalLsOperations:
    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_directory(self, path: Path) -> bool:
        return path.is_dir()

    def list_dir(self, path: Path) -> list[Path]:
        return list(path.iterdir())


class LocalFindOperations:
    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_directory(self, path: Path) -> bool:
        return path.is_dir()


_EVAL_NS: dict[str, Any] = {}

# 输出限制：平衡 LLM 上下文窗口利用率与响应速度
MAX_GREP_RESULTS = 100    # grep 最多返回 100 行匹配
MAX_GLOB_RESULTS = 200    # glob 最多返回 200 个文件
MAX_LS_ENTRIES = 500      # ls 最多列出 500 个条目
_RG_MISSING_HINT_EMITTED = False

_BLOCKED_CALL_NAMES = frozenset(
    {
        # 代码执行（防止任意代码执行）
        "exec",
        "eval",
        "compile",

        # I/O 操作（防止文件系统访问）
        "open",
        "input",

        # 动态导入（防止加载未审查的模块）
        "__import__",

        # 动态属性操作（防止访问私有属性和绕过限制）
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "locals",
        "globals",

        # 调试工具（防止交互式中断）
        "breakpoint",
    }
)

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


def build_code_tools(
    project_root: Path,
    grep_ops: GrepOperations | None = None,
    ls_ops: LsOperations | None = None,
    find_ops: FindOperations | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[ToolSpec, ...]:
    root = project_root.resolve()
    local_grep = grep_ops or LocalGrepOperations()
    local_ls = ls_ops or LocalLsOperations()
    local_find = find_ops or LocalFindOperations()

    def _cancel_check() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Tool cancelled")

    def grep_search(data: ToolInput) -> str:
        pattern = str(data.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern is required")
        base = _safe_path(root, str(data.get("path", ".")))
        glob = data.get("glob")
        max_results = int(data.get("max_results", MAX_GREP_RESULTS))
        ignore_case = bool(data.get("ignore_case", False))
        literal = bool(data.get("literal", False))
        context = int(data.get("context", 0))
        _cancel_check()
        return _grep(
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

    def glob_files(data: ToolInput) -> str:
        _cancel_check()
        pattern = str(data.get("pattern", "*")).strip() or "*"
        base = _safe_path(root, str(data.get("path", ".")))
        max_results = int(data.get("max_results", MAX_GLOB_RESULTS))
        return _glob_files(root, base, pattern, max_results)

    def find_files(data: ToolInput) -> str:
        _cancel_check()
        pattern = str(data.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern is required")
        base = _safe_path(root, str(data.get("path", ".")))
        max_results = int(data.get("max_results", MAX_GLOB_RESULTS))
        return _find_files_fd(root, base, pattern, max_results, local_find)

    def ls_files(data: ToolInput) -> str:
        _cancel_check()
        raw_path = str(data.get("path", ".")).strip()
        base = _safe_path(root, raw_path)
        limit = int(data.get("limit", MAX_LS_ENTRIES))
        return _ls(root, base, limit, local_ls)

    def evaluate_python(data: ToolInput) -> str:
        _cancel_check()
        global _EVAL_NS
        code = str(data.get("code", "")).strip()
        if not code:
            raise ValueError("code is required")
        try:
            tree = ast.parse(code, mode="exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = (
                        fn.attr
                        if isinstance(fn, ast.Attribute)
                        else fn.id
                        if isinstance(fn, ast.Name)
                        else ""
                    )
                    if name in _BLOCKED_CALL_NAMES:
                        return f"Error: {name}() is not allowed"
            safe_globals: dict[str, Any] = {
                "__builtins__": _SAFE_BUILTINS,
            }
            safe_globals.update(_EVAL_NS)
            code_object = compile(tree, "<programmatic>", "exec")
            exec(code_object, safe_globals)
            _EVAL_NS.clear()
            _EVAL_NS.update(
                {k: v for k, v in safe_globals.items() if k not in ("__builtins__",)}
            )
            result = _EVAL_NS.get("result") or _EVAL_NS.get("output", "")
            if not result:
                return "ok (no result)"
            return str(result)[:5000]
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"

    def reset_namespace(data: ToolInput) -> str:
        global _EVAL_NS
        _EVAL_NS.clear()
        return "namespace cleared"

    return (
        ToolSpec(
            name="glob_files",
            description="Find project files by glob pattern. Use **/*.py for recursive search.",
            input_hint='JSON: {"path": ".", "pattern": "**/*.py", "max_results": 100}',
            handler=glob_files,
            risk="low",
            read_only=True,
            concurrency_safe=True,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
            },
        ),
        ToolSpec(
            name="find_files",
            description="Find files by glob pattern using fd (fast) with Python glob fallback. Respects .gitignore.",
            input_hint='JSON: {"path": ".", "pattern": "*.py", "max_results": 100}',
            handler=find_files,
            risk="low",
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
            risk="low",
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
            risk="low",
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
                    },
                },
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="evaluate_python",
            description="Execute Python code in a persistent namespace. Intermediate results stay in the namespace, not in LLM context. "
            "Store the final output in a variable named 'result' or 'output' to return it.",
            input_hint='JSON: {"code": "result = 2 + 2"}',
            handler=evaluate_python,
            risk="low",
            read_only=True,
            concurrency_safe=False,
            execution_mode="sequential",
            schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use 'result' variable for return value.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="reset_namespace",
            description="Reset the persistent Python namespace used by evaluate_python. Call this between unrelated tasks.",
            input_hint="{}",
            handler=reset_namespace,
            risk="low",
            read_only=True,
            execution_mode="sequential",
            schema={
                "type": "object",
                "properties": {},
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
    global _RG_MISSING_HINT_EMITTED
    rg = ensure_tool("rg", silent=True)
    if rg:
        command = [rg, "--line-number", "--no-heading", "--color", "never"]
        if ignore_case:
            command.append("--ignore-case")
        if literal:
            command.append("--fixed-strings")
        if context > 0:
            command.extend(["--context", str(context)])
        command.append("--")
        command.append(pattern)
        if glob:
            command.extend(["-g", glob])
        command.append(str(base))
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
                    proc.kill()
                    break
            proc.wait(timeout=5)
        finally:
            proc.stdout.close()
            proc.stderr.close()
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


def _find_files_fd(
    root: Path,
    base: Path,
    pattern: str,
    max_results: int,
    find_ops: FindOperations,
) -> str:
    if not find_ops.exists(base):
        raise FileNotFoundError(f"Path not found: {_display(root, base)}")
    if not find_ops.is_directory(base):
        raise NotADirectoryError(f"Not a directory: {_display(root, base)}")

    fd = ensure_tool("fd", silent=True)
    if fd:
        try:
            args = [
                fd,
                "--glob",
                "--color=never",
                "--hidden",
                "--no-require-git",
                "--max-results",
                str(max_results),
            ]
            adjusted = pattern
            if (
                "/" in pattern
                and not adjusted.startswith("**/")
                and not adjusted.startswith("/")
            ):
                adjusted = f"**/{pattern}"
                args.append("--full-path")
            elif adjusted.startswith("**/") or adjusted.startswith("/"):
                args.append("--full-path")
            args.extend(["--", adjusted, str(base)])
            completed = subprocess.run(
                args,
                cwd=root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
                check=False,
            )
            if completed.returncode == 0 or completed.stdout.strip():
                lines = completed.stdout.strip().splitlines()
                relativized: list[str] = []
                for line in lines:
                    raw = line.strip()
                    if not raw:
                        continue
                    p = Path(raw)
                    try:
                        rel = p.resolve().relative_to(root)
                        suffix = "/" if p.is_dir() else ""
                        relativized.append(rel.as_posix() + suffix)
                    except ValueError:
                        relativized.append(str(p))
                if not relativized:
                    return "No files found."
                result = "\n".join(relativized)
                total_hits = len(lines)
                truncated_by_count = len(relativized) >= max_results
                tr = truncate_tail(result)
                if tr.truncated:
                    result = tr.content
                if truncated_by_count or tr.truncated:
                    result += (
                        f"\n[Found {total_hits} results, showing {tr.output_lines if tr.truncated else len(relativized)}. "
                        f"Use 'max_results=N*2' for more, or refine pattern.]"
                    )
                return result
        except subprocess.TimeoutExpired:
            pass

    return _glob_files(root, base, pattern, max_results)


def _glob_files(root: Path, base: Path, pattern: str, max_results: int) -> str:
    files = sorted(base.glob(pattern) if base.is_dir() else [base])
    matches: list[str] = []
    for path in files:
        if len(matches) >= max_results:
            break
        if path.is_file() and not _is_blocked(root, path):
            matches.append(_display(root, path))
    if not matches:
        return "No files found."
    suffix = "\n... truncated" if len(matches) >= max_results else ""
    return "\n".join(matches) + suffix


def _ls(
    root: Path,
    base: Path,
    limit: int,
    ls_ops: LsOperations,
) -> str:
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
    files = sorted(base.rglob(glob or "*")) if grep_ops.is_directory(base) else [base]

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


def _safe_path(root: Path, raw_path: str) -> Path:
    path = resolve_project_path(root, raw_path)
    if _is_blocked(root, path):
        raise ValueError(f"path is blocked: {_display(root, path)}")
    return path


def _is_blocked(root: Path, path: Path) -> bool:
    return is_path_blocked(root, path)


def _truncate(text: str) -> str:
    return truncate_output(text)


def _display(root: Path, path: Path) -> str:
    return display_path(root, path)
