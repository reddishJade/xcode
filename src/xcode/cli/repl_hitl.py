"""HITL 授权处理器：交互式用户提示桥接。

不执行 grant 查找或写入。PermissionEngine 负责授权检查与持久化。
"""

from __future__ import annotations

import threading
from queue import Empty, Queue

import questionary
from rich.console import Console
from rich.panel import Panel

from xcode.agent.types import ApprovalRequest, ApprovalScope, ToolInput, ToolSpec
from xcode.harness.security import HITLDecision, HITLResult, HITLScope
from xcode.harness.security.permission_model.utils import command_grant_pattern

from .repl_tools import brief_input

_DEFAULT_HITL_TIMEOUT: float = 300.0
_SCOPE_CHOICE = {
    "once": "Allow (once)",
    "session": "Allow this session",
    "permanent": "Always allow",
}


def hitl_choices(allowed_scopes: tuple[ApprovalScope, ...]) -> tuple[str, ...]:
    """将引擎允许的授权范围转换成界面选项。"""
    return tuple(_SCOPE_CHOICE[scope] for scope in allowed_scopes) + ("Deny",)


HITL_CHOICES = hitl_choices(("once", "session", "permanent"))
_console = Console()


def parse_hitl_choice(text: str) -> HITLResult | None:
    """将用户输入的权限选择文本解析为 HITLResult。"""
    normalized = text.strip().lower()
    mapping: dict[str, tuple[HITLDecision, HITLScope]] = {
        "deny": ("deny", "once"),
        "allow (once)": ("allow", "once"),
        "allow once": ("allow", "once"),
        "allow this session": ("allow", "session"),
        "always allow": ("allow", "permanent"),
    }
    pair = mapping.get(normalized)
    if pair is None:
        return None
    return HITLResult(*pair)


class ReplHITLHandler:
    """HITL 授权处理器——仅交互式提示。

    PermissionEngine 负责 grant 查找与写入。
    本类只桥接用户的选择并返回 HITLResult。
    """

    def __init__(
        self,
        prompt: object | None = None,
        timeout: float = _DEFAULT_HITL_TIMEOUT,
    ) -> None:
        self._prompt = prompt
        self._timeout = timeout

    def __call__(self, request: ApprovalRequest) -> HITLResult:
        tool = request.tool
        action_input = request.action_input
        _print_tool_preview(tool, action_input, request.allowed_scopes)
        choice = _ask_hitl_choice_with_timeout(
            tool,
            action_input,
            choices=hitl_choices(request.allowed_scopes),
            timeout=self._timeout,
        )
        suggestion = ""
        if choice == "Deny":
            suggestion = _ask_suggestion_with_timeout(timeout=self._timeout)
        return self._apply_choice(choice, suggestion)

    def _apply_choice(self, choice: str | None, suggestion: str = "") -> HITLResult:
        if choice == "Allow (once)":
            return HITLResult("allow", "once")
        if choice == "Allow this session":
            return HITLResult("allow", "session")
        if choice == "Always allow":
            return HITLResult("allow", "permanent")
        return HITLResult("deny", "once", suggestion=suggestion)


# ── Preview ──


def _print_tool_preview(
    tool: ToolSpec,
    action_input: ToolInput,
    allowed_scopes: tuple[ApprovalScope, ...],
) -> None:
    """在 HITL 提示之前打印丰富的工具预览信息。"""
    lines = tool_preview_lines(tool, action_input)
    lines.extend(approval_scope_lines(tool, action_input, allowed_scopes))

    if lines:
        _console.print()
        _console.print(
            Panel.fit(
                "\n".join(lines), title="Authorization Request", border_style="yellow"
            )
        )
        _console.print()


def tool_preview_lines(tool: ToolSpec, action_input: ToolInput) -> list[str]:
    """生成供 REPL 和 TUI 共用的授权预览内容。"""
    lines: list[str] = []

    # Tool name
    lines.append(f"[bold]Tool:[/bold] {tool.name}")

    # ToolInput is always dict[str, Any]
    if tool.name == "edit_file":
        _preview_edit_file(action_input, lines)
    elif tool.name == "bash":
        _preview_bash(action_input, lines)
    elif tool.name == "write_file":
        _preview_write_file(action_input, lines)
    elif tool.name == "read_file":
        path = action_input.get("path", "")
        lines.append(f"[bold]File:[/bold] {path}")
    elif tool.name in ("grep_search", "glob_files", "find_files"):
        _preview_search(tool.name, action_input, lines)
    else:
        brief = brief_input(tool.name, action_input)
        lines.append(f"[bold]Input:[/bold] {brief}")

    return lines


def approval_scope_lines(
    tool: ToolSpec,
    action_input: ToolInput,
    allowed_scopes: tuple[ApprovalScope, ...],
) -> list[str]:
    """说明 session/permanent 选项实际会授予的范围。"""
    if not {"session", "permanent"}.intersection(allowed_scopes):
        return []

    if tool.name == "bash":
        command = action_input.get("command") or action_input.get("input", "")
        subject = (
            f"bash commands matching [bold]{command_grant_pattern(command)}[/bold]"
            if command
            else "this bash action"
        )
    else:
        subject = f"this [bold]{tool.name}[/bold] action"

    lines: list[str] = []
    if "session" in allowed_scopes:
        lines.append(f"[dim]Session allow: {subject} until this session ends.[/dim]")
    if "permanent" in allowed_scopes:
        lines.append(
            "[yellow]Always allow: save this rule for this project in "
            ".xcode/approval_grants.json.[/yellow]"
        )
    return lines


def _preview_edit_file(action_input: dict, lines: list[str]) -> None:
    """为 edit_file 工具显示文件路径和 mini diff。"""
    path = action_input.get("path", "")
    old_text = action_input.get("old_text", "")
    new_text = action_input.get("new_text", "")
    replace_all = action_input.get("replace_all", False)

    lines.append(f"[bold]File:[/bold] {path}")
    if replace_all:
        lines.append("[yellow]Replace all occurrences[/yellow]")

    if old_text or new_text:
        # Show compact diff snippet
        old_snippet = old_text[:280].replace("\n", "¶ ")
        new_snippet = new_text[:280].replace("\n", "¶ ")
        if old_snippet:
            lines.append(f"[red]─ {old_snippet}[/red]")
        if new_snippet:
            lines.append(f"[green]+ {new_snippet}[/green]")

        if len(old_text) > 280 or len(new_text) > 280:
            lines.append("[dim]  (truncated, see above for full)[/dim]")


def _preview_bash(action_input: dict, lines: list[str]) -> None:
    """为 bash 工具显示命令和结构化分析。"""
    command = action_input.get("command") or action_input.get("input", "")
    if not command:
        lines.append("[bold]Command:[/bold] (empty)")
        return

    lines.append(f"[bold]Command:[/bold] {command[:500]}")

    parts = command.strip().split()
    if not parts:
        return

    cmd = parts[0].lower()

    # 危险 flag 检测
    dangerous_flags = {"--force", "-f", "--hard", "--destroy", "--delete"}
    flags = {p for p in parts[1:] if p.startswith("-")}
    matched_danger = flags & dangerous_flags
    if matched_danger:
        lines.append(f"[red]⚠  Danger flags: {' '.join(matched_danger)}[/red]")

    # 命令解释
    interpretations = {
        "rm": "Remove files/directories",
        "cp": "Copy files",
        "mv": "Move/rename files",
        "git": "Git operation",
        "docker": "Docker container operation",
        "pip": "Python package management",
        "npm": "Node package management",
        "curl": "HTTP request",
        "wget": "Download file",
        "chmod": "Change file permissions",
        "chown": "Change file owner",
        "mkdir": "Create directory",
        "touch": "Create file / update timestamp",
        "cat": "Display file contents",
        "echo": "Output text",
        "grep": "Search text patterns",
        "find": "Search files",
        "sed": "Stream editor (file modification)",
        "awk": "Text processing",
        "sort": "Sort lines",
        "head": "Show first lines of file",
        "tail": "Show last lines of file",
        "wc": "Count lines/words/bytes",
        "ls": "List directory contents",
        "ps": "Show process status",
        "kill": "Terminate a process",
        "systemctl": "Systemd service management",
        "apt": "APT package manager (Debian/Ubuntu)",
        "yum": "YUM package manager (RHEL/CentOS)",
        "brew": "Homebrew package manager (macOS)",
        "python": "Run Python interpreter/script",
        "python3": "Run Python 3 interpreter/script",
        "node": "Run Node.js script",
        "npx": "Execute Node package",
        "deno": "Run Deno script",
        "cargo": "Rust package manager",
        "go": "Go toolchain",
        "rustc": "Rust compiler",
        "dotnet": ".NET CLI",
        " terraform": "Terraform infrastructure",
        "kubectl": "Kubernetes CLI",
        "helm": "Kubernetes package manager",
        "ssh": "SSH remote connection",
        "scp": "Secure copy over SSH",
        "rsync": "Remote file sync",
        "make": "Build automation",
        "cmake": "CMake build system",
        "gradle": "Gradle build tool",
        "mvn": "Maven build tool",
    }
    if cmd in interpretations:
        lines.append(f"[dim]• {interpretations[cmd]}[/dim]")
    else:
        # 未知命令标记
        lines.append(f"[dim]• Command: [italic]{cmd}[/italic][/dim]")

    # 文件路径提取
    file_paths = [
        p
        for p in parts[1:]
        if (p.startswith(("./", "/", "~")) or "." in p or "\\" in p)
        and not p.startswith("-")
    ]
    if file_paths:
        paths_str = ", ".join(file_paths[:5])
        if len(file_paths) > 5:
            paths_str += f" … and {len(file_paths) - 5} more"
        lines.append(f"[dim]• Paths: {paths_str}[/dim]")


def _preview_write_file(action_input: dict, lines: list[str]) -> None:
    """为 write_file 工具显示路径和内容预览。"""
    path = action_input.get("path", "")
    content = action_input.get("content", "")
    lines.append(f"[bold]File:[/bold] {path}")
    if content:
        # Show first/last lines of content
        content_lines = content.split("\n")
        display_lines: list[str] = []
        for cl in content_lines[:10]:
            display_lines.append(f"  {cl[:200]}")
        if len(content_lines) > 10:
            display_lines.append(f"  … ({len(content_lines) - 10} more lines)")
        lines.append("[bold]Content:[/bold]")
        lines.extend(display_lines)


def _preview_search(tool_name: str, action_input: dict, lines: list[str]) -> None:
    """为搜索类工具显示查询信息和路径。"""
    if tool_name == "grep_search":
        pattern = (
            action_input.get("pattern")
            or action_input.get("query")
            or action_input.get("input", "")
        )
        path = action_input.get("path") or action_input.get("include", "workspace")
        lines.append(f"[bold]Pattern:[/bold] {pattern[:200]}")
        lines.append(f"[bold]Search in:[/bold] {path}")
    elif tool_name == "glob_files" or tool_name == "find_files":
        pattern = action_input.get("pattern") or action_input.get("path", "")
        lines.append(f"[bold]Pattern:[/bold] {pattern[:200]}")


# ── Timeout-aware HITL prompt ──


def _ask_hitl_choice_with_timeout(
    tool: ToolSpec,
    action_input: ToolInput,
    choices: tuple[str, ...],
    timeout: float,
) -> str | None:
    """显示 questionary 授权选择，支持超时自动 deny。"""
    results: Queue[str | None | BaseException] = Queue(maxsize=1)

    def run_prompt() -> None:
        try:
            results.put(_show_select_prompt(tool, action_input, choices))
        except (
            EOFError,
            KeyboardInterrupt,
            OSError,
            RuntimeError,
            SystemExit,
            TypeError,
            ValueError,
        ) as exc:
            results.put(exc)

    thread = threading.Thread(target=run_prompt, name="xcode-hitl-prompt", daemon=True)
    thread.start()

    try:
        result = results.get(timeout=timeout)
    except Empty:
        _console.print(
            "[yellow]⚠  Authorization request timed out, auto-denied.[/yellow]"
        )
        return None

    if isinstance(result, BaseException):
        raise result
    return result


def _show_select_prompt(
    tool: ToolSpec, action_input: ToolInput, choices: tuple[str, ...]
) -> str | None:
    """显示 questionary 授权选择界面。"""
    brief = brief_input(tool.name, action_input)
    return questionary.select(
        f"Authorization required: {tool.name}\nInput: {brief}",
        choices=choices,
    ).ask()


def _ask_suggestion_with_timeout(timeout: float) -> str:
    """当用户选择 Deny 时，可选的引导建议（超时返回空字符串）。"""
    results: Queue[str | BaseException] = Queue(maxsize=1)

    def run_prompt() -> None:
        try:
            result = questionary.text(
                "Tell model what to do:",
                default="",
                instruction="(press Enter to skip)",
            ).ask()
            results.put(result or "")
        except (
            EOFError,
            KeyboardInterrupt,
            OSError,
            RuntimeError,
            SystemExit,
            TypeError,
            ValueError,
        ) as exc:
            results.put(exc)

    thread = threading.Thread(
        target=run_prompt, name="xcode-hitl-suggestion", daemon=True
    )
    thread.start()

    try:
        result = results.get(timeout=timeout)
    except Empty:
        return ""

    if isinstance(result, BaseException):
        return ""
    return result or ""
