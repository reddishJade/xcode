from __future__ import annotations

import logging

from collections.abc import Callable
from dataclasses import dataclass
import platform
from pathlib import Path
from typing import Any

from .git_preflight import build_git_preflight
from .contextual import ContextualRetrievalState
from ...experimental.tasks import TaskStore
from ...harness.skill_loader import SkillLoader
from ..skills import ToolSpec, build_tool_prompt
from ...harness.tools.shell_adapter import ShellSpec


CORE_IDENTITY = (
    "You are Xcode, a lightweight coding agent running in a local terminal. "
    "Use tools deliberately, respect the project sandbox, and keep answers grounded in observed results."
)

MAX_CWD_ENTRIES = 12
MAX_INSTRUCTION_CHARS = 4000

TOOL_DISCIPLINE = """<tool-discipline>
Tools must serve the current response. If no external facts or workspace evidence
are needed — simple greetings, capability questions, conceptual explanations,
general knowledge — answer directly without any tool calls.
Code changes, file operations, diagnostics, validation, and command execution
require tools. The <git-preflight> block is already injected; do not manually
repeat git status/diff commands unless the user asks or the task specifically
requires a fresh check.
</tool-discipline>"""

SEARCH_STRATEGY = """<search-strategy>
Code tasks use the following retrieval layers in order:
1. lexical search: use glob_files for file/path discovery, grep_search for exact text, and read_file for known files.
2. contextual retrieval: use explicit @file context, git preflight, recent files, and recent tool summaries only as task orientation.
</search-strategy>"""

SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "<system-prompt-dynamic-boundary />"


@dataclass(frozen=True)
class PromptContext:
    project_root: Path
    registry: tuple[ToolSpec, ...]
    question: str
    skill_loader: SkillLoader | None = None
    resumed_notice: str | None = None
    interrupted_notice: str | None = None
    contextual_state: ContextualRetrievalState | None = None
    shell_spec: ShellSpec | None = None
    modules: tuple[str, ...] = (
        "identity",
        "tool_discipline",
        "tools",
        "environment",
        "git_preflight",
        "search_strategy",
        "contextual_retrieval",
        "cwd",
        "instructions",
        "skills",
        "notices",
    )


class SystemPromptBuilder:
    """根据稳定模块、动态模块和易失模块构造每轮 system prompt，最大化 Prompt Cache 命中率。"""

    def __init__(self) -> None:
        self._stable_cache: str | None = None
        self._stable_key: tuple[Any, ...] | None = None

        self._dynamic_cache: str | None = None
        self._dynamic_key: tuple[Any, ...] | None = None

    def build(self, context: PromptContext) -> str:
        enabled = set(context.modules)
        stable_prompt = self._build_stable_region(context, enabled)
        dynamic_prompt = self._build_dynamic_region(context, enabled)
        volatile_parts = _build_volatile_region(context, enabled)

        full_parts = []
        if stable_prompt.strip():
            full_parts.append(stable_prompt)
            if dynamic_prompt.strip() or volatile_parts:
                full_parts.append(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        if dynamic_prompt.strip():
            full_parts.append(dynamic_prompt)
        if volatile_parts:
            full_parts.append("\n\n".join(volatile_parts))

        return "\n\n".join(part for part in full_parts if part.strip())

    def _build_stable_region(self, context: PromptContext, enabled: set[str]) -> str:
        try:
            mtimes = tuple(
                (context.project_root / name).stat().st_mtime
                for name in ("AGENTS.md", "CLAUDE.md")
                if (context.project_root / name).is_file()
            )
        except Exception:
            mtimes = ()

        stable_enabled = enabled.intersection(
            {"identity", "tool_discipline", "tools", "search_strategy", "instructions"}
        )
        stable_key = (
            tuple(
                (
                    t.name,
                    t.description,
                    t.input_hint,
                    t.risk,
                    tuple(tuple(sorted(example.items())) for example in t.examples),
                )
                for t in context.registry
            ),
            mtimes,
            frozenset(stable_enabled),
        )

        if self._stable_cache is not None and self._stable_key == stable_key:
            return self._stable_cache

        stable_parts: list[str] = []
        if "identity" in enabled:
            stable_parts.append(CORE_IDENTITY)
        if "tool_discipline" in enabled:
            stable_parts.append(TOOL_DISCIPLINE)
        if "tools" in enabled:
            stable_parts.append(
                "Available tools:\n" + build_tool_prompt(context.registry)
            )
        if "search_strategy" in enabled:
            stable_parts.append(SEARCH_STRATEGY)
        if "instructions" in enabled:
            instructions = _project_instructions(context.project_root)
            if instructions:
                stable_parts.append(instructions)

        stable_prompt = "\n\n".join(stable_parts)
        self._stable_cache = stable_prompt
        self._stable_key = stable_key
        return stable_prompt

    def _build_dynamic_region(self, context: PromptContext, enabled: set[str]) -> str:
        dynamic_enabled = enabled.intersection({"environment", "cwd"})
        dynamic_key = (
            context.project_root,
            context.shell_spec.name if context.shell_spec else None,
            frozenset(dynamic_enabled),
        )

        if self._dynamic_cache is not None and self._dynamic_key == dynamic_key:
            return self._dynamic_cache

        dynamic_parts: list[str] = []
        if "environment" in enabled:
            dynamic_parts.append(
                _environment_info(context.project_root, context.shell_spec)
            )
        if "cwd" in enabled:
            dynamic_parts.append(_cwd_info(context.project_root))

        dynamic_prompt = "\n\n".join(dynamic_parts)
        self._dynamic_cache = dynamic_prompt
        self._dynamic_key = dynamic_key
        return dynamic_prompt


def _build_volatile_region(context: PromptContext, enabled: set[str]) -> list[str]:
    volatile_parts: list[str] = []
    if "git_preflight" in enabled:
        volatile_parts.append(build_git_preflight(context.project_root))
    if "contextual_retrieval" in enabled and context.contextual_state is not None:
        rendered = context.contextual_state.render()
        if rendered.strip():
            volatile_parts.append(rendered)
    if "skills" in enabled and context.skill_loader is not None:
        volatile_parts.append(
            context.skill_loader.get_catalog(question=context.question)
        )
    if "notices" in enabled:
        notices = [context.resumed_notice, context.interrupted_notice]
        notice_text = "\n".join(notice for notice in notices if notice)
        if notice_text:
            volatile_parts.append(
                "<session-notices>\n" + notice_text + "\n</session-notices>"
            )

    metadata_parts = _build_post_compact_metadata(context.project_root)
    if metadata_parts:
        volatile_parts.append(
            "<post-compact-metadata>\n"
            + "\n\n".join(metadata_parts)
            + "\n</post-compact-metadata>"
        )

    return volatile_parts


def _build_post_compact_metadata(project_root: Path) -> list[str]:
    active_tasks = []
    try:
        store = TaskStore(project_root)
        for task in store.list():
            if task.status in ("pending", "claimed"):
                task_info = f"- [{task.status.upper()}] Task #{task.id}: {task.title}"
                fl = task.payload.get("feature_list")
                if isinstance(fl, list) and fl:
                    completed = sum(
                        1
                        for item in fl
                        if isinstance(item, dict) and item.get("status") == "completed"
                    )
                    task_info += f" ({completed}/{len(fl)} subtasks completed)"
                blocked_by = task.payload.get("blocked_by")
                if blocked_by:
                    task_info += f" [Blocked by: {blocked_by}]"
                active_tasks.append(task_info)
    except Exception:
        logging.warning("failed to build active-tasks graph", exc_info=True)

    if not active_tasks:
        return []
    return [
        "<active-tasks-graph>\n" + "\n".join(active_tasks) + "\n</active-tasks-graph>"
    ]


def build_runtime_context_provider(
    project_root: Path,
    registry: tuple[ToolSpec, ...],
    skill_loader: SkillLoader | None = None,
    prompt_builder: SystemPromptBuilder | None = None,
    resumed_notice: Callable[[], str | None] | None = None,
    interrupted_notice: Callable[[], str | None] | None = None,
    contextual_state: ContextualRetrievalState | None = None,
    modules: tuple[str, ...] | None = None,
    shell_spec: ShellSpec | None = None,
) -> Callable[[str], list[str]]:
    builder = prompt_builder or SystemPromptBuilder()
    root = project_root.resolve()

    def provide(question: str) -> list[str]:
        return [
            builder.build(
                PromptContext(
                    project_root=root,
                    registry=registry,
                    question=question,
                    skill_loader=skill_loader,
                    resumed_notice=resumed_notice() if resumed_notice else None,
                    interrupted_notice=interrupted_notice()
                    if interrupted_notice
                    else None,
                    contextual_state=contextual_state,
                    modules=modules
                    or PromptContext(
                        project_root=root, registry=(), question=""
                    ).modules,
                    shell_spec=shell_spec,
                )
            )
        ]

    return provide


def _environment_info(project_root: Path, shell_spec: ShellSpec | None = None) -> str:
    lines = [
        "<environment>",
        f"os={platform.system()} {platform.release()}",
        f"python={platform.python_version()}",
        f"cwd={project_root.resolve()}",
    ]
    if shell_spec is not None:
        lines.append(
            f'<shell tool="bash" name="{shell_spec.name}" syntax="{shell_spec.syntax}" />'
        )
        lines.append(
            "When using the bash tool, write commands for the shell named in <shell> above. "
            "Do not probe for shell availability unless asked."
        )
    lines.append("</environment>")
    return "\n".join(lines)


def _cwd_info(project_root: Path) -> str:
    names = []
    for path in sorted(project_root.iterdir()):
        if path.name in {".git", ".venv", "__pycache__"}:
            continue
        names.append(path.name + ("/" if path.is_dir() else ""))
        if len(names) >= MAX_CWD_ENTRIES:
            break
    return "<cwd-info>\n" + "\n".join(names) + "\n</cwd-info>"


def _project_instructions(project_root: Path) -> str:
    parts = []
    for name in ("AGENTS.md", "CLAUDE.md"):
        path = project_root / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append(f"<{name}>\n{text[:MAX_INSTRUCTION_CHARS]}\n</{name}>")
    return "\n\n".join(parts)
