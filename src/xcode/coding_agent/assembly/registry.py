"""工具注册表构建。"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xcode.agent.types import ToolSpec
from xcode.ai.providers.base import ModelProvider
from xcode.coding_agent.registry import build_project_scoped_registry
from xcode.coding_agent.tools import ShellSpec, detect_shell
from xcode.coding_agent.tools.subagent import (
    BUILD_SUBAGENT_PROMPTS,
    build_subagent_tools,
)
from xcode.harness.agent_runtime import CancellationToken, ContextualRetrievalState
from xcode.harness.agent_runtime.context_window import (
    ContextWindowController,
    build_new_context_tool,
)
from xcode.harness.agent_runtime.subagents import SubagentSessionManager
from xcode.harness.config import XcodeRuntimeConfig
from xcode.harness.execution_env import Shell
from xcode.harness.session_todo import SessionTodoState

from .security import build_shell_from_security

if TYPE_CHECKING:
    from xcode.harness.mcp import McpRuntimeRegistry
    from xcode.harness.session.recorder import SessionRecorder
    from xcode.harness.skills import SkillRegistry


def build_search_tools_tool(
    registry_provider: Callable[[], tuple[ToolSpec, ...]],
) -> ToolSpec:
    """按关键字搜索所有已注册工具。"""

    def search_tools(
        data: dict[str, Any], _on_update: Callable[[str], None] | None = None
    ) -> str:
        registry = registry_provider()
        query = str(data.get("query", "")).strip().lower()
        if not query:
            lines = [f"Available tools ({len(registry)}):"]
            for t in sorted(registry, key=lambda x: x.name):
                lines.append(f"  {t.name}: {t.description[:80]}")
            return "\n".join(lines)
        results = []
        for t in registry:
            if query in t.name.lower() or query in t.description.lower():
                schema_str = json.dumps(t.schema or {}, ensure_ascii=False)[:200]
                results.append(
                    f"{t.name}:\n  description: {t.description[:200]}\n  schema: {schema_str}"
                )
        if not results:
            return f"No tools matching '{query}'."
        return f"Found {len(results)} tool(s) matching '{query}':\n" + "\n\n".join(
            results[:5]
        )

    return ToolSpec(
        name="search_tools",
        description="Search available tools by keyword. Returns tool descriptions and schemas matching the query.",
        input_hint='JSON: {"query": "file"}',
        handler=search_tools,
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword to search for in tool names and descriptions",
                }
            },
            "additionalProperties": False,
        },
    )


def _discover_skills(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
    skills_dir: Path | None,
) -> SkillRegistry | None:
    from xcode.harness.skills import SkillRegistry, build_skill_search_dirs

    skill_registry = SkillRegistry()
    skill_registry.discover(
        build_skill_search_dirs(
            project_root,
            trust_project_skills=runtime_config.skills.trust_project_skills,
            skills_dir=skills_dir,
        )
    )
    return skill_registry


def _build_base_project_registry(
    project_root: Path,
    shell_spec: ShellSpec,
    cancel_event: threading.Event | None,
    shell: Shell | None,
    skill_registry: SkillRegistry | None,
    contextual_state: ContextualRetrievalState | None = None,
    todo_state: SessionTodoState | None = None,
) -> tuple[ToolSpec, ...]:
    return build_project_scoped_registry(
        project_root=project_root,
        contextual_state=contextual_state,
        shell_spec=shell_spec,
        cancel_event=cancel_event,
        shell=shell,
        skill_registry=skill_registry,
        todo_state=todo_state,
    )


def _build_child_registry(
    registry: tuple[ToolSpec, ...],
    subagent_extra_tools: set[str],
) -> tuple[ToolSpec, ...]:
    CORE_TOOLS = frozenset(
        {
            "read_file",
            "write_file",
            "edit_file",
            "apply_patch",
            "glob_files",
            "find_files",
            "list_dir",
            "grep_search",
            "webfetch",
            "websearch",
            "bash",
        }
    )
    allowed_tools = CORE_TOOLS | subagent_extra_tools
    return tuple(tool for tool in registry if tool.name in allowed_tools)


def _extend_registry_with_features(
    registry: tuple[ToolSpec, ...],
    project_root: Path,
    mcp_runtime_registry: McpRuntimeRegistry,
    runtime_config: XcodeRuntimeConfig,
    memory_manager: Any | None = None,
    session_history: Any | None = None,
    context_window_controller: ContextWindowController | None = None,
) -> tuple[ToolSpec, ...]:
    from xcode.harness.mcp import build_mcp_tools

    registry += build_mcp_tools(project_root, mcp_runtime_registry)

    from xcode.harness.memory import MemoryManager, build_memory_tools

    if memory_manager is not None:
        registry += build_memory_tools(memory_manager)
    else:
        registry += build_memory_tools(MemoryManager(project_root))
    if session_history is not None:
        from xcode.harness.session import build_history_tools

        registry += build_history_tools(session_history)
    if context_window_controller is not None:
        registry += build_new_context_tool(context_window_controller, project_root)
    return registry


def build_tool_registry(
    project_root: Path,
    subagent_provider: ModelProvider,
    runtime_config: XcodeRuntimeConfig,
    session_recorder: SessionRecorder,
    contextual_state: ContextualRetrievalState | None = None,
    cancel_event: CancellationToken | None = None,
    shell: Shell | None = None,
    skills_dir: Path | None = None,
    memory_manager: Any | None = None,
    session_history: Any | None = None,
    todo_state: SessionTodoState | None = None,
    context_window_controller: ContextWindowController | None = None,
) -> tuple[
    tuple[ToolSpec, ...],
    ShellSpec,
    tuple[Callable[[], None], ...],
    SkillRegistry | None,
    McpRuntimeRegistry,
    SubagentSessionManager,
]:
    from xcode.harness.mcp import McpRuntimeRegistry

    closers: list[Callable[[], None]] = []
    shell_spec = detect_shell(runtime_config.tools.shell)
    shell = shell or build_shell_from_security(project_root, runtime_config.security)

    skill_registry = _discover_skills(project_root, runtime_config, skills_dir)

    registry = _build_base_project_registry(
        project_root,
        shell_spec,
        cancel_event,
        shell,
        skill_registry,
        contextual_state=contextual_state,
        todo_state=todo_state,
    )
    mcp_runtime_registry = McpRuntimeRegistry()
    mcp_runtime_registry.configure_runtime(
        workspace_roots=(project_root,),
        cancel_event=cancel_event,
    )
    registry = _extend_registry_with_features(
        registry,
        project_root,
        mcp_runtime_registry,
        runtime_config,
        memory_manager=memory_manager,
        session_history=session_history,
        context_window_controller=context_window_controller,
    )

    child_registry = _build_child_registry(
        registry,
        set(runtime_config.tools.subagent_extra_tools),
    )
    registry += (build_search_tools_tool(lambda: registry),)

    subagents = SubagentSessionManager(
        provider=subagent_provider,
        coding_tools=child_registry,
        research_tools=child_registry,
        system_prompts=BUILD_SUBAGENT_PROMPTS,
        parent_store=session_recorder.store,
    )
    registry += build_subagent_tools(subagents)

    closers.append(subagents.close)
    closers.append(mcp_runtime_registry.close)

    return (
        registry,
        shell_spec,
        tuple(closers),
        skill_registry,
        mcp_runtime_registry,
        subagents,
    )
