"""工具注册表构建。"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING

from xcode.ai.providers.base import ModelProvider
from xcode.agent.types import ToolSpec
from xcode.coding_agent.tools.subagent import build_subagent_tool
from xcode.coding_agent.tools import ShellSpec, detect_shell
from xcode.coding_agent.registry import build_project_scoped_registry

from xcode.harness.config import XcodeRuntimeConfig, AgentConfig
from xcode.harness.execution_env import ExecutionEnv
from xcode.harness.agent_runtime import CancellationToken, ContextualRetrievalState
from xcode.harness.session_todo import SessionTodoState
from xcode.harness.security import PolicyEvaluator
from xcode.harness.observability import ExternalHookRunner

if TYPE_CHECKING:
    from xcode.harness.skills import SkillRegistry
    from xcode.harness.mcp import McpRuntimeRegistry


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
    env: ExecutionEnv | None,
    skill_registry: SkillRegistry | None,
    contextual_state: ContextualRetrievalState | None = None,
    todo_state: SessionTodoState | None = None,
) -> tuple[ToolSpec, ...]:
    return build_project_scoped_registry(
        project_root=project_root,
        contextual_state=contextual_state,
        shell_spec=shell_spec,
        cancel_event=cancel_event,
        env=env,
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
) -> tuple[ToolSpec, ...]:
    from xcode.harness.mcp import build_mcp_tools

    registry += build_mcp_tools(project_root, mcp_runtime_registry)

    from xcode.harness.memory import MemoryManager, build_memory_tools

    if memory_manager is not None:
        registry += build_memory_tools(memory_manager)
    else:
        registry += build_memory_tools(MemoryManager(project_root))
    return registry


def build_tool_registry(
    project_root: Path,
    llm: ModelProvider,
    llm_profiles: Mapping[str, ModelProvider] | None,
    config: AgentConfig,
    runtime_config: XcodeRuntimeConfig,
    contextual_state: ContextualRetrievalState | None = None,
    compact_controller: Any = None,
    cancel_event: CancellationToken | None = None,
    env: ExecutionEnv | None = None,
    skills_dir: Path | None = None,
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
    external_hook_runner: ExternalHookRunner | None = None,
    memory_manager: Any | None = None,
    todo_state: SessionTodoState | None = None,
) -> tuple[
    tuple[ToolSpec, ...],
    ShellSpec,
    tuple[Callable[[], None], ...],
    SkillRegistry | None,
    McpRuntimeRegistry,
]:
    from xcode.harness.mcp import McpRuntimeRegistry

    closers: list[Callable[[], None]] = []
    shell_spec = detect_shell(runtime_config.tools.shell)

    skill_registry = _discover_skills(project_root, runtime_config, skills_dir)

    registry = _build_base_project_registry(
        project_root,
        shell_spec,
        cancel_event,
        env,
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
    )

    child_registry = _build_child_registry(
        registry,
        set(runtime_config.tools.subagent_extra_tools),
    )
    registry += (build_search_tools_tool(lambda: registry),)

    registry += (
        build_subagent_tool(
            model=llm,
            coding_tools=list(child_registry),
            research_tools=list(child_registry),
            cancellation_token=cancel_event,
        ),
    )

    closers.append(mcp_runtime_registry.close)

    return (
        registry,
        shell_spec,
        tuple(closers),
        skill_registry,
        mcp_runtime_registry,
    )
