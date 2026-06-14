"""应用装配工厂函数。

从 app.py 提取的配置解析、共享基础设施构建、provider 组装、
工具注册、agent 构建和可选服务加载逻辑。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from xcode.harness.execution_env import ExecutionEnv

from xcode.harness.config import (
    AgentConfig,
    PROFILE_MAIN,
    PROFILE_SUBAGENT,
    SecurityRuntimeConfig,
    XcodeRuntimeConfig,
    discover_runtime_config,
    resolve_config_path,
)
from xcode.harness.agent_runtime import (
    CancellationToken,
    ContextualRetrievalState,
    ManagedSubagentRunner,
    StructuredAgent,
    build_managed_subagent_tools,
)
from xcode.harness.agent_runtime.config import AgentRuntimeConfig, GateConfig
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.harness.agent_runtime.compaction import CompactController, LayeredCompactor
from xcode.ai.providers.protocol import ModelProvider
from xcode.harness.observability import (
    JsonlAuditLogger,
    HookManager,
    PermissionPolicy,
    PermissionRule,
)
from xcode.harness.observability.permission_model import PolicyEvaluator
from xcode.harness.observability.hooks import HookEvent
from xcode.harness.skills import ToolInput, ToolSpec
from xcode.harness.skill_loader import SkillLoader, build_skill_loader_tool
from xcode.coding_agent.registry import build_project_scoped_registry
from xcode.coding_agent.tools import ShellSpec
from xcode.ai.providers.factory import (
    ProviderSettings,
    build_provider_bundle,
)

if TYPE_CHECKING:
    from xcode.harness.daemon import HeartbeatDaemon
    from xcode.harness.mailbox import AgentMailbox

EXPERIMENTAL_FEATURE_GROUPS = frozenset(
    {
        "mcp",
        "memory",
        "plugins",
    }
)


@dataclass(frozen=True)
class OptInServices:
    daemon: HeartbeatDaemon | None = None
    mailbox: AgentMailbox | None = None
    progress: bool | None = None


@dataclass(frozen=True)
class ResolvedConfig:
    runtime_config: XcodeRuntimeConfig
    agent_config: AgentConfig
    skills_dir: Path | None
    audit_path: Path | None
    env_files: tuple[Path, ...]


@dataclass(frozen=True)
class SharedInfra:
    contextual_state: ContextualRetrievalState
    cancellation_token: CancellationToken
    compact_controller: CompactController
    compactor: LayeredCompactor


# ── 配置解析 ──


def resolve_config(
    project_root: Path,
    env_files: tuple[Path, ...] | None,
    agent_config: AgentConfig | None,
    skills_dir: Path | None,
    audit_path: Path | None,
    runtime_config: XcodeRuntimeConfig | None,
) -> ResolvedConfig:
    runtime_config = runtime_config or discover_runtime_config(project_root)
    agent_config = agent_config or runtime_config.agent
    skills_dir = skills_dir or resolve_config_path(
        project_root, runtime_config.paths.skills_dir
    )
    audit_path = audit_path or resolve_config_path(
        project_root, runtime_config.observability.audit_path
    )
    _pkg_root = Path(__file__).resolve().parent.parent
    env_files = env_files or (
        _pkg_root / ".env",
        project_root / ".env",
        project_root / "xcode" / ".env",
    )
    return ResolvedConfig(
        runtime_config=runtime_config,
        agent_config=agent_config,
        skills_dir=skills_dir,
        audit_path=audit_path,
        env_files=env_files,
    )


def effective_enabled_groups(configured_groups: tuple[str, ...]) -> set[str]:
    enabled = set(configured_groups)
    if "experimental" in enabled:
        enabled.update(EXPERIMENTAL_FEATURE_GROUPS)
    return enabled


# ── 共享基础设施 ──


def build_shared_infra(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
    enabled: set[str],
) -> SharedInfra:
    contextual_state = ContextualRetrievalState(project_root)
    cancellation_token = CancellationToken()
    compact_controller = CompactController()

    transcript_dir = (
        resolve_config_path(project_root, runtime_config.paths.sessions_dir)
        if runtime_config.paths.sessions_dir
        else project_root / ".local" / "sessions"
    )
    on_compact = None
    if "memory" in enabled:
        from xcode.experimental.memory import MemoryManager

        on_compact = MemoryManager(project_root).consolidate

    compactor = LayeredCompactor(
        transcript_dir=transcript_dir,
        max_recent_messages=runtime_config.agent.max_recent_messages,
        on_compact=on_compact,
    )
    return SharedInfra(
        contextual_state=contextual_state,
        cancellation_token=cancellation_token,
        compact_controller=compact_controller,
        compactor=compactor,
    )


# ── Provider ──


def build_providers(runtime_config: XcodeRuntimeConfig, env_files: tuple[Path, ...]):
    return build_provider_bundle(
        ProviderSettings(
            env_files=env_files,
            model_profiles=runtime_config.provider.model_profiles,
        )
    )


# ── 工具注册 ──


def build_search_tools_tool(
    registry: tuple[ToolSpec, ...],
) -> ToolSpec:
    """按关键字搜索所有已注册工具。"""

    def search_tools(data: ToolInput) -> str:
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
        group="core",
        read_only=True,
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


def build_tool_registry(
    project_root: Path,
    llm: ModelProvider,
    llm_profiles: Mapping[str, ModelProvider] | None,
    config: AgentConfig,
    runtime_config: XcodeRuntimeConfig,
    skills_dir: Path | None,
    contextual_state: ContextualRetrievalState | None = None,
    compact_controller: CompactController | None = None,
    cancel_event: threading.Event | None = None,
    env: ExecutionEnv | None = None,
) -> tuple[
    tuple[ToolSpec, ...], SkillLoader | None, ShellSpec, tuple[Callable[[], None], ...]
]:
    from xcode.coding_agent.tools import detect_shell

    enabled = effective_enabled_groups(runtime_config.tools.enabled_groups)
    closers: list[Callable[[], None]] = []
    shell_spec = detect_shell(runtime_config.tools.shell)
    skills_dir = skills_dir or project_root / ".agents" / "skills"
    skill_loader = None
    if "skills" in enabled and skills_dir.exists():
        skill_loader = SkillLoader(skills_dir)

    registry = build_project_scoped_registry(
        project_root=project_root,
        enabled=enabled,
        contextual_state=contextual_state,
        shell_spec=shell_spec,
        cancel_event=cancel_event,
        env=env,
    )
    registry = _extend_registry_with_features(
        registry, project_root, enabled, skill_loader
    )

    child_registry = registry
    registry += (build_search_tools_tool(registry),)

    subagent_closers, subagent_tools = _build_subagent_integration(
        project_root=project_root,
        llm=llm,
        llm_profiles=llm_profiles,
        config=config,
        runtime_config=runtime_config,
        enabled=enabled,
        child_registry=child_registry,
        contextual_state=contextual_state,
        skill_loader=skill_loader,
        shell_spec=shell_spec,
        cancel_event=cancel_event,
        env=env,
    )
    closers.extend(subagent_closers)
    registry += subagent_tools
    return registry, skill_loader, shell_spec, tuple(closers)


def _extend_registry_with_features(
    registry: tuple[ToolSpec, ...],
    project_root: Path,
    enabled: set[str],
    skill_loader: SkillLoader | None,
) -> tuple[ToolSpec, ...]:
    """添加可选功能工具到注册表。"""
    if "worktree" in enabled:
        from xcode.coding_agent.tools.worktree import (
            WorktreeTaskRunner,
            build_worktree_tools,
        )

        registry += build_worktree_tools(WorktreeTaskRunner(project_root))
    if "mcp" in enabled:
        from xcode.experimental.mcp import build_mcp_tools

        registry += build_mcp_tools(project_root)
    if "tasks" in enabled:
        from xcode.harness.task_store import TaskStore, build_task_tools

        registry += build_task_tools(TaskStore(project_root))
    if "mailbox" in enabled:
        from xcode.harness.mailbox import AgentMailbox, build_mailbox_tools

        registry += build_mailbox_tools(AgentMailbox(project_root))
    if "progress" in enabled:
        from xcode.harness.task_progress import build_progress_tools
        from xcode.harness.task_store import TaskStore

        registry += build_progress_tools(TaskStore(project_root))
    if skill_loader is not None:
        registry += (build_skill_loader_tool(skill_loader),)
    return registry


def _build_subagent_integration(
    project_root: Path,
    llm: ModelProvider,
    llm_profiles: Mapping[str, ModelProvider] | None,
    config: AgentConfig,
    runtime_config: XcodeRuntimeConfig,
    enabled: set[str],
    child_registry: tuple[ToolSpec, ...],
    contextual_state: ContextualRetrievalState | None,
    skill_loader: SkillLoader | None,
    shell_spec: ShellSpec,
    cancel_event: threading.Event | None,
    env: ExecutionEnv | None,
) -> tuple[list[Callable[[], None]], tuple[ToolSpec, ...]]:
    """构建子代理运行器和工具，返回 (closers, subagent_tools)。"""
    if "subagent" not in enabled:
        return [], ()

    child_llms = dict(llm_profiles or {})
    if not child_llms:
        child_llms[PROFILE_MAIN] = llm
    child_llms.setdefault(PROFILE_SUBAGENT, child_llms[PROFILE_MAIN])

    async def run_child(prompt, model_profile=PROFILE_SUBAGENT, cwd_override=None):
        child_root = project_root.resolve()
        child_contextual_state = contextual_state
        effective_registry = child_registry
        if cwd_override is not None:
            child_root = Path(cwd_override).resolve()
            child_contextual_state = ContextualRetrievalState(child_root)
            effective_registry = build_project_scoped_registry(
                project_root=child_root,
                enabled=enabled,
                contextual_state=child_contextual_state,
                shell_spec=shell_spec,
                cancel_event=cancel_event,
                env=env,
            )
        result = await StructuredAgent(
            provider=child_llms[model_profile],
            registry=effective_registry,
            config=config,
            runtime=AgentRuntimeConfig(
                runtime_context_provider=build_runtime_context_provider(
                    child_root,
                    effective_registry,
                    skill_loader,
                    shell_spec=shell_spec,
                    contextual_state=child_contextual_state,
                    modules=runtime_config.prompt.modules,
                ),
            ),
        ).run_async(prompt)
        return result.answer

    if "worktree" in enabled:
        from xcode.coding_agent.tools.worktree import WorktreeTaskRunner

        worktree_runner = WorktreeTaskRunner(project_root)
    else:
        worktree_runner = None
    managed_runner = ManagedSubagentRunner(
        run_child,
        available_profiles=tuple(child_llms),
        default_profile=PROFILE_SUBAGENT,
        worktree_runner=worktree_runner,
    )
    return [managed_runner.shutdown], build_managed_subagent_tools(managed_runner)


# ── 可选服务 ──


def load_opt_in_services(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
    enabled: set[str],
) -> OptInServices:
    daemon = None
    if "daemon" in enabled:
        from xcode.harness.daemon import HeartbeatDaemon

        daemon = HeartbeatDaemon(
            project_root=project_root,
            interval_seconds=runtime_config.daemon.interval_seconds,
        )
    mailbox = None
    if "mailbox" in enabled:
        from xcode.harness.mailbox import AgentMailbox

        mailbox = AgentMailbox(project_root)
    progress: bool | None = None
    if "progress" in enabled:
        progress = True
    return OptInServices(daemon=daemon, mailbox=mailbox, progress=progress)


# ── Agent 构建 ──


def build_agent(
    project_root: Path,
    llm: ModelProvider,
    registry: tuple[ToolSpec, ...],
    config: AgentConfig,
    audit_path: Path | None,
    runtime_config: XcodeRuntimeConfig,
    skill_loader: SkillLoader | None,
    contextual_state: ContextualRetrievalState | None = None,
    shell_spec: ShellSpec | None = None,
    compact_controller: CompactController | None = None,
    cancellation_token: CancellationToken | None = None,
    compactor: LayeredCompactor | None = None,
    fallback_provider: ModelProvider | None = None,
    plugins_hooks: dict[str, list[Callable]] | None = None,
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
) -> StructuredAgent:
    hook_manager = None
    if contextual_state is not None:
        hook_manager = HookManager()

        def record_post_tool(record) -> None:
            contextual_state.record_tool_result(record.tool, record.output)

        hook_manager.register("post_tool", record_post_tool)

    if hook_manager is None and plugins_hooks:
        hook_manager = HookManager()

    if hook_manager is not None:
        hook_manager.register("before_agent_start", lambda r: None)
        hook_manager.register("before_provider_request", lambda r: None)

        if plugins_hooks:
            for event, callbacks in plugins_hooks.items():
                for cb in callbacks:
                    hook_manager.register(cast("HookEvent", event), cb)

    sec = runtime_config.security
    return StructuredAgent(
        provider=llm,
        registry=registry,
        config=config,
        gate=GateConfig(
            permission_policy=_permission_policy_from_security(sec),
            restricted_dirs=sec.restricted_dirs,
            allowlist_mode=bool(sec.allow_tools),
            hook_constraint_providers=hook_constraint_providers,
            hook_manager=hook_manager,
            audit_logger=JsonlAuditLogger(audit_path).write if audit_path else None,
        ),
        runtime=AgentRuntimeConfig(
            compactor=compactor,
            compact_controller=compact_controller,
            cancellation_token=cancellation_token,
            runtime_context_provider=build_runtime_context_provider(
                project_root,
                registry,
                skill_loader,
                shell_spec=shell_spec,
                contextual_state=contextual_state,
                modules=runtime_config.prompt.modules,
            ),
            fallback_provider=fallback_provider,
            project_root=project_root,
            request_hygiene=runtime_config.request_hygiene,
        ),
    )


def _permission_policy_from_security(
    security: SecurityRuntimeConfig,
) -> PermissionPolicy | None:
    """将运行时 security 配置转换为工具权限规则。"""
    rules: list[PermissionRule] = []
    rules.extend(PermissionRule(tool, "deny") for tool in security.deny_tools)
    rules.extend(PermissionRule(tool, "ask") for tool in security.ask_tools)
    rules.extend(PermissionRule(tool, "allow") for tool in security.allow_tools)
    if security.resolve_approval_policy() == "always":
        rules.append(PermissionRule("*", "ask"))
    if not rules:
        return None
    return PermissionPolicy(tuple(rules))
