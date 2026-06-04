"""应用装配工厂函数。

从 app.py 提取的配置解析、共享基础设施构建、provider 组装、
工具注册、agent 构建和实验性服务加载逻辑。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from xcode.harness.config import (
    AgentConfig,
    PROFILE_MAIN,
    PROFILE_SUBAGENT,
    XcodeRuntimeConfig,
    discover_runtime_config,
    resolve_config_path,
    to_agent_config,
)
from xcode.harness.agent_runtime import (
    CancellationToken,
    ContextualRetrievalState,
    ManagedSubagentRunner,
    StructuredAgent,
    build_managed_subagent_tools,
    build_runtime_context_provider,
)
from xcode.harness.agent_runtime.compaction import CompactController, LayeredCompactor
from xcode.ai.providers.protocol import ModelProvider
from xcode.harness.observability import JsonlAuditLogger, HookManager
from xcode.harness.observability.hooks import HookEvent
from xcode.harness.skills import ToolSpec
from xcode.harness.skill_loader import SkillLoader, build_skill_loader_tool
from xcode.harness.tools import (
    ShellSpec,
    build_bash_tool,
    build_code_tools,
    build_file_tools,
)
from xcode.ai.providers.factory import (
    ModelProfileProto,
    ProviderSettings,
    build_provider_bundle,
)

if TYPE_CHECKING:
    from xcode.experimental.daemon import HeartbeatDaemon
    from xcode.experimental.mailbox import AgentMailbox
    from xcode.experimental.progress import TaskProgress
    from xcode.experimental.speculation import SpeculationPlanner


EXPERIMENTAL_FEATURE_GROUPS = frozenset(
    {
        "worktree",
        "mcp",
        "tasks",
        "memory",
        "plugins",
        "daemon",
        "mailbox",
        "progress",
        "speculation",
    }
)


@dataclass(frozen=True)
class ExperimentalServices:
    daemon: HeartbeatDaemon | None = None
    mailbox: AgentMailbox | None = None
    progress: type[TaskProgress] | None = None


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
    agent_config = agent_config or to_agent_config(runtime_config)
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
    model_profiles = cast(
        "dict[str, ModelProfileProto]", runtime_config.provider.model_profiles
    )
    return build_provider_bundle(
        ProviderSettings(env_files=env_files, model_profiles=model_profiles)
    )


# ── 工具注册 ──


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
) -> tuple[
    tuple[ToolSpec, ...], SkillLoader | None, ShellSpec, tuple[Callable[[], None], ...]
]:
    from xcode.harness.tools.shell_adapter import detect_shell

    enabled = effective_enabled_groups(runtime_config.tools.enabled_groups)
    closers: list[Callable[[], None]] = []
    shell_spec = detect_shell(runtime_config.tools.shell)
    registry = build_project_scoped_registry(
        project_root=project_root,
        enabled=enabled,
        contextual_state=contextual_state,
        shell_spec=shell_spec,
        cancel_event=cancel_event,
    )
    if "worktree" in enabled:
        from xcode.experimental.worktree import WorktreeTaskRunner, build_worktree_tools

        registry += build_worktree_tools(WorktreeTaskRunner(project_root))
    if "mcp" in enabled:
        from xcode.experimental.mcp import build_mcp_tools

        registry += build_mcp_tools(project_root)
    if "tasks" in enabled:
        from xcode.experimental.tasks import TaskStore, build_task_tools

        registry += build_task_tools(TaskStore(project_root))
    if "mailbox" in enabled:
        from xcode.experimental.mailbox import AgentMailbox, build_mailbox_tools

        registry += build_mailbox_tools(AgentMailbox(project_root))
    if "progress" in enabled:
        from xcode.experimental.progress import build_progress_tools
        from xcode.experimental.tasks import TaskStore

        registry += build_progress_tools(TaskStore(project_root))

    skills_dir = skills_dir or project_root / "xcode" / "skills"
    skill_loader = None
    if "skills" in enabled and skills_dir.exists():
        skill_loader = SkillLoader(skills_dir)
        registry += (build_skill_loader_tool(skill_loader),)

    child_registry = registry
    if "subagent" in enabled or "experimental" in enabled:
        child_llms = llm_profiles_dict(llm, llm_profiles)

        async def run_child(prompt, model_profile=PROFILE_SUBAGENT, cwd_override=None):
            effective_registry = child_registry
            if cwd_override is not None:
                effective_registry = build_project_scoped_registry(
                    project_root=Path(cwd_override),
                    enabled=enabled,
                    contextual_state=contextual_state,
                    shell_spec=shell_spec,
                    cancel_event=cancel_event,
                )
            result = await StructuredAgent(
                provider=child_llms[model_profile],
                registry=effective_registry,
                config=config,
            ).run_async(prompt)
            return result.answer

        worktree_runner = (
            _build_worktree_runner(project_root) if "worktree" in enabled else None
        )
        managed_runner = ManagedSubagentRunner(
            run_child,
            available_profiles=tuple(child_llms),
            default_profile=PROFILE_SUBAGENT,
            worktree_runner=worktree_runner,
        )
        closers.append(managed_runner.shutdown)
        if "subagent" in enabled:
            registry += build_managed_subagent_tools(managed_runner)
    return registry, skill_loader, shell_spec, tuple(closers)


def build_project_scoped_registry(
    project_root: Path,
    enabled: set[str],
    contextual_state: ContextualRetrievalState | None,
    shell_spec: ShellSpec,
    cancel_event: threading.Event | None = None,
) -> tuple[ToolSpec, ...]:
    from xcode.harness.skills import BASE_REGISTRY

    registry = BASE_REGISTRY
    registry += build_file_tools(project_root, context_state=contextual_state)
    registry += build_code_tools(project_root)
    registry += (
        build_bash_tool(project_root, shell_spec=shell_spec, cancel_event=cancel_event),
    )
    return tuple(t for t in registry if t.group in enabled)


def llm_profiles_dict(
    llm: ModelProvider, llm_profiles: Mapping[str, ModelProvider] | None
) -> dict[str, ModelProvider]:
    profiles = dict(llm_profiles or {})
    if not profiles:
        profiles[PROFILE_MAIN] = llm
    profiles.setdefault(PROFILE_SUBAGENT, profiles[PROFILE_MAIN])
    return profiles


def _build_worktree_runner(project_root: Path):
    from xcode.experimental.worktree import WorktreeTaskRunner

    return WorktreeTaskRunner(project_root)


# ── 实验性服务 ──


def load_experimental_services(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
    enabled: set[str],
) -> ExperimentalServices:
    daemon = None
    if "daemon" in enabled:
        from xcode.experimental.daemon import HeartbeatDaemon

        daemon = HeartbeatDaemon(
            project_root=project_root,
            interval_seconds=runtime_config.daemon.interval_seconds,
        )
    mailbox = None
    if "mailbox" in enabled:
        from xcode.experimental.mailbox import AgentMailbox

        mailbox = AgentMailbox(project_root)
    progress = None
    if "progress" in enabled:
        from xcode.experimental.progress import TaskProgress

        progress = TaskProgress
    return ExperimentalServices(daemon=daemon, mailbox=mailbox, progress=progress)


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
    speculation_planner: SpeculationPlanner | None = None,
    fallback_provider: ModelProvider | None = None,
    plugins_hooks: dict[str, list[Callable]] | None = None,
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

    return StructuredAgent(
        provider=llm,
        registry=registry,
        config=config,
        audit_logger=JsonlAuditLogger(audit_path).write if audit_path else None,
        hook_manager=hook_manager,
        runtime_context_provider=build_runtime_context_provider(
            project_root,
            registry,
            skill_loader,
            shell_spec=shell_spec,
            contextual_state=contextual_state,
            modules=runtime_config.prompt.modules,
        ),
        compactor=compactor,
        compact_controller=compact_controller,
        cancellation_token=cancellation_token,
        speculation_planner=speculation_planner,
        fallback_provider=fallback_provider,
        project_root=project_root,
    )
