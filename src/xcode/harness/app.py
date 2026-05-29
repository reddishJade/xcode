from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xcode.harness.config import (
    AgentConfig,
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
from xcode.harness.agent_runtime.provider import ModelProvider
from xcode.harness.observability import JsonlAuditLogger, HookManager
from xcode.harness.skills import ToolSpec
from xcode.harness.skill_loader import SkillLoader, build_skill_loader_tool
from xcode.harness.tools import (
    ShellSpec,
    build_bash_tool,
    build_code_tools,
    build_file_tools,
)
from xcode.ai.providers.factory import ProviderSettings, build_provider_bundle
from xcode.experimental.worktree import WorktreeTaskRunner, build_worktree_tools


@dataclass
class XcodeApp:
    """Xcode 应用句柄。"""

    agent: Any
    registry: tuple[Any, ...] = ()
    contextual_state: Any = None
    daemon: Any = None
    _model_profiles: dict[str, Any] | None = None
    _env_files: tuple[Path, ...] = ()
    _closers: tuple[Callable[[], None], ...] = ()
    _closed: bool = False

    def set_model(
        self,
        *,
        model: str,
        profile: str = "main",
        base_url: str | None = None,
        api_key: str | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        from xcode.ai.providers import build_provider_bundle, ProviderSettings
        from xcode.ai.providers.factory import ModelProfileConfig, ModelProfileProto

        if not self._model_profiles:
            return getattr(self.agent.provider, "model", "unknown")
        profile_config = self._model_profiles.get(profile)
        if not profile_config:
            return getattr(self.agent.provider, "model", "unknown")
        new_cfg: ModelProfileProto = ModelProfileConfig(
            transport=profile_config.transport,
            chat_model=model,
            base_url=base_url or profile_config.base_url,
            api_key=api_key or profile_config.api_key,
            thinking=thinking if thinking is not None else profile_config.thinking,
            reasoning_effort=reasoning_effort
            if reasoning_effort is not None
            else profile_config.reasoning_effort,
        )
        bundle = build_provider_bundle(
            ProviderSettings(
                env_files=self._env_files,
                model_profiles={profile: new_cfg},
            )
        )
        self.agent.provider = (
            bundle.llm if profile == "main" else bundle.llms.get("subagent", bundle.llm)
        )
        self._model_profiles[profile] = new_cfg
        return model

    def get_model_info(self) -> dict[str, str]:
        provider = getattr(self.agent, "provider", None)
        model_name = getattr(provider, "model", "unknown") if provider else "unknown"
        base_url = getattr(provider, "base_url", "") if provider else ""
        thinking = getattr(provider, "thinking", None)
        reasoning_effort = getattr(provider, "reasoning_effort", None)
        info: dict[str, str] = {
            "model": model_name,
            "base_url": base_url,
            "profile": "main",
        }
        if thinking is not None:
            info["thinking"] = str(thinking)
        if reasoning_effort:
            info["reasoning_effort"] = reasoning_effort
        return info

    def ask(self, question: str) -> str:
        return self.agent.run(question).answer

    async def aask(self, question: str) -> str:
        return (await self.agent.run_async(question)).answer

    def ask_stream(self, question: str, mode: str | None = None) -> Iterator[Any]:
        yield from self.agent.run_stream(question, mode=mode)

    async def aask_stream(
        self, question: str, mode: str | None = None
    ) -> AsyncIterator[Any]:
        async for event in self.agent.arun_stream(question, mode=mode):
            yield event

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for closer in self._closers:
            closer()

    async def close_async(self) -> None:
        self.close()


# ── 应用装配 ──


def _build_providers(runtime_config: XcodeRuntimeConfig, env_files: tuple[Path, ...]):
    from xcode.ai.providers.factory import ModelProfileProto
    from typing import cast

    model_profiles = cast(
        "dict[str, ModelProfileProto]", runtime_config.provider.model_profiles
    )
    return build_provider_bundle(
        ProviderSettings(
            env_files=env_files,
            model_profiles=model_profiles,
        )
    )


def _build_tool_registry(
    project_root: Path,
    llm,
    llm_profiles: Mapping[str, ModelProvider] | None,
    config: AgentConfig,
    runtime_config: XcodeRuntimeConfig,
    skills_dir: Path | None,
    contextual_state: ContextualRetrievalState | None = None,
    compact_controller: CompactController | None = None,
    cancel_event: Any = None,
) -> tuple[
    tuple[ToolSpec, ...], SkillLoader | None, ShellSpec, tuple[Callable[[], None], ...]
]:
    from xcode.harness.tools.shell_adapter import detect_shell

    enabled = set(runtime_config.tools.enabled_groups)
    closers: list[Callable[[], None]] = []
    shell_spec = detect_shell(runtime_config.tools.shell)
    registry = _build_project_scoped_registry(
        project_root=project_root,
        enabled=enabled,
        contextual_state=contextual_state,
        shell_spec=shell_spec,
        cancel_event=cancel_event,
    )
    if "worktree" in enabled:
        registry += build_worktree_tools(WorktreeTaskRunner(project_root))
    if "mcp" in enabled:
        from xcode.experimental.mcp import build_mcp_tools

        registry += build_mcp_tools(project_root)
    if "tasks" in enabled:
        from xcode.experimental.tasks import TaskStore, build_task_tools

        registry += build_task_tools(TaskStore(project_root))

    skills_dir = skills_dir or project_root / "xcode" / "skills"
    skill_loader = None
    if "skills" in enabled and skills_dir.exists():
        skill_loader = SkillLoader(skills_dir)
        registry += (build_skill_loader_tool(skill_loader),)

    child_registry = registry
    if "subagent" in enabled or "experimental" in enabled:
        child_llms = _llm_profiles(llm, llm_profiles)

        async def run_child(prompt, model_profile="subagent", cwd_override=None):
            effective_registry = child_registry
            if cwd_override is not None:
                effective_registry = _build_project_scoped_registry(
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
            WorktreeTaskRunner(project_root) if "worktree" in enabled else None
        )
        managed_runner = ManagedSubagentRunner(
            run_child,
            available_profiles=tuple(child_llms),
            default_profile="subagent",
            worktree_runner=worktree_runner,
        )
        closers.append(managed_runner.shutdown)
        if "subagent" in enabled:
            registry += build_managed_subagent_tools(managed_runner)
    return registry, skill_loader, shell_spec, tuple(closers)


def _build_project_scoped_registry(
    project_root: Path,
    enabled: set[str],
    contextual_state: ContextualRetrievalState | None,
    shell_spec: ShellSpec,
    cancel_event: Any = None,
) -> tuple[ToolSpec, ...]:
    from xcode.harness.skills import BASE_REGISTRY

    registry = BASE_REGISTRY
    registry += build_file_tools(project_root, context_state=contextual_state)
    registry += build_code_tools(project_root)
    registry += (
        build_bash_tool(project_root, shell_spec=shell_spec, cancel_event=cancel_event),
    )
    return tuple(t for t in registry if t.group in enabled)


def _llm_profiles(llm, llm_profiles: Mapping[str, ModelProvider] | None):
    profiles = dict(llm_profiles or {})
    if not profiles:
        profiles["main"] = llm
    profiles.setdefault("subagent", profiles["main"])
    return profiles


def _build_agent(
    project_root: Path,
    llm,
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
    fallback_provider=None,
    plugins_hooks=None,
):

    hook_manager = None
    if contextual_state is not None:
        hook_manager = HookManager()

        def record_post_tool(record) -> None:
            contextual_state.record_tool_result(record.tool, record.output)

        hook_manager.register("post_tool", record_post_tool)

    if hook_manager is None and plugins_hooks:
        hook_manager = HookManager()

    if plugins_hooks and hook_manager is not None:
        for event, callbacks in plugins_hooks.items():
            for cb in callbacks:
                hook_manager.register(event, cb)

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
        fallback_provider=fallback_provider,
        project_root=project_root,
    )


def build_app(
    project_root: Path,
    env_files: tuple[Path, ...] | None = None,
    agent_config: AgentConfig | None = None,
    skills_dir: Path | None = None,
    audit_path: Path | None = None,
    runtime_config: XcodeRuntimeConfig | None = None,
) -> XcodeApp:
    """装配完整的 Xcode 应用。"""
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

    providers = _build_providers(runtime_config, env_files)
    contextual_state = ContextualRetrievalState(project_root)
    cancellation_token = CancellationToken()
    compact_controller = CompactController()

    transcript_dir = (
        resolve_config_path(project_root, runtime_config.paths.sessions_dir)
        if runtime_config.paths.sessions_dir
        else project_root / ".local" / "sessions"
    )

    from xcode.experimental.memory import MemoryManager

    memory_manager = MemoryManager(project_root)

    compactor = LayeredCompactor(
        transcript_dir=transcript_dir,
        max_recent_messages=runtime_config.agent.max_recent_messages,
        on_compact=memory_manager.consolidate,
    )

    from xcode.experimental.plugins import PluginManager

    plugin_mgr = PluginManager(project_root)
    plugins_data = plugin_mgr.scan_and_load()

    registry, skill_loader, shell_spec, closers = _build_tool_registry(
        project_root=project_root,
        llm=providers.llm,
        llm_profiles=providers.llms,
        config=agent_config,
        runtime_config=runtime_config,
        skills_dir=skills_dir,
        contextual_state=contextual_state,
        compact_controller=compact_controller,
        cancel_event=cancellation_token.event,
    )

    if plugins_data.get("tools"):
        registry = registry + tuple(plugins_data["tools"])

    fallback_provider = providers.llms.get("fallback")
    agent = _build_agent(
        project_root=project_root,
        llm=providers.llm,
        registry=registry,
        config=agent_config,
        audit_path=audit_path,
        runtime_config=runtime_config,
        skill_loader=skill_loader,
        contextual_state=contextual_state,
        shell_spec=shell_spec,
        compactor=compactor,
        compact_controller=compact_controller,
        cancellation_token=cancellation_token,
        fallback_provider=fallback_provider,
        plugins_hooks=plugins_data.get("hooks"),
    )

    daemon = None
    if runtime_config.daemon.enabled:
        from xcode.experimental.daemon import HeartbeatDaemon

        daemon = HeartbeatDaemon(
            project_root=project_root,
            interval_seconds=runtime_config.daemon.interval_seconds,
        )

    return XcodeApp(
        agent=agent,
        registry=registry,
        contextual_state=contextual_state,
        daemon=daemon,
        _env_files=env_files,
        _model_profiles=runtime_config.provider.model_profiles,
        _closers=closers,
    )
