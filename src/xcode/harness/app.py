"""Xcode 应用入口。

XcodeApp 数据类和 build_app 编排函数。装配逻辑委托给 assembly.py。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from xcode.harness.config import ExecutionMode
from xcode.harness.agent_runtime import ContextualRetrievalState, StructuredAgent
from xcode.harness.skills import ToolSpec
from . import assembly as _assembly
from .assembly import (
    ResolvedConfig,
    SharedInfra,
    ExperimentalServices,
    build_agent,
    build_shared_infra,
)
# re-export for test backward compatibility (mock targets)
_build_providers = _assembly.build_providers
_build_tool_registry = _assembly.build_tool_registry
_build_project_scoped_registry = _assembly.build_project_scoped_registry
_effective_enabled_groups = _assembly.effective_enabled_groups
_load_experimental_services = _assembly.load_experimental_services
_resolve_config = _assembly.resolve_config
_build_worktree_runner = _assembly._build_worktree_runner
from xcode.harness.tools import build_bash_tool  # noqa: F401 — test mock target

if TYPE_CHECKING:
    from xcode.experimental.daemon import HeartbeatDaemon
    from xcode.experimental.mailbox import AgentMailbox
    from xcode.experimental.progress import TaskProgress


@dataclass
class XcodeApp:
    """Xcode 应用句柄。"""

    agent: StructuredAgent
    registry: tuple[ToolSpec, ...] = ()
    contextual_state: ContextualRetrievalState | None = None
    daemon: HeartbeatDaemon | None = None
    mailbox: AgentMailbox | None = None
    progress: type[TaskProgress] | None = None
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
        from xcode.ai.providers.factory import ModelProfileConfig
        from xcode.ai.providers.factory import ModelProfileProto

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
            reasoning_effort=reasoning_effort if reasoning_effort is not None else profile_config.reasoning_effort,
        )
        bundle = build_provider_bundle(ProviderSettings(
            env_files=self._env_files, model_profiles={profile: new_cfg},
        ))
        self.agent.provider = bundle.llm if profile == "main" else bundle.llms.get("subagent", bundle.llm)
        self._model_profiles[profile] = new_cfg
        return model

    def get_model_info(self) -> dict[str, str]:
        provider = getattr(self.agent, "provider", None)
        model_name = getattr(provider, "model", "unknown") if provider else "unknown"
        base_url = getattr(provider, "base_url", "") if provider else ""
        thinking = getattr(provider, "thinking", None)
        reasoning_effort = getattr(provider, "reasoning_effort", None)
        info: dict[str, str] = {"model": model_name, "base_url": base_url, "profile": "main"}
        if thinking is not None:
            info["thinking"] = str(thinking)
        if reasoning_effort:
            info["reasoning_effort"] = reasoning_effort
        return info

    def ask(self, question: str) -> str:
        return self.agent.run(question).answer

    async def aask(self, question: str) -> str:
        return (await self.agent.run_async(question)).answer

    def ask_stream(self, question: str, mode: ExecutionMode | None = None) -> Iterator[Any]:
        yield from self.agent.run_stream(question, mode=mode)

    async def aask_stream(self, question: str, mode: ExecutionMode | None = None) -> AsyncIterator[Any]:
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


def build_app(
    project_root: Path,
    env_files: tuple[Path, ...] | None = None,
    agent_config: Any | None = None,
    skills_dir: Path | None = None,
    audit_path: Path | None = None,
    runtime_config: Any | None = None,
) -> XcodeApp:
    """装配完整的 Xcode 应用。"""
    cfg = _assembly.resolve_config(project_root, env_files, agent_config, skills_dir, audit_path, runtime_config)
    enabled = _assembly.effective_enabled_groups(cfg.runtime_config.tools.enabled_groups)
    infra = build_shared_infra(project_root, cfg.runtime_config, enabled)

    providers = _assembly.build_providers(cfg.runtime_config, cfg.env_files)

    plugins_data: dict[str, Any] = {"tools": [], "hooks": {}, "skills": []}
    if "plugins" in enabled:
        from xcode.experimental.plugins import PluginManager
        plugins_data = PluginManager(project_root).scan_and_load()

    speculation_planner = None
    if "speculation" in enabled:
        from xcode.experimental.speculation import SpeculationPlanner
        speculation_planner = SpeculationPlanner()

    registry, skill_loader, shell_spec, closers = _assembly.build_tool_registry(
        project_root=project_root, llm=providers.llm, llm_profiles=providers.llms,
        config=cfg.agent_config, runtime_config=cfg.runtime_config, skills_dir=cfg.skills_dir,
        contextual_state=infra.contextual_state, compact_controller=infra.compact_controller,
        cancel_event=infra.cancellation_token.event,
    )

    if plugins_data.get("tools"):
        registry = registry + tuple(plugins_data["tools"])

    fallback_provider = providers.llms.get("fallback")
    agent = build_agent(
        project_root=project_root, llm=providers.llm, registry=registry,
        config=cfg.agent_config, audit_path=cfg.audit_path, runtime_config=cfg.runtime_config,
        skill_loader=skill_loader, contextual_state=infra.contextual_state, shell_spec=shell_spec,
        compactor=infra.compactor, compact_controller=infra.compact_controller,
        cancellation_token=infra.cancellation_token, fallback_provider=fallback_provider,
        speculation_planner=speculation_planner, plugins_hooks=plugins_data.get("hooks"),
    )

    experimental_services = _assembly.load_experimental_services(project_root, cfg.runtime_config, enabled)

    return XcodeApp(
        agent=agent, registry=registry, contextual_state=infra.contextual_state,
        daemon=experimental_services.daemon, mailbox=experimental_services.mailbox,
        progress=experimental_services.progress, _env_files=cfg.env_files,
        _model_profiles=cfg.runtime_config.provider.model_profiles, _closers=closers,
    )
