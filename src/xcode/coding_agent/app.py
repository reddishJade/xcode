"""Xcode 应用入口。

XcodeApp 数据类和 build_app 编排函数。装配逻辑委托给 assembly 子包。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from xcode.coding_agent.execution_modes import ExecutionMode
from xcode.harness.config import AgentConfig, XcodeRuntimeConfig
from xcode.harness.agent_runtime import (
    ContextualRetrievalState,
    AgentHarnessEvent,
)
from xcode.coding_agent.harness import CodingAgentHarness
from xcode.agent.messages import AgentMessage, UserMessage
from xcode.agent.types import ToolSpec
from xcode.harness.observability import ExternalHookDiagnostic, ExternalHookRunner
from xcode.harness.session_todo import SessionTodoState
from xcode.harness.session import SessionStore
from xcode.harness.session.recorder import SessionRecorder
from xcode.harness.session.replay import replay_session
from xcode.ai.providers.registry import ProviderSettings, build_provider_bundle
from . import assembly as _assembly
from .assembly import (
    build_agent,
    build_shared_infra,
)

if TYPE_CHECKING:
    from xcode.harness.memory import MemoryManager
    from xcode.harness.mcp import McpRuntimeRegistry


@dataclass
class XcodeApp:
    """Xcode 应用句柄。"""

    agent: CodingAgentHarness
    registry: tuple[ToolSpec, ...] = ()
    contextual_state: ContextualRetrievalState | None = None
    external_hook_runner: ExternalHookRunner | None = None

    memory_manager: MemoryManager | None = None
    mcp_runtime: McpRuntimeRegistry | None = None
    session_recorder: SessionRecorder | None = None
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
        from xcode.ai.providers.registry import ModelProfileConfig
        from xcode.ai.providers.registry import ModelProfileProto

        if profile != "main":
            raise ValueError("only the main composition can be replaced at runtime")
        if not self._model_profiles:
            return self.agent.provider.model
        profile_config = self._model_profiles.get(profile)
        if not profile_config:
            return self.agent.provider.model
        new_cfg: ModelProfileProto = ModelProfileConfig(
            transport=profile_config.transport,
            chat_model=model,
            base_url=base_url or profile_config.base_url,
            api_key=api_key or profile_config.api_key,
            thinking=thinking if thinking is not None else profile_config.thinking,
            reasoning_effort=reasoning_effort
            if reasoning_effort is not None
            else profile_config.reasoning_effort,
            clear_thinking=profile_config.clear_thinking,
            tool_stream=profile_config.tool_stream,
            response_format=profile_config.response_format,
        )
        bundle = build_provider_bundle(
            ProviderSettings(
                env_files=self._env_files,
                model_profiles={profile: new_cfg},
            )
        )
        new_provider = bundle.llm
        self.agent.replace_primary_provider(new_provider)
        self._model_profiles[profile] = new_cfg
        return model

    def get_model_info(self) -> dict[str, str]:
        provider = self.agent.provider
        active = getattr(provider, "active_provider", provider)
        info: dict[str, str] = {
            "model": active.model,
            "base_url": active.base_url,
            "transport": active.transport,
            "profile": "main",
        }
        if active.thinking:
            info["thinking"] = str(active.thinking)
        else:
            info["thinking"] = "off"
        if active.reasoning_effort is not None:
            info["reasoning_effort"] = active.reasoning_effort
        return info

    def ask(self, question: str) -> str:
        answer = ""
        for event in self.ask_stream(question):
            if event.type == "final":
                answer = event.data.answer
        return answer

    async def aask(self, question: str) -> str:
        answer = ""
        async for event in self.aask_stream(question):
            if event.type == "final":
                answer = event.data.answer
        return answer

    @property
    def session_store(self) -> SessionStore:
        recorder = self.session_recorder
        if recorder is None:
            raise RuntimeError("session recorder is not configured")
        return recorder.store

    def ask_stream(
        self,
        question: str | None,
        mode: ExecutionMode | None = None,
        *,
        display_question: str | None = None,
    ) -> Iterator[AgentHarnessEvent]:
        recorder = self.session_recorder
        if recorder is None:
            raise RuntimeError("session recorder is not configured")
        recorder.bind_agent(self.agent)
        if question is not None:
            self.agent.followup(
                UserMessage(content=question),
                display_text=display_question,
            )
        for event in self.agent.run_stream(
            None,
            mode=mode,
            display_question=display_question,
        ):
            recorder.record_event(event)
            yield event

    async def aask_stream(
        self,
        question: str | None,
        mode: ExecutionMode | None = None,
        *,
        display_question: str | None = None,
    ) -> AsyncIterator[AgentHarnessEvent]:
        recorder = self.session_recorder
        if recorder is None:
            raise RuntimeError("session recorder is not configured")
        recorder.bind_agent(self.agent)
        if question is not None:
            self.agent.followup(
                UserMessage(content=question),
                display_text=display_question,
            )
        async for event in self.agent.arun_stream(
            None,
            mode=mode,
            display_question=display_question,
        ):
            recorder.record_event(event)
            yield event

    def hook_diagnostics(self) -> tuple[ExternalHookDiagnostic, ...]:
        """返回外部命令 hook 的运行时诊断。"""
        if self.external_hook_runner is None:
            return ()
        return self.external_hook_runner.diagnostics()

    def record_compaction(
        self,
        *,
        summary: str,
        messages_before: int,
        messages_after: int,
        tokens_before: int,
        tokens_after: int,
        replacement: list[AgentMessage],
    ) -> str:
        recorder = self.session_recorder
        if recorder is None:
            raise RuntimeError("session recorder is not configured")
        return recorder.record_compaction(
            summary=summary,
            messages_before=messages_before,
            messages_after=messages_after,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            replacement=replacement,
        )

    def restore_session(self) -> None:
        """从当前 session branch 恢复完整 agent 运行状态。"""
        replay_session(self.agent, self.session_store, self.contextual_state)

    def mcp_status(self) -> tuple[dict[str, object], ...]:
        """返回 MCP server 运行时状态快照。"""
        if self.mcp_runtime is None:
            return ()
        return tuple(status.__dict__ for status in self.mcp_runtime.status_snapshot())

    def reload_mcp(self) -> tuple[str, ...]:
        """重新读取 MCP 配置并返回当前工具名快照。"""
        if self.mcp_runtime is None:
            return ()
        return tuple(tool.name for tool in self.mcp_runtime.reload())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for closer in self._closers:
            closer()


def build_app(
    project_root: Path,
    env_files: tuple[Path, ...] | None = None,
    agent_config: AgentConfig | None = None,
    skills_dir: Path | None = None,
    audit_path: Path | None = None,
    runtime_config: XcodeRuntimeConfig | None = None,
    sessions_dir: Path | None = None,
) -> XcodeApp:
    """装配完整的 Xcode 应用。"""
    cfg = _assembly.resolve_config(
        project_root, env_files, agent_config, skills_dir, audit_path, runtime_config
    )
    infra = build_shared_infra(
        project_root,
        cfg.runtime_config,
        sessions_dir=sessions_dir,
    )

    # 使用共享的 MemoryManager 实例，确保 compactor 和 agent 使用同一实例
    memory_manager = infra.memory_manager

    providers = build_provider_bundle(
        ProviderSettings(
            env_files=cfg.env_files,
            model_profiles=cfg.runtime_config.provider.model_profiles,
        )
    )
    todo_state = SessionTodoState()
    external_hook_runner = (
        ExternalHookRunner(cfg.runtime_config.hooks.entries, project_root)
        if cfg.runtime_config.hooks.entries
        else None
    )
    (
        registry_state,
        shell_spec,
        closers,
        skill_registry,
        mcp_runtime_registry,
    ) = _assembly.build_tool_registry(
        project_root=project_root,
        llm=providers.llm,
        runtime_config=cfg.runtime_config,
        session_recorder=infra.session_recorder,
        contextual_state=infra.contextual_state,
        cancel_event=infra.cancellation_token,
        skills_dir=cfg.skills_dir,
        memory_manager=memory_manager,
        session_history=infra.session_history,
        todo_state=todo_state,
    )

    fallback_provider = providers.llms.get("fallback")
    # 为 LayeredCompactor 接入 LLM 驱动的摘要生成，替代纯规则 fallback
    from xcode.harness.agent_runtime.compaction import build_compact_summarize_fn

    infra.compactor.summarize_fn = build_compact_summarize_fn(providers.llm)

    agent = build_agent(
        project_root=project_root,
        llm=providers.llm,
        registry=registry_state,
        config=cfg.agent_config,
        audit_path=cfg.audit_path,
        runtime_config=cfg.runtime_config,
        session_recorder=infra.session_recorder,
        session_inbox=infra.session_inbox,
        contextual_state=infra.contextual_state,
        shell_spec=shell_spec,
        compactor=infra.compactor,
        compact_controller=infra.compact_controller,
        cancellation_token=infra.cancellation_token,
        fallback_provider=fallback_provider,
        skill_registry=skill_registry,
        external_hook_runner=external_hook_runner,
        memory_manager=memory_manager,
        session_history=infra.session_history,
        todo_state=todo_state,
    )

    return XcodeApp(
        agent=agent,
        registry=registry_state,
        contextual_state=infra.contextual_state,
        external_hook_runner=external_hook_runner,
        memory_manager=memory_manager,
        mcp_runtime=mcp_runtime_registry,
        session_recorder=infra.session_recorder,
        _env_files=cfg.env_files,
        _model_profiles=cfg.runtime_config.provider.model_profiles,
        _closers=closers,
    )
