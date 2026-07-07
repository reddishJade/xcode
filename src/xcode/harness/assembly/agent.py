"""Agent 构建。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from xcode.ai.providers.base import ModelProvider
from xcode.agent.types import ToolSpec
from xcode.coding_agent.tools import ShellSpec
from xcode.coding_agent.registry import build_project_scoped_registry

from ..agent_runtime import CancellationToken, CodingAgentHarness, ContextualRetrievalState
from ..agent_runtime.config import AgentRuntimeConfig, GateConfig
from ..agent_runtime.compaction import CompactController, LayeredCompactor
from ..agent_runtime.prompting import build_runtime_context_provider
from ..config import AgentConfig, XcodeRuntimeConfig
from ..observability import (
    ExternalHookRunner,
    HookManager,
    HookRecord,
    JsonlAuditLogger,
)
from ..observability.permission_model import PolicyEvaluator

from .security import (
    external_directories_from_security,
    mode_rulesets_from_runtime_config,
    permission_policy_from_security,
)

if TYPE_CHECKING:
    from ..agent_skills import SkillRegistry
    from ..memory import MemoryManager


def build_hook_manager(
    contextual_state: ContextualRetrievalState | None,
    external_hook_runner: ExternalHookRunner | None,
    project_root: Path,
    *,
    subagent: bool,
) -> HookManager | None:
    if contextual_state is None and external_hook_runner is None:
        return None
    manager = HookManager()
    if contextual_state is not None:

        def record_post_tool(record: object) -> None:
            contextual_state.record_tool_result(
                getattr(record, "tool", ""),
                getattr(record, "output", ""),
            )

        manager.register("post_tool", record_post_tool)
    if external_hook_runner is not None:
        for event in (
            "post_tool",
            "on_error",
            "on_compact",
            "before_agent_start",
            "before_provider_request",
        ):

            def run_external(
                record: HookRecord,
                runner: ExternalHookRunner = external_hook_runner,
                is_subagent: bool = subagent,
            ) -> None:
                runner.execute(
                    record,
                    subagent=is_subagent,
                    cwd=project_root,
                )

            manager.register_background(event, run_external)
    return manager


def build_agent(
    project_root: Path,
    llm: ModelProvider,
    registry: tuple[ToolSpec, ...],
    config: AgentConfig,
    audit_path: Path | None,
    runtime_config: XcodeRuntimeConfig,
    contextual_state: ContextualRetrievalState | None = None,
    shell_spec: ShellSpec | None = None,
    compact_controller: CompactController | None = None,
    cancellation_token: CancellationToken | None = None,
    compactor: LayeredCompactor | None = None,
    fallback_provider: ModelProvider | None = None,
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
    skill_registry: SkillRegistry | None = None,
    external_hook_runner: ExternalHookRunner | None = None,
    memory_manager: Any | None = None,
) -> CodingAgentHarness:
    from ..memory import MemoryManager

    memory_manager = memory_manager or MemoryManager(project_root)

    hook_manager = build_hook_manager(
        contextual_state,
        external_hook_runner,
        project_root,
        subagent=False,
    )

    sec = runtime_config.security
    return CodingAgentHarness(
        provider=llm,
        registry=registry,
        config=config,
        gate=GateConfig(
            permission_policy=permission_policy_from_security(sec),
            restricted_dirs=sec.restricted_dirs,
            hook_constraint_providers=hook_constraint_providers,
            hook_manager=hook_manager,
            external_hook_runner=external_hook_runner,
            external_hooks_cwd=project_root,
            audit_logger=JsonlAuditLogger(audit_path).write if audit_path else None,
            external_directories=external_directories_from_security(sec),
            user_rulesets=mode_rulesets_from_runtime_config(runtime_config),
        ),
        runtime=AgentRuntimeConfig(
            compactor=compactor,
            compact_controller=compact_controller,
            cancellation_token=cancellation_token,
            runtime_context_provider=build_runtime_context_provider(
                project_root,
                registry,
                shell_spec=shell_spec,
                contextual_state=contextual_state,
                modules=runtime_config.prompt.modules,
                memory_manager=memory_manager,
            ),
            fallback_provider=fallback_provider,
            project_root=project_root,
            request_hygiene=runtime_config.request_hygiene,
            skill_registry=skill_registry,
            prompt_instructions=tuple(
                i.model_dump(exclude_none=True)
                for i in runtime_config.prompt.instructions
            ),
            memory_manager=memory_manager,
        ),
    )
