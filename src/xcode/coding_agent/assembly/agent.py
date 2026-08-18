"""Agent 构建。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from xcode.ai.providers.base import ModelProvider
from xcode.agent.types import ToolSpec
from xcode.agent.context import (
    ActiveDiffCollector,
    ContextCollectorRegistry,
    DefaultContextAssembler,
    InstructionCollector,
    NotesCollector,
    RecentValidationCollector,
)
from xcode.coding_agent.tools import ShellSpec
from xcode.coding_agent.tools.apply_patch import extract_patch_paths
from xcode.coding_agent.execution_modes import (
    DEFAULT_MODE_FALLBACKS,
    DEFAULT_SHELL_UNRESOLVED_POLICIES,
    build_default_mode_rulesets,
)

from xcode.harness.agent_runtime import (
    AgentComposition,
    CancellationToken,
    ContextualRetrievalState,
)
from xcode.coding_agent.harness import CodingAgentHarness
from xcode.harness.agent_runtime.config import (
    GateConfig,
    GateRuntimeConfig,
    build_request_assembler,
    resolve_permission_policy,
)
from xcode.harness.agent_runtime.compaction import CompactController, LayeredCompactor
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.harness.config import AgentConfig, XcodeRuntimeConfig
from xcode.harness.session_todo import SessionTodoState
from xcode.harness.observability import (
    ExternalHookRunner,
    HookManager,
    HookRecord,
    JsonlAuditLogger,
    SignalHookManager,
)
from xcode.harness.security.permission_model import PolicyEvaluator
from xcode.harness.session.recorder import SessionRecorder
from xcode.harness.session.inbox import SessionInbox

from .security import (
    external_directories_from_security,
    mode_rulesets_from_runtime_config,
    permission_policy_from_security,
    sensitive_path_overrides_from_security,
)
from ..runtime import CodingAgentRuntimeConfig
from ..prompting import CORE_IDENTITY

if TYPE_CHECKING:
    from xcode.harness.skills import SkillRegistry


def build_hook_manager(
    contextual_state: ContextualRetrievalState | None,
    external_hook_runner: ExternalHookRunner | None,
    session_recorder: SessionRecorder,
    project_root: Path,
    *,
    subagent: bool,
) -> HookManager:
    manager = SignalHookManager()
    manager.register(
        "before_provider_request",
        session_recorder.record_provider_request,
    )
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
    session_recorder: SessionRecorder,
    session_inbox: SessionInbox,
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
    session_history: Any | None = None,
    todo_state: SessionTodoState | None = None,
) -> CodingAgentHarness:
    from xcode.harness.memory import MemoryManager

    memory_manager = memory_manager or MemoryManager(project_root)
    hook_manager = build_hook_manager(
        contextual_state,
        external_hook_runner,
        session_recorder,
        project_root,
        subagent=False,
    )

    sec = runtime_config.security
    context_collectors = ContextCollectorRegistry()
    context_collectors.register(
        InstructionCollector(
            sources=tuple(
                i.model_dump(exclude_none=True)
                for i in runtime_config.prompt.instructions
            ),
            project_root=project_root,
        )
    )
    context_collectors.register(ActiveDiffCollector(project_root))
    context_collectors.register(RecentValidationCollector())
    context_collectors.register(NotesCollector(project_root))
    if skill_registry is not None:
        from xcode.harness.skills import SkillIndexCollector

        context_collectors.register(SkillIndexCollector(skill_registry))

    runtime_context_provider = build_runtime_context_provider(
        project_root,
        registry,
        shell_spec=shell_spec,
        contextual_state=contextual_state,
        modules=runtime_config.prompt.modules,
        memory_manager=memory_manager,
        todo_context_provider=(
            todo_state.render_context if todo_state is not None else None
        ),
        identity=CORE_IDENTITY,
    )
    gate = GateConfig(
        permission_policy=resolve_permission_policy(
            project_root,
            permission_policy_from_security(sec),
        ),
        restricted_dirs=sec.restricted_dirs,
        hook_constraint_providers=hook_constraint_providers,
        external_directories=external_directories_from_security(sec),
        sensitive_path_overrides=sensitive_path_overrides_from_security(
            sec, project_root
        ),
        user_rulesets=mode_rulesets_from_runtime_config(runtime_config),
        default_mode_rulesets=build_default_mode_rulesets(project_root),
        mode_fallbacks=DEFAULT_MODE_FALLBACKS,
        shell_unresolved_policies=DEFAULT_SHELL_UNRESOLVED_POLICIES,
        tool_path_extractors={"apply_patch": extract_patch_paths},
    )
    composition = AgentComposition.create(
        primary_provider=llm,
        fallback_provider=fallback_provider,
        registry=registry,
        config=config,
        gate=gate,
        request_assembler=build_request_assembler(
            runtime_config.request_hygiene,
            context_collectors,
            DefaultContextAssembler(),
        ),
        runtime_context_provider=runtime_context_provider,
    )

    return CodingAgentHarness(
        composition=composition,
        runtime=CodingAgentRuntimeConfig(
            session_inbox=session_inbox,
            gate=GateRuntimeConfig(
                hook_manager=hook_manager,
                external_hook_runner=external_hook_runner,
                external_hooks_cwd=project_root,
                audit_logger=(
                    JsonlAuditLogger(audit_path).write if audit_path else None
                ),
            ),
            compactor=compactor,
            compact_controller=compact_controller,
            cancellation_token=cancellation_token,
            project_root=project_root,
            skill_registry=skill_registry,
            memory_manager=memory_manager,
            session_history=session_history,
            todo_state=todo_state,
        ),
    )
