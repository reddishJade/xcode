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
from uuid import uuid4
from typing import TYPE_CHECKING, Any, cast

from xcode.harness.execution_env import ExecutionEnv

from xcode.harness.config import (
    AgentConfig,
    ModeRuleRuntimeConfig,
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
    StructuredAgent,
    build_subagent_tools,
    SubagentRunner,
)
from xcode.harness.agent_runtime.events import (
    FinalStructuredEvent,
    TextDeltaStructuredEvent,
    ToolResultStructuredEvent,
    ToolUpdateStructuredEvent,
    ToolUseStructuredEvent,
)
from xcode.harness.agent_runtime.config import AgentRuntimeConfig, GateConfig
from xcode.harness.agent_runtime.prompting import build_runtime_context_provider
from xcode.harness.agent_runtime.compaction import CompactController, LayeredCompactor
from xcode.ai.providers.base import ModelProvider
from xcode.harness.observability import (
    ExternalHookRunner,
    HookRecord,
    JsonlAuditLogger,
    HookManager,
    InMemoryGrantStore,
    PermissionDecision,
    PermissionPolicy,
)
from xcode.harness.observability.permission_model import ExternalDirectory
from xcode.harness.observability.permission_model import StaticPermission
from xcode.harness.observability.permission_model import PolicyEvaluator
from xcode.harness.observability.permission_model import Rule
from xcode.agent.types import ToolSpec
from xcode.coding_agent.registry import build_project_scoped_registry
from xcode.coding_agent.tools import ShellSpec

if TYPE_CHECKING:
    from xcode.harness.memory import MemoryManager
    from xcode.harness.mcp import McpRuntimeRegistry
    from xcode.harness.agent_skills import SkillRegistry


@dataclass(frozen=True)
class SharedServices:
    pass


def build_shared_services(project_root: Path) -> SharedServices:
    return SharedServices()


class OptInServices:
    pass


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
    memory_manager: MemoryManager


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


# ── 共享基础设施 ──


def build_shared_infra(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
) -> SharedInfra:
    contextual_state = ContextualRetrievalState(project_root)
    cancellation_token = CancellationToken()
    compact_controller = CompactController()

    transcript_dir = (
        resolve_config_path(project_root, runtime_config.paths.sessions_dir)
        if runtime_config.paths.sessions_dir
        else project_root / ".local" / "sessions"
    )
    from xcode.harness.memory import MemoryManager

    memory_manager = MemoryManager(project_root)

    def _combined_on_compact(summary: str) -> None:
        """压缩时从摘要提取记忆并检测消费端反馈。

        - consolidate：从结构化摘要的 Key Decisions / Goal 节抽取记忆
        - record_explicit_references：子串匹配检测摘要是否引用了已注入的记忆
        - record_llm_references：LLM 补充检测隐式引用（当配置了 reference_judge_fn 时）
        - record_compaction_referenced_feedback：对引用的记忆做采纳 + utility 更新
        """
        memory_manager.consolidate(summary)
        memory_manager.record_explicit_references(summary)
        memory_manager.record_llm_references(summary)
        memory_manager.record_compaction_referenced_feedback()

    checkpoint_dir = (
        resolve_config_path(project_root, runtime_config.paths.sessions_dir)
        if runtime_config.paths.sessions_dir
        else project_root / ".xcode" / "checkpoints"
    )

    compactor = LayeredCompactor(
        transcript_dir=transcript_dir,
        checkpoint_dir=checkpoint_dir,
        max_recent_messages=runtime_config.agent.max_recent_messages,
        keep_recent_tokens=runtime_config.agent.keep_recent_tokens,
        reserve_tokens=runtime_config.agent.reserve_tokens,
        on_compact=_combined_on_compact,
    )
    return SharedInfra(
        contextual_state=contextual_state,
        cancellation_token=cancellation_token,
        compact_controller=compact_controller,
        compactor=compactor,
        memory_manager=memory_manager,
    )


# ── 工具注册 ──




def build_search_tools_tool(
    registry_provider: Callable[[], tuple[ToolSpec, ...]],
) -> ToolSpec:
    """按关键字搜索所有已注册工具。"""

    def search_tools(data: ToolInput) -> str:
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


def build_tool_registry(
    project_root: Path,
    llm: ModelProvider,
    llm_profiles: Mapping[str, ModelProvider] | None,
    config: AgentConfig,
    runtime_config: XcodeRuntimeConfig,
    shared_services: SharedServices,
    contextual_state: ContextualRetrievalState | None = None,
    compact_controller: CompactController | None = None,
    cancel_event: threading.Event | None = None,
    env: ExecutionEnv | None = None,
    skills_dir: Path | None = None,
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
    external_hook_runner: ExternalHookRunner | None = None,
    memory_manager: Any | None = None,
) -> tuple[
    tuple[ToolSpec, ...],
    ShellSpec,
    tuple[Callable[[], None], ...],
    SkillRegistry | None,
    McpRuntimeRegistry,
]:
    from xcode.coding_agent.tools import detect_shell
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
        shared_services,
    )

    child_registry = _build_child_registry(
        registry,
        set(runtime_config.tools.subagent_extra_tools),
    )
    registry += (build_search_tools_tool(lambda: registry),)

    subagent_closers, subagent_tools = _build_subagent_integration(
        project_root=project_root,
        llm=llm,
        llm_profiles=llm_profiles,
        config=config,
        runtime_config=runtime_config,
        shared_services=shared_services,
        child_registry=child_registry,
        contextual_state=contextual_state,
        shell_spec=shell_spec,
        cancel_event=cancel_event,
        env=env,
        hook_constraint_providers=hook_constraint_providers,
        external_hook_runner=external_hook_runner,
    )
    closers.extend(subagent_closers)
    registry += subagent_tools

    closers.append(mcp_runtime_registry.close)

    return (
        registry,
        shell_spec,
        tuple(closers),
        skill_registry,
        mcp_runtime_registry,
    )


def _discover_skills(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
    skills_dir: Path | None,
) -> SkillRegistry | None:
    """发现并注册项目与用户技能。"""
    from xcode.harness.agent_skills import (
        SkillRegistry,
        build_skill_search_dirs,
    )

    skill_registry: SkillRegistry | None = SkillRegistry()
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
) -> tuple[ToolSpec, ...]:
    """构建项目级基础工具注册表。"""
    return build_project_scoped_registry(
        project_root=project_root,
        contextual_state=contextual_state,
        shell_spec=shell_spec,
        cancel_event=cancel_event,
        env=env,
        skill_registry=skill_registry,
    )


def _build_child_registry(
    registry: tuple[ToolSpec, ...],
    subagent_extra_tools: set[str],
) -> tuple[ToolSpec, ...]:
    """从主注册表过滤出子代理可用的工具集。"""
    CORE_TOOLS = frozenset(
        {
            "read_file", "write_file", "edit_file", "apply_patch",
            "glob_files", "find_files", "list_dir", "grep_search",
            "webfetch", "websearch", "bash",
        }
    )
    allowed_tools = CORE_TOOLS | subagent_extra_tools
    return tuple(tool for tool in registry if tool.name in allowed_tools)


def _extend_registry_with_features(
    registry: tuple[ToolSpec, ...],
    project_root: Path,
    mcp_runtime_registry: McpRuntimeRegistry,
    runtime_config: XcodeRuntimeConfig,
    shared_services: SharedServices,
    memory_manager: Any | None = None,
) -> tuple[ToolSpec, ...]:
    """添加可选功能工具到注册表，复用共享实例。"""
    from xcode.harness.mcp import build_mcp_tools

    registry += build_mcp_tools(project_root, mcp_runtime_registry)
    from xcode.harness.memory import build_memory_tools

    if memory_manager is not None:
        registry += build_memory_tools(memory_manager)
    else:
        from xcode.harness.memory import MemoryManager

        registry += build_memory_tools(MemoryManager(project_root))
    return registry


def _build_subagent_integration(
    project_root: Path,
    llm: ModelProvider,
    llm_profiles: Mapping[str, ModelProvider] | None,
    config: AgentConfig,
    runtime_config: XcodeRuntimeConfig,
    shared_services: SharedServices,
    child_registry: tuple[ToolSpec, ...],
    contextual_state: ContextualRetrievalState | None,
    shell_spec: ShellSpec,
    cancel_event: threading.Event | None,
    env: ExecutionEnv | None,
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
    external_hook_runner: ExternalHookRunner | None = None,
) -> tuple[list[Callable[[], None]], tuple[ToolSpec, ...]]:
    """构建子代理运行器和工具，返回 (closers, subagent_tools)。"""
    child_llms = dict(llm_profiles or {})
    if not child_llms:
        child_llms[PROFILE_MAIN] = llm
    child_llms.setdefault(PROFILE_SUBAGENT, child_llms[PROFILE_MAIN])

    async def run_child(
        prompt,
        model_profile=PROFILE_SUBAGENT,
        cwd_override=None,
        on_update=None,
    ):
        child_root = project_root.resolve()
        child_contextual_state = contextual_state
        effective_registry = child_registry
        if cwd_override is not None:
            child_root = Path(cwd_override).resolve()
            child_contextual_state = ContextualRetrievalState(child_root)
            effective_registry = build_project_scoped_registry(
                project_root=child_root,
                contextual_state=child_contextual_state,
                shell_spec=shell_spec,
                cancel_event=cancel_event,
                env=env,
            )

        sec = runtime_config.security
        child_hook_manager = _build_hook_manager(
            child_contextual_state,
            external_hook_runner,
            child_root,
            subagent=True,
        )

        child_audit_path = resolve_config_path(
            project_root, runtime_config.observability.audit_path
        )
        from xcode.harness.memory import MemoryManager

        memory_manager = MemoryManager(child_root)

        subagent_session_id = f"subagent-{uuid4().hex[:8]}"
        child_agent = StructuredAgent(
            provider=child_llms[model_profile],
            registry=effective_registry,
            config=config,
            gate=GateConfig(
                session_id=subagent_session_id,
                permission_policy=_permission_policy_from_security(sec),
                restricted_dirs=sec.restricted_dirs,
                hook_constraint_providers=hook_constraint_providers,
                hook_manager=child_hook_manager,
                external_hook_runner=external_hook_runner,
                external_hooks_subagent=True,
                external_hooks_cwd=child_root,
                audit_logger=(
                    JsonlAuditLogger(child_audit_path).write
                    if child_audit_path
                    else None
                ),
                external_directories=_external_directories_from_security(sec),
                session_grant_store=InMemoryGrantStore(session_id=subagent_session_id),
                user_rulesets=_mode_rulesets_from_runtime_config(runtime_config),
            ),
            runtime=AgentRuntimeConfig(
                runtime_context_provider=build_runtime_context_provider(
                    child_root,
                    effective_registry,
                    shell_spec=shell_spec,
                    contextual_state=child_contextual_state,
                    modules=runtime_config.prompt.modules,
                    memory_manager=memory_manager,
                ),
                project_root=child_root,
                prompt_instructions=tuple(
                    i.model_dump(exclude_none=True)
                    for i in runtime_config.prompt.instructions
                ),
            ),
        )
        result = None
        async for event in child_agent.arun_stream(prompt):
            update = _format_child_event_update(event)
            if update and on_update is not None:
                on_update(update)
            if isinstance(event, FinalStructuredEvent):
                result = event.data
        if result is None:
            raise RuntimeError("subagent finished without final result")
        return result.answer

    managed_runner = SubagentRunner(
        run_child,
        available_profiles=tuple(child_llms),
        default_profile=PROFILE_SUBAGENT,
        max_active_jobs=config.subagent_workers,
    )
    return [managed_runner.shutdown], build_subagent_tools(managed_runner)


def _format_child_event_update(event: object) -> str | None:
    """将子 Agent 结构化事件压缩为用户可见的委派进度。"""
    if isinstance(event, TextDeltaStructuredEvent):
        text = _single_line(event.data)
        return f"text: {text}" if text else None
    if isinstance(event, ToolUseStructuredEvent):
        args = json.dumps(event.data.input, ensure_ascii=False, sort_keys=True)
        return f"tool: {event.data.name} {args[:240]}"
    if isinstance(event, ToolUpdateStructuredEvent):
        text = _single_line(event.data.partial_result)
        return f"{event.data.tool_name}: {text}" if text else None
    if isinstance(event, ToolResultStructuredEvent):
        text = _single_line(event.data.content)
        status = "ok" if event.data.status == "ok" else "error"
        return f"tool_result: {status} {text[:240]}"
    if isinstance(event, FinalStructuredEvent):
        reason = event.data.termination_reason.value
        return f"final: {reason}"
    return None


def _single_line(text: str) -> str:
    return " ".join(text.strip().split())[:240]


# ── 可选服务 ──


def load_opt_in_services(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
    shared_services: object,
) -> OptInServices:
    return OptInServices()


# ── Agent 构建 ──


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
) -> StructuredAgent:
    from xcode.harness.memory import MemoryManager

    memory_manager = memory_manager or MemoryManager(project_root)

    hook_manager = _build_hook_manager(
        contextual_state,
        external_hook_runner,
        project_root,
        subagent=False,
    )

    sec = runtime_config.security
    return StructuredAgent(
        provider=llm,
        registry=registry,
        config=config,
        gate=GateConfig(
            permission_policy=_permission_policy_from_security(sec),
            restricted_dirs=sec.restricted_dirs,
            hook_constraint_providers=hook_constraint_providers,
            hook_manager=hook_manager,
            external_hook_runner=external_hook_runner,
            external_hooks_cwd=project_root,
            audit_logger=JsonlAuditLogger(audit_path).write if audit_path else None,
            external_directories=_external_directories_from_security(sec),
            user_rulesets=_mode_rulesets_from_runtime_config(runtime_config),
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


def _build_hook_manager(
    contextual_state: ContextualRetrievalState | None,
    external_hook_runner: ExternalHookRunner | None,
    project_root: Path,
    *,
    subagent: bool,
) -> HookManager | None:
    """组合内部订阅者和外部命令 hook。"""
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


def _rule_from_runtime_config(rule: ModeRuleRuntimeConfig) -> Rule:
    return Rule(
        action=rule.action,
        effect=rule.effect,
        command=rule.command,
        subcommand=rule.subcommand,
        subcommand_in=set(rule.subcommand_in)
        if rule.subcommand_in is not None
        else None,
        flags_any=set(rule.flags_any) if rule.flags_any is not None else None,
        flags_all=set(rule.flags_all) if rule.flags_all is not None else None,
        resource_pattern=rule.resource_pattern,
    )


def _mode_rulesets_from_runtime_config(
    runtime_config: XcodeRuntimeConfig,
) -> dict[str, tuple[Rule, ...]]:
    modes = runtime_config.execution_modes
    result: dict[str, tuple[Rule, ...]] = {}
    for mode_name, ruleset in (
        ("plan", modes.plan),
        ("build", modes.build),
        ("act", modes.act),
    ):
        if ruleset.rules:
            result[mode_name] = tuple(
                _rule_from_runtime_config(rule) for rule in ruleset.rules
            )
    return result


def _external_directories_from_security(
    security: SecurityRuntimeConfig,
) -> tuple[ExternalDirectory, ...]:
    dirs: list[ExternalDirectory] = [
        ExternalDirectory(path=Path(ed.path), access=ed.access)
        for ed in security.external_directories
    ]
    # 自动添加 ~/.xcode 和 ~/.agents 为只读外部目录，使 skill 文件可读
    home = Path.home()
    for p in (home / ".xcode", home / ".agents"):
        if p.is_dir():
            ext = ExternalDirectory(path=p, access="read")
            if ext not in dirs:
                dirs.append(ext)
    return tuple(dirs)


def _permission_policy_from_security(
    security: SecurityRuntimeConfig,
) -> PermissionPolicy | None:
    """将运行时 security 配置转换为 PermissionPolicy。"""
    rules: list[StaticPermission] = []
    for rd in security.rules:
        rules.append(
            StaticPermission(
                tool=rd["tool"],
                decision=rd["decision"],
                target=rd.get("target"),
                target_type=rd.get("target_type"),
                input_contains=rd.get("input_contains"),
                input_prefix=rd.get("input_prefix"),
                input_regex=rd.get("input_regex"),
            )
        )
    global_default: str | None = security.global_default
    if global_default is None and security.resolve_approval_policy() == "always":
        global_default = "ask"
    if not rules and global_default is None:
        return None
    return PermissionPolicy(
        tuple(rules), global_default=cast(PermissionDecision, global_default)
    )
