from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import subprocess

from .app_contract import ReplApp
from .commands import (
    COMMAND_GROUP_EXIT,
    COMMAND_GROUP_INFO,
    COMMAND_GROUP_MODE,
    COMMAND_GROUP_MODEL,
    COMMAND_GROUP_SESSION,
    CommandContext,
    CommandEntry,
    PromptLike,
    ReplState,
    command_names,
    generate_help_text,
)
from .markdown import MarkdownRenderer
from .repl_rendering import clear_terminal_display, print_startup_banner
from .repl_sessions import (
    current_view,
    print_loaded_history,
    resume_interactively,
    resume_latest,
    resumed_message,
    select_session_interactively,
    sync_agent_history,
)
from .repl_settings import (
    handle_effort_command,
    handle_model_command,
    handle_permissions,
    handle_thinking_command,
)
from .repl_skills import activate_skill
from .repl_tools import run_tool_command
from xcode.harness.observability import (
    FileGrantStore,
    InMemoryGrantStore,
    PermissionApprovalCallback,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
)
from xcode.harness.skills import ToolSpec
from xcode.harness.session import FORK_TYPES, SessionStore
from xcode.harness.snapshot import SnapshotStore, TurnSnapshotRecord


def cmd_help(cmd: str, ctx: CommandContext) -> bool:
    """打印帮助信息。"""
    print(HELP_TEXT)
    return False


def cmd_clear(cmd: str, ctx: CommandContext) -> bool:
    """清空当前会话记录并开始新会话。"""
    ctx.store.clear()
    sync_agent_history(ctx.app, ctx.store)
    clear_terminal_display()
    print_startup_banner(ctx.app, ctx.project_root)
    return False


def cmd_fork(cmd: str, ctx: CommandContext) -> bool:
    """从当前会话创建独立分支。"""
    parts = cmd.split(maxsplit=1)
    fork_type = parts[1].strip() if len(parts) == 2 else None
    if fork_type is not None and fork_type not in FORK_TYPES:
        print(f"fork_type must be one of {sorted(FORK_TYPES)}, got {fork_type!r}")
        return False
    parent_session_id = ctx.store.session_id
    meta = ctx.store.fork_into(fork_type)
    if ctx.snapshot_store is not None:
        ctx.snapshot_store.fork_session(parent_session_id, meta.id)
    sync_agent_history(ctx.app, ctx.store)
    label = f" ({fork_type})" if fork_type else ""
    print(f'Forked: "{meta.title}"{label}')
    return False


def cmd_rewind(cmd: str, ctx: CommandContext) -> bool:
    """回退最近的 N 轮用户交互。"""
    parts = cmd.split()
    turns = int(parts[1]) if len(parts) > 1 else 1
    removed = ctx.store.rewind_turns(turns)
    if ctx.snapshot_store is not None:
        ctx.snapshot_store.rewind_to_turn_count(
            ctx.store.session_id,
            ctx.store.user_turn_count(),
        )
    sync_agent_history(ctx.app, ctx.store)
    turn_label = "turn" if turns == 1 else "turns"
    print(f"Rewound {turns} user {turn_label} ({removed} transcript records removed).")
    return False


def cmd_resume(cmd: str, ctx: CommandContext) -> bool:
    """从最近的或指定的会话恢复。"""
    parts = cmd.split(maxsplit=1)
    if len(parts) == 2:
        target = parts[1].strip()
        if target == "last":
            view = resume_latest(ctx.store)
            if view:
                print(resumed_message(view))
                print_loaded_history(ctx.store)
                sync_agent_history(ctx.app, ctx.store)
            else:
                print("No conversations found.")
            return False
        ctx.store.resume(target)
        print(resumed_message(current_view(ctx.store)))
        print_loaded_history(ctx.store)
        sync_agent_history(ctx.app, ctx.store)
        return False
    resume_interactively(ctx.store, ctx.prompt_session)
    sync_agent_history(ctx.app, ctx.store)
    return False


def cmd_tree(cmd: str, ctx: CommandContext) -> bool:
    """显示会话分支树。"""
    nodes = ctx.store.get_tree()
    if not nodes:
        print("No session tree available (no metadata).")
        return False

    for node in nodes:
        indent = "  " * node.depth
        prefix = "└─ " if node.depth > 0 else ""
        branch = "🌿 " if node.fork_type else "  "
        marker = " ← current" if node.is_current else ""
        label = f"{branch}{node.title}"
        print(f"{indent}{prefix}{label}{marker}")
    return False


def cmd_branch(cmd: str, ctx: CommandContext) -> bool:
    """列出或切换到指定分支。"""
    parts = cmd.split(maxsplit=1)
    if len(parts) == 1 or parts[1].strip() in {"list", "tree"}:
        return cmd_tree("/tree", ctx)

    target = parts[1].strip()
    try:
        view = ctx.store.switch_branch(target)
    except ValueError as exc:
        print(str(exc))
        return False
    sync_agent_history(ctx.app, ctx.store)
    print(resumed_message(view))
    print_loaded_history(ctx.store)
    return False


def cmd_continue(cmd: str, ctx: CommandContext) -> bool:
    """切换到当前项目最新的有意义会话。"""
    view = ctx.store.find_latest_for_project(ctx.project_root)
    if view is None:
        print("No prior session found for this project.")
        return False
    if view.id == ctx.store.session_id:
        print(f"Already on the latest session: {view.title}")
        return False
    ctx.store.resume(view.id)
    print(resumed_message(view))
    print_loaded_history(ctx.store)
    sync_agent_history(ctx.app, ctx.store)
    return False


def cmd_sessions(cmd: str, ctx: CommandContext) -> bool:
    """交互式选择并恢复历史会话。"""
    sessions = ctx.store.list_session_infos()
    if not sessions:
        print("No conversations found.")
        return False

    selected = select_session_interactively(sessions, "Select session to resume:")
    if selected is None:
        return False

    ctx.store.resume(selected.id)
    sync_agent_history(ctx.app, ctx.store)
    print(resumed_message(selected))
    print_loaded_history(ctx.store)
    return False


def cmd_model(cmd: str, ctx: CommandContext) -> bool:
    """显示或切换当前模型。"""
    handle_model_command(cmd, ctx.app)
    return False


def cmd_effort(cmd: str, ctx: CommandContext) -> bool:
    """显示或设置 reasoning effort 级别。"""
    handle_effort_command(cmd, ctx.app)
    return False


def cmd_thinking(cmd: str, ctx: CommandContext) -> bool:
    """显示或切换 thinking 开/关。"""
    handle_thinking_command(cmd, ctx.app)
    return False


def cmd_plan(cmd: str, ctx: CommandContext) -> bool:
    """进入 Plan Mode（只读检查，禁止编辑和 shell）。"""
    ctx.state.mode = "plan"
    print(
        "Plan Mode enabled. Read-only inspection tools are available; edits and shell are blocked."
    )
    return False


def cmd_build(cmd: str, ctx: CommandContext) -> bool:
    """进入 Build Mode（允许普通文件变更，高风险操作需审批）。"""
    ctx.state.mode = "build"
    print(
        "Build Mode enabled. Ordinary file mutations are allowed; high-risk actions require approval."
    )
    return False


def cmd_act(cmd: str, ctx: CommandContext) -> bool:
    """进入 Act Mode 恢复工具使用权限，支持 --clear 选项。"""
    is_clear = False
    parts = cmd.split(maxsplit=1)
    if len(parts) == 2 and parts[1].strip() == "--clear":
        is_clear = True

    choice = "1" if is_clear else None
    if choice is None:
        choice = _select_act_transition()
        if choice is None:
            print("Cancelled.")
            return False

    if choice == "1":
        records = ctx.store.load_records()
        last_assistant_content = None
        for record in reversed(records):
            if record.type == "assistant":
                last_assistant_content = record.content
                break

        if not last_assistant_content or not str(last_assistant_content).strip():
            print(
                "Error: No plan found in the last assistant reply. Cannot Clear and Act."
            )
            return False

        parent_id = ctx.store.current_path.stem.removeprefix("session-")
        plan_text = (
            f"# Approved Plan (Forked from {parent_id})\n"
            f"Date: {datetime.now().isoformat(timespec='seconds')}\n\n"
            f"{last_assistant_content}"
        )

        plan_file = ctx.store.artifacts_dir / f"plan-{parent_id}.md"
        try:
            plan_file.write_text(plan_text, encoding="utf-8")
        except OSError as exc:
            print(f"Warning: Failed to write plan artifact: {exc}")

        meta = ctx.store.fork_clean_into(
            "isolate", title=f"Act Continuation of Plan {parent_id}"
        )
        sync_agent_history(ctx.app, ctx.store)

        ctx.state.approved_plan = str(last_assistant_content)
        ctx.state.mode = "act"
        print(f'Clean Fork created: "{meta.title}"')
        print("Act Mode enabled with approved plan.")
    elif choice == "2":
        ctx.state.mode = "act"
        print("Act Mode enabled. Normal tool use restored within policy.")
    elif choice == "3":
        ctx.state.mode = "build"
        print(
            "Build Mode enabled. Ordinary file mutations are allowed; high-risk actions require approval."
        )
    elif choice == "4":
        print("Continuing in Plan Mode.")
    else:
        print(f"Invalid choice: {choice}")
    return False


def _select_act_transition() -> str | None:
    """显示 Act 模式切换菜单。"""
    import questionary

    return questionary.select(
        "Select action:",
        choices=[
            questionary.Choice(
                title="Clear and Act (Clear context, keep plan, and act)",
                value="1",
            ),
            questionary.Choice(
                title="Keep and Act (Keep current context and act directly)",
                value="2",
            ),
            questionary.Choice(title="Build Mode", value="3"),
            questionary.Choice(title="Continue in Plan Mode", value="4"),
        ],
        default="2",
    ).ask()


def cmd_verbose(cmd: str, ctx: CommandContext) -> bool:
    """设置输出详细程度: normal, verbose, debug。"""
    parts = cmd.split(maxsplit=1)
    if len(parts) == 2:
        val = parts[1].strip().lower()
        if val in ("normal", "verbose", "debug"):
            ctx.state.verbosity = val
            print(f"Verbosity set to {val}.")
        elif val == "on":
            ctx.state.verbosity = "verbose"
            print("Verbose mode on.")
        elif val == "off":
            ctx.state.verbosity = "normal"
            print("Verbose mode off.")
        else:
            print(f"Unknown level: {val}. Use normal, verbose, debug, on, or off.")
    else:
        print(f"Current verbosity: {ctx.state.verbosity}")
        print("Usage: /verbose normal|verbose|debug|on|off")
    return False


def cmd_debug(cmd: str, ctx: CommandContext) -> bool:
    """切换 debug 模式（显示推理预览和展开工具结果）。"""
    parts = cmd.split(maxsplit=1)
    if len(parts) == 2 and parts[1] == "on":
        ctx.state.verbosity = "debug"
        print("Debug mode on: reasoning preview and expanded tool results shown.")
    elif len(parts) == 2 and parts[1] == "off":
        ctx.state.verbosity = "normal"
        print("Debug mode off.")
    elif ctx.state.verbosity == "debug":
        ctx.state.verbosity = "normal"
        print("Debug mode off.")
    else:
        ctx.state.verbosity = "debug"
        print("Debug mode on: reasoning preview and expanded tool results shown.")
    return False


def cmd_queue(cmd: str, ctx: CommandContext) -> bool:
    """启用或禁用 agent 回合期间的输入队列。"""
    parts = cmd.split(maxsplit=1)
    if len(parts) == 1:
        ctx.state.queue_mode = not ctx.state.queue_mode
        print(f"Queue mode {'enabled' if ctx.state.queue_mode else 'disabled'}.")
        return False
    value = parts[1].strip().lower()
    if value not in {"on", "off"}:
        print("Usage: /queue on|off")
        return False
    ctx.state.queue_mode = value == "on"
    print(f"Queue mode {'enabled' if ctx.state.queue_mode else 'disabled'}.")
    return False


def cmd_compact(cmd: str, ctx: CommandContext) -> bool:
    """手动触发上下文压缩和会话日志裁剪。"""
    agent = getattr(ctx.app, "agent", None)
    should_request_active = _should_request_active_compaction(ctx)
    if (
        should_request_active
        and agent is not None
        and hasattr(agent, "request_compaction")
    ):
        agent.request_compaction()
        message_count, recent_window = _active_context_compaction_size(ctx)
        if message_count > recent_window + 1:
            print(
                "Active context compaction requested for the next agent run "
                f"({message_count} messages exceed recent window {recent_window})."
            )
        else:
            print(
                "Active context compaction requested for the next agent run "
                "(large tool results present)."
            )
    compacted = ctx.store.compact_current_session(max_tool_result_chars=200)
    if compacted > 0:
        print(f"Compacted {compacted} large tool results in the session log.")
    elif should_request_active:
        print("No large tool results to compact in the session log.")
    else:
        print(
            "No context compaction needed: active context is within the recent-message "
            "window and the session log has no large tool results."
        )
    return False


def _should_request_active_compaction(ctx: CommandContext) -> bool:
    """判断当前活跃上下文是否超过摘要保留窗口。"""
    agent = getattr(ctx.app, "agent", None)
    if agent is None or not hasattr(agent, "request_compaction"):
        return False
    message_count, recent_window = _active_context_compaction_size(ctx)
    return message_count > recent_window + 1 or _has_large_tool_result(ctx)


def _active_context_compaction_size(ctx: CommandContext) -> tuple[int, int]:
    """返回可见会话消息数量和活跃上下文最近消息窗口。"""
    records = ctx.store.load_records()
    message_count = sum(
        1 for record in records if record.type in {"system", "user", "assistant"}
    )
    agent = getattr(ctx.app, "agent", None)
    compactor = getattr(agent, "compactor", None) if agent is not None else None
    recent_window = getattr(compactor, "max_recent_messages", 8)
    if not isinstance(recent_window, int):
        recent_window = 8
    return message_count, recent_window


def _has_large_tool_result(
    ctx: CommandContext, max_tool_result_chars: int = 200
) -> bool:
    """判断会话日志中是否存在需要裁剪的大工具结果。"""
    for record in ctx.store.load_records():
        if record.type != "event" or not isinstance(record.content, dict):
            continue
        if record.content.get("type") != "tool_result":
            continue
        data = record.content.get("data")
        if not isinstance(data, dict) or "content" not in data:
            continue
        if len(str(data["content"])) > max_tool_result_chars:
            return True
    return False


def cmd_permissions(cmd: str, ctx: CommandContext) -> bool:
    """列出、撤销或清除权限规则。"""
    handle_permissions(
        cmd,
        ctx.session_grant_store,
        ctx.permanent_grant_store,
        static_policy=ctx.static_policy,
        restricted_dirs=ctx.restricted_dirs,
    )
    return False


def cmd_hooks(cmd: str, ctx: CommandContext) -> bool:
    """显示外部命令 hook 配置来源和最近运行状态。"""
    diagnostics = ctx.app.hook_diagnostics()
    if not diagnostics:
        print("No external hooks configured.")
        return False

    print(f"External hooks ({len(diagnostics)}):")
    for diagnostic in diagnostics:
        matcher = diagnostic.matcher or "*"
        status = (
            f"{diagnostic.last_status} at {diagnostic.last_run_at}"
            if diagnostic.last_run_at
            else diagnostic.last_status
        )
        print(
            f"  [{diagnostic.index}] {diagnostic.event} "
            f"{'enabled' if diagnostic.enabled else 'disabled'} "
            f"matcher={matcher} policy={diagnostic.failure_policy} "
            f"subagents={'yes' if diagnostic.inherit_to_subagents else 'no'}"
        )
        print(
            f"      source={diagnostic.source} runs={diagnostic.run_count} last={status}"
        )
        if diagnostic.last_error:
            print(f"      error={diagnostic.last_error}")
    return False


def cmd_tool(cmd: str, ctx: CommandContext) -> bool:
    """直接执行一个已注册的工具。"""
    output = run_tool_command(cmd, ctx.app)
    ctx.store.append("event", {"type": "tool_command", "data": cmd})
    ctx.store.append("event", {"type": "tool_result", "data": output})
    ctx.renderer.render(output)
    return False


def cmd_skill(cmd: str, ctx: CommandContext) -> bool:
    """显式激活一个已发现的技能。"""
    parts = cmd.split(maxsplit=1)
    if len(parts) != 2 or not parts[1].strip():
        print("Usage: /skill NAME")
        return False
    result = activate_skill(ctx.app, ctx.store, parts[1].strip())
    print(result.message)
    return False


def cmd_exit(cmd: str, ctx: CommandContext) -> bool:
    """退出 REPL。"""
    return True


def _parse_undo_count(cmd: str) -> int:
    parts = cmd.split()
    if len(parts) <= 1:
        return 1
    try:
        n = int(parts[1])
        return max(n, 1)
    except ValueError:
        return 1


def cmd_undo(cmd: str, ctx: CommandContext) -> bool:
    """回退最近 N 轮用户轮次的文件变更（基于快照恢复）。"""
    if ctx.snapshot_store is None:
        print(
            "Snapshot undo requires a git repository. This project is not a git repo."
        )
        return False

    parts = cmd.split()
    if len(parts) >= 2 and parts[1] == "--list":
        records = ctx.snapshot_store.list_records(ctx.store.session_id)
        if not records:
            print("No snapshot records found.")
        else:
            print(f"Snapshot records ({len(records)} total):")
            for r in reversed(records):
                status = "UNDONE" if r.undone else "active"
                print(
                    f"  turn {r.turn_id} [{status}]: "
                    f"{len(r.changed_files)} files, "
                    f"{len(r.skipped_files)} skipped"
                )
        return False

    n = _parse_undo_count(cmd)
    records = ctx.snapshot_store.get_undoable_records(ctx.store.session_id, n)
    if not records:
        if ctx.snapshot_store.list_records(ctx.store.session_id):
            print("Nothing to undo (all turns already undone).")
        else:
            print("Nothing to undo (no snapshot records).")
        return False

    from typing import cast

    agent = getattr(ctx.app, "agent", None)
    approval_callback = cast(
        "PermissionApprovalCallback | None",
        getattr(agent, "approval_callback", None) if agent else None,
    )

    for record in reversed(records):
        result = _revert_turn(ctx, approval_callback, record)
        _report_undo_result(record, result)
        if result.fatal_error:
            print("Fatal error during undo. Stack preserved.")
            return False
        if result.skipped:
            print(f"Turn {record.turn_id}: undo incomplete; record remains active.")
            continue
        record.undone = True
        ctx.snapshot_store.update_record(ctx.store.session_id, record)
    return False


@dataclass
class _RevertResult:
    restored: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    fatal_error: bool = False


def _revert_turn(
    ctx: CommandContext,
    approval_callback: PermissionApprovalCallback | None,
    record: TurnSnapshotRecord,
) -> _RevertResult:
    store = ctx.snapshot_store
    assert store is not None, "snapshot_store required for undo"
    svc = store.service(ctx.store.session_id)
    result = _RevertResult()

    for entry in record.changed_files:
        try:
            # Layer 1: 路径安全校验
            svc._validate_path(entry.path)

            # Layer 2: 确认路径在 changed_files 中
            all_paths = {c.path for c in record.changed_files}
            if entry.path not in all_paths:
                result.skipped.append((entry.path, "path not in turn changed_files"))
                continue

            # Layer 3: 冲突检测 — 当前文件必须与 post 快照一致
            if svc.has_conflict(record.post_snapshot_id, entry.path):
                result.skipped.append((entry.path, "conflict: file changed after turn"))
                continue

            # Layer 4: 权限检查
            if entry.kind == "created":
                tool_name = "delete_file"
                tool_input: dict[str, object] = {"path": entry.path}
            elif entry.kind == "deleted":
                result.skipped.append(
                    (entry.path, "file was deleted during the turn — cannot restore")
                )
                continue
            else:
                tool_name = "write_file"
                tool_input = {"path": entry.path}

            tool_spec = next(
                (
                    spec
                    for spec in tuple(getattr(ctx.app, "registry", ()) or ())
                    if spec.name == tool_name
                ),
                None,
            )
            if tool_spec is None:
                tool_spec = ToolSpec(
                    name=tool_name,
                    description="Restore a workspace file from a snapshot.",
                    input_hint='JSON: {"path": "relative/path"}',
                    handler=lambda _input: "",
                    schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                )

            undo_agent = getattr(ctx.app, "agent", None)
            engine = PermissionEngine(
                PermissionEngineConfig(
                    static_policy=ctx.static_policy,
                    restricted_dirs=ctx.restricted_dirs,
                    project_root=ctx.project_root,
                    external_directories=getattr(undo_agent, "external_directories", ())
                    if undo_agent is not None
                    else (),
                    session_grant_store=ctx.session_grant_store,
                    permanent_grant_store=ctx.permanent_grant_store,
                )
            )
            perm_result = engine.decide(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_spec=tool_spec,
                approval_callback=approval_callback,
            )
            if perm_result.blocked:
                result.skipped.append(
                    (entry.path, f"permission denied: {perm_result.reason}")
                )
                continue

            # Layer 5: 执行
            if entry.kind == "created":
                abs_path = (ctx.project_root / entry.path).resolve()
                if abs_path.exists():
                    abs_path.unlink()
                    result.restored.append(entry.path)
                else:
                    result.skipped.append((entry.path, "file already removed"))
            else:
                svc.restore_file(record.pre_snapshot_id, entry.path)
                result.restored.append(entry.path)

        except (ValueError, OSError, subprocess.CalledProcessError) as e:
            result.skipped.append((entry.path, str(e)))
            continue

    return result


def _report_undo_result(record: TurnSnapshotRecord, result: _RevertResult) -> None:
    if result.fatal_error:
        print(f"Turn {record.turn_id}: fatal error, stack preserved.")
        return
    if result.restored:
        print(f"Turn {record.turn_id}: reverted {len(result.restored)} file(s):")
        for p in result.restored:
            print(f"  restored: {p}")
    if result.skipped:
        print(f"Turn {record.turn_id}: {len(result.skipped)} file(s) skipped:")
        for path, reason in result.skipped:
            print(f"  skipped: {path} ({reason})")


COMMAND_REGISTRY: dict[str, CommandEntry] = {
    "/help": CommandEntry(
        handler=cmd_help, desc="Show this help.", group=COMMAND_GROUP_INFO
    ),
    "/clear": CommandEntry(
        handler=cmd_clear,
        desc="Start a new session transcript.",
        group=COMMAND_GROUP_SESSION,
    ),
    "/continue": CommandEntry(
        handler=cmd_continue,
        desc="Resume the latest session for this project.",
        group=COMMAND_GROUP_SESSION,
    ),
    "/new": CommandEntry(
        handler=cmd_clear,
        desc="Start a new session transcript.",
        group=COMMAND_GROUP_SESSION,
    ),
    "/fork": CommandEntry(
        handler=cmd_fork,
        desc="Fork current session into an independent branch.",
        args_desc="[explore|verify|isolate]",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION,
    ),
    "/rewind": CommandEntry(
        handler=cmd_rewind,
        desc="Remove the last N user turns from the transcript.",
        args_desc="N",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION,
    ),
    "/resume": CommandEntry(
        handler=cmd_resume,
        desc="Choose a recent conversation to resume.",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION,
    ),
    "/sessions": CommandEntry(
        handler=cmd_sessions,
        desc="List and resume recent conversations.",
        group=COMMAND_GROUP_SESSION,
    ),
    "/tree": CommandEntry(
        handler=cmd_tree,
        desc="Show session fork tree.",
        group=COMMAND_GROUP_SESSION,
    ),
    "/branch": CommandEntry(
        handler=cmd_branch,
        desc="List or switch session branches.",
        args_desc="list|tree|<id|title>",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION,
    ),
    "/model": CommandEntry(
        handler=cmd_model,
        desc="Show current model info.",
        args_desc="[profile/]name[:thinking] [--thinking <level>]",
        accepts_args=True,
        group=COMMAND_GROUP_MODEL,
    ),
    "/effort": CommandEntry(
        handler=cmd_effort,
        desc="Show current reasoning effort.",
        args_desc="<off|minimal|low|medium|high|xhigh|max>",
        accepts_args=True,
        group=COMMAND_GROUP_MODE,
    ),
    "/thinking": CommandEntry(
        handler=cmd_thinking,
        desc="Show current thinking state (on/off).",
        args_desc="on|off",
        accepts_args=True,
        group=COMMAND_GROUP_MODE,
    ),
    "/plan": CommandEntry(
        handler=cmd_plan,
        desc="Enter Plan Mode: read-only inspection tools, no edits or shell.",
        group=COMMAND_GROUP_MODE,
    ),
    "/build": CommandEntry(
        handler=cmd_build,
        desc="Enter Build Mode: ordinary file mutations allowed, high-risk actions require approval.",
        group=COMMAND_GROUP_MODE,
    ),
    "/act": CommandEntry(
        handler=cmd_act,
        desc="Enter Act Mode and allow normal tool use within policy.",
        accepts_args=True,
        group=COMMAND_GROUP_MODE,
    ),
    "/verbose": CommandEntry(
        handler=cmd_verbose,
        desc="Set output verbosity level: normal, verbose, or debug.",
        args_desc="normal|verbose|debug",
        accepts_args=True,
        group=COMMAND_GROUP_MODE,
    ),
    "/debug": CommandEntry(
        handler=cmd_debug,
        desc="Toggle debug mode (reasoning preview + expanded tool results).",
        args_desc="on|off",
        accepts_args=True,
        group=COMMAND_GROUP_MODE,
    ),
    "/queue": CommandEntry(
        handler=cmd_queue,
        desc="Enable queued input while an agent turn is streaming.",
        args_desc="on|off",
        accepts_args=True,
        group=COMMAND_GROUP_MODE,
    ),
    "/compact": CommandEntry(
        handler=cmd_compact,
        desc="Manually request context compaction and shrink the session log.",
        group=COMMAND_GROUP_SESSION,
    ),
    "/permissions": CommandEntry(
        handler=cmd_permissions,
        desc="List / revoke / clear permission rules.",
        accepts_args=True,
        group=COMMAND_GROUP_INFO,
    ),
    "/hooks": CommandEntry(
        handler=cmd_hooks,
        desc="Show external hook sources and recent status.",
        group=COMMAND_GROUP_INFO,
    ),
    "/tool": CommandEntry(
        handler=cmd_tool,
        desc="Run one registered tool directly, or list tools.",
        args_desc="NAME INPUT|list",
        accepts_args=True,
        group=COMMAND_GROUP_INFO,
    ),
    "/skill": CommandEntry(
        handler=cmd_skill,
        desc="Activate a discovered skill for this session.",
        args_desc="NAME",
        accepts_args=True,
        group=COMMAND_GROUP_INFO,
    ),
    "/undo": CommandEntry(
        handler=cmd_undo,
        desc="回退最近 N 轮用户轮次的文件变更（基于快照恢复）。",
        args_desc="[N|--list]",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION,
    ),
    "/exit": CommandEntry(
        handler=cmd_exit, desc="退出 REPL.", group=COMMAND_GROUP_EXIT
    ),
    "/quit": CommandEntry(
        handler=cmd_exit, desc="退出 REPL.", visible=False, group=COMMAND_GROUP_EXIT
    ),
}

COMMAND_NAMES = command_names(COMMAND_REGISTRY)
HELP_TEXT = generate_help_text(COMMAND_REGISTRY)
COMMAND_REGISTRY_EXPORT = COMMAND_REGISTRY


def handle_command(
    command: str,
    store: SessionStore,
    app: ReplApp,
    renderer: MarkdownRenderer,
    state: ReplState,
    prompt_session: PromptLike,
    session_grant_store: InMemoryGrantStore | None = None,
    permanent_grant_store: FileGrantStore | None = None,
    static_policy: PermissionPolicy | None = None,
    restricted_dirs: tuple[str, ...] = (),
    snapshot_store: SnapshotStore | None = None,
) -> bool:
    ctx = CommandContext(
        store=store,
        app=app,
        renderer=renderer,
        state=state,
        prompt_session=prompt_session,
        project_root=store.project_root,
        session_grant_store=session_grant_store,
        permanent_grant_store=permanent_grant_store,
        static_policy=static_policy,
        restricted_dirs=restricted_dirs,
        snapshot_store=snapshot_store,
    )
    for prefix in sorted(COMMAND_REGISTRY, key=len, reverse=True):
        entry = COMMAND_REGISTRY[prefix]
        if command == prefix or (
            entry.accepts_args and command.startswith(prefix + " ")
        ):
            return entry.handler(command, ctx)
    print(f"Unknown command: {command}")
    return False
