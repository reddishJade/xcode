from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import questionary

from .app_contract import ReplApp
from .commands import (
    COMMAND_GROUP_EXIT,
    COMMAND_GROUP_INFO,
    COMMAND_GROUP_MODE,
    COMMAND_GROUP_MODEL,
    COMMAND_GROUP_SESSION_BRANCH,
    COMMAND_GROUP_SESSION_LIFECYCLE,
    COMMAND_GROUP_SESSION_ROLLBACK,
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
from .setup_wizard import CONFIG_FILENAME, _load_existing_config, _save_config
from .config_cmd import _cmd_add, _cmd_edit, _cmd_delete
from .config_cmd import BOOL_FIELDS
from .repl_tools import run_tool_command
from xcode.harness.observability import (
    FileGrantStore,
    InMemoryGrantStore,
    PermissionApprovalCallback,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionPolicy,
)
from xcode.harness.memory import MemoryLayer, MemoryLayerFilter, MemoryManager
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


def cmd_rename(cmd: str, ctx: CommandContext) -> bool:
    """重命名当前会话。"""
    parts = cmd.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        current = ctx.store.current_metadata()
        if current:
            print(f'Current title: "{current.title}"')
        print("Usage: /rename <title>")
        return False
    new_title = parts[1].strip()
    meta = ctx.store.rename_session(new_title)
    if meta is None:
        print("No active session to rename.")
        return False
    print(f'Session renamed to: "{meta.title}"')
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


def cmd_config(cmd: str, ctx: CommandContext) -> bool:
    """管理 provider 配置 profile。"""
    config_path = ctx.project_root / CONFIG_FILENAME
    config = _load_existing_config(config_path)

    parts = cmd.split(maxsplit=2)
    sub = parts[1].strip() if len(parts) >= 2 else ""

    if not sub or sub == "list":
        _config_cmd_list(config, config_path)
        return False

    if sub == "reload":
        print("Reloading config from file...")
        config = _load_existing_config(config_path)
        _config_cmd_list(config, config_path)
        print("(some changes may require restart)")
        return False

    if sub == "add":
        parts = cmd.split(maxsplit=2)
        name = parts[2] if len(parts) >= 3 else questionary.text("Profile name:").ask()
        if not name:
            return False
        _cmd_add(config_path, name)
        return False

    if sub == "edit":
        parts = cmd.split(maxsplit=2)
        if len(parts) >= 3:
            name = parts[2]
        else:
            profiles = config.get("provider", {}).get("model_profiles", {})
            if not profiles:
                print("No profiles found. Use '/config add <name>' first.")
                return False
            name = questionary.select(
                "Select profile:", choices=sorted(profiles.keys())
            ).ask()
        if not name:
            return False
        _cmd_edit(config_path, name)
        return False

    if sub == "delete":
        parts = cmd.split(maxsplit=2)
        if len(parts) >= 3:
            name = parts[2]
        else:
            profiles = config.get("provider", {}).get("model_profiles", {})
            if not profiles:
                print("No profiles found.")
                return False
            name = questionary.select(
                "Select profile to delete:", choices=sorted(profiles.keys())
            ).ask()
        if not name:
            return False
        _cmd_delete(config_path, name)
        return False

    if sub == "set":
        set_parts = cmd.split(maxsplit=4)
        if len(set_parts) >= 5:
            _, _, name, field, value = set_parts
            _config_cmd_set(config, config_path, name, field, value)
            return False
        _config_cmd_set_interactive(config, config_path)
        return False

    print(
        "Usage: /config [list|add <name>|edit <name>|delete <name>"
        "|set <profile> <field> <value>|reload]"
    )
    return False


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{'*' * max(0, len(key) - 4)}{key[-4:]}"


BOOL_CONFIG_FIELDS = frozenset({"thinking", "clear_thinking", "tool_stream"})


def _coerce_config_value(field: str, value: str) -> Any:
    if field in BOOL_CONFIG_FIELDS:
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Invalid bool value for '{field}': {value}")
    if value.lower() == "null":
        return None
    return value


def _config_cmd_list(config: dict[str, Any], config_path: Path) -> None:
    profiles = config.get("provider", {}).get("model_profiles", {})
    if not profiles:
        print(f"No profiles found in {config_path.name}.")
        return

    print(f"Profiles in {config_path.name}:\n")
    for name, profile in profiles.items():
        if isinstance(profile, str):
            print(f"  {name}: (inherits from main, model={profile})")
            continue
        print(f"  {name}:")
        for fname in (
            "transport",
            "chat_model",
            "base_url",
            "api_key",
            "thinking",
            "reasoning_effort",
            "clear_thinking",
            "tool_stream",
        ):
            val = profile.get(fname)
            if val is None:
                continue
            if fname == "api_key" and val:
                val = _mask_key(str(val))
            print(f"    {fname:20s}: {val}")
        print()


def _config_cmd_set(
    config: dict[str, Any], config_path: Path, name: str, field: str, value: str
) -> None:
    profiles = config.setdefault("provider", {}).setdefault("model_profiles", {})

    if name not in profiles:
        available = ", ".join(sorted(profiles.keys()))
        print(f"Profile '{name}' not found. Available: {available}")
        return

    profile = profiles[name]
    if isinstance(profile, str):
        print(f"Profile '{name}' is a string alias. Edit main first.")
        return

    try:
        coerced = _coerce_config_value(field, value)
    except ValueError as exc:
        print(str(exc))
        return

    if coerced is None:
        profile.pop(field, None)
    else:
        profile[field] = coerced

    _save_config(config, config_path)
    print(f"  {name}.{field} = {value} (saved to {config_path.name})")


def _config_cmd_set_interactive(config: dict[str, Any], config_path: Path) -> None:
    profiles = config.setdefault("provider", {}).setdefault("model_profiles", {})
    if not profiles:
        print("No profiles found. Use '/config add <name>' first.")
        return

    name = questionary.select("Select profile:", choices=sorted(profiles.keys())).ask()
    if name is None:
        return

    profile = profiles[name]
    if isinstance(profile, str):
        print(f"Profile '{name}' is a string alias. Edit main first.")
        return

    SET_FIELDS = (
        ("transport", "Transport"),
        ("chat_model", "Chat Model"),
        ("base_url", "Base URL"),
        ("api_key", "API Key"),
        ("thinking", "Thinking"),
        ("reasoning_effort", "Reasoning Effort"),
        ("clear_thinking", "Clear Thinking"),
        ("tool_stream", "Tool Stream"),
    )
    field_choices = [
        questionary.Choice(title=f"{label} ({profile.get(key, 'not set')})", value=key)
        for key, label in SET_FIELDS
    ]
    field = questionary.select("Select field to change:", choices=field_choices).ask()
    if field is None:
        return

    current = profile.get(field, "")
    current_str = str(current) if current is not None else "(not set)"
    if field == "api_key":
        value = questionary.password(
            f"API Key (current: {_mask_key(current_str)}):"
        ).ask()
    elif field in BOOL_FIELDS:
        default_choice = "true" if current else "false"
        value = questionary.select(
            f"{field}:",
            choices=["true", "false"],
            default=default_choice,
        ).ask()
    else:
        value = questionary.text(f"{field} (current: {current_str}):").ask()
    if value is None:
        return
    if not value and field != "api_key":
        return

    try:
        coerced = _coerce_config_value(field, value)
    except ValueError as exc:
        print(str(exc))
        return

    if coerced is None:
        profile.pop(field, None)
    else:
        profile[field] = coerced

    _save_config(config, config_path)
    print(f"  {name}.{field} = {value} (saved to {config_path.name})")


def cmd_plan(cmd: str, ctx: CommandContext) -> bool:
    """进入 Plan Mode（只读检查，禁止编辑和 shell）。"""
    ctx.state.mode = "plan"
    print(
        "Plan Mode enabled. Read-only inspection tools are available; edits and shell are blocked."
    )
    parts = cmd.split(maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        ctx.state.pending_inject = parts[1].strip()
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

    choice = "1" if is_clear else "2"

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
    """手动触发上下文压缩，立即执行完整压缩管线并显示结构化摘要。"""
    from xcode.agent.compaction import estimate_message_tokens
    from xcode.agent.message_converter import COMPACTION_SUMMARY_PREFIX
    from xcode.harness.agent_runtime.agent_helpers import to_dict
    from xcode.harness.agent_runtime.message_codec import (
        messages_from_compacted_dicts,
    )

    agent = getattr(ctx.app, "agent", None)
    if agent is None:
        print("No agent available.")
        return False

    # 1) 获取 agent 当前消息
    history_messages = getattr(agent, "history_messages", None)
    if not callable(history_messages):
        print("Agent does not expose history.")
        return False
    before_msgs = history_messages()
    if not before_msgs:
        print("No messages to compact.")
        return False

    before_tokens = estimate_message_tokens(
        [to_dict(m) for m in before_msgs]
    )

    # 2) 检查是否需要压缩
    compactor = getattr(agent, "compactor", None)
    if compactor is None:
        print("No compactor configured.")
        return False

    load_history = getattr(agent, "load_history", None)
    if not callable(load_history):
        print("Agent does not support history replacement.")
        return False

    # 3) 立即运行压缩
    dict_messages = [to_dict(m) for m in before_msgs]
    compacted_dicts = compactor(dict_messages)
    after_msgs = messages_from_compacted_dicts(compacted_dicts)

    # 4) 提取结构化摘要
    summary_text = _extract_compact_summary(compacted_dicts)
    after_tokens = estimate_message_tokens(
        [to_dict(m) for m in after_msgs]
    )

    # 5) 替换 agent 历史
    load_history(after_msgs)

    # 6) 也裁剪会话日志
    ctx.store.compact_current_session(max_tool_result_chars=200)

    # 7) 打印结构化摘要——类似 pi 的格式
    saved = before_tokens - after_tokens
    print(f"\n [compaction]\n")
    print(f" Compacted from {before_tokens:,} tokens")
    print()
    if summary_text:
        # 去除 [Compressed] 前缀，打印清晰的摘要
        clean = summary_text
        if clean.startswith("[Compressed]"):
            clean = clean[len("[Compressed]"):].strip()
        # 按行打印，每行不超过终端宽度
        for line in clean.splitlines():
            stripped = line.rstrip()
            if stripped:
                print(f" {stripped}")
            else:
                print()
    else:
        print(" (no summary extracted)")
    print()
    print(f" \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(
        f" Context compacted: {len(before_msgs)} messages \u2192 {len(after_msgs)} messages"
        f" ({before_tokens:,} \u2192 {after_tokens:,} tokens, saved {saved:,})"
    )
    return False


def _extract_compact_summary(messages: list[dict[str, Any]]) -> str | None:
    """从压缩后的消息列表中提取结构化摘要文本。"""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and isinstance(content, str) and content.startswith("[Compressed]"):
            return content
    return None


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
    """列出或清除权限规则。"""
    handle_permissions(
        cmd,
        ctx.session_grant_store,
        ctx.permanent_grant_store,
        static_policy=ctx.static_policy,
        restricted_dirs=ctx.restricted_dirs,
        project_root=ctx.project_root,
        app=ctx.app,
        store=ctx.store,
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


def cmd_mcp(cmd: str, ctx: CommandContext) -> bool:
    """显示 MCP 状态或手动重载配置。"""
    parts = cmd.split(maxsplit=1)
    action = parts[1].strip() if len(parts) == 2 else "status"
    if action == "reload":
        reload_mcp = getattr(ctx.app, "reload_mcp", None)
        if reload_mcp is None:
            print("MCP runtime is not available.")
            return False
        names = reload_mcp()
        print(f"Reloaded MCP config. Registered {len(names)} MCP tools.")
        return False
    if action != "status":
        print("Usage: /mcp status|reload")
        return False
    mcp_status = getattr(ctx.app, "mcp_status", None)
    if mcp_status is None:
        print("MCP runtime is not available.")
        return False
    statuses = mcp_status()
    if not statuses:
        print("No MCP servers configured.")
        return False
    for status in statuses:
        identity = status.get("server_info") or {}
        identity_text = ""
        if isinstance(identity, dict) and identity:
            name = identity.get("name", "?")
            version = identity.get("version", "?")
            identity_text = f" identity={name}@{version}"
        protocol = status.get("protocol_version")
        protocol_text = f" protocol={protocol}" if protocol else ""
        error = status.get("last_error")
        error_text = f" error={error}" if error else ""
        print(
            f"{status['server_name']}: state={status['state']} "
            f"tools={status['tool_count']} deferred={status['deferred']}"
            f"{protocol_text}{identity_text}{error_text}"
        )
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
    parts = cmd.split(maxsplit=2)
    if len(parts) < 2 or not parts[1].strip():
        print("Usage: /skill NAME [prompt]")
        return False
    result = activate_skill(ctx.app, ctx.store, parts[1].strip(), mode=ctx.state.mode)
    print(result.message)
    if result.status in {"activated", "already_active"} and len(parts) == 3:
        prompt = parts[2].strip()
        if prompt:
            ctx.state.pending_inject = prompt
    return False


def cmd_memory(cmd: str, ctx: CommandContext) -> bool:
    """检索、列出或显式添加项目级与用户级记忆。"""
    manager = MemoryManager(ctx.project_root)
    parts = cmd.split(maxsplit=2)
    action = parts[1].lower() if len(parts) >= 2 else "list"
    payload = parts[2].strip() if len(parts) >= 3 else ""

    if action == "list":
        return _list_memory(manager, payload)
    if action == "search":
        return _search_memory(manager, payload)
    if action == "add":
        return _add_memory(manager, payload)

    print("Usage: /memory list [all|project|user]")
    print("       /memory search <query>")
    print(
        "       /memory add [project|user] "
        "<title> | <context> | <solution> | <files> | <takeaways>"
    )
    print(
        "Example: /memory add project Title | Context here | Solution here | "
        "src/file.py | Key takeaway"
    )
    return False


def _list_memory(manager: MemoryManager, raw_layer: str) -> bool:
    """列出指定记忆层级中的标题。"""
    layer = raw_layer.lower() or "all"
    if layer not in {"all", "project", "user"}:
        print("Memory layer must be one of: all, project, user.")
        return False

    records = manager.read_memory_records(layer=cast(MemoryLayerFilter, layer))
    if not records:
        print("No memory records found.")
        return False

    print(f"Memory records ({len(records)}):")
    for record in records:
        print(f"  [{record.layer}] {record.title}")
    return False


def _search_memory(manager: MemoryManager, query: str) -> bool:
    """打印跨层级记忆检索结果。"""
    if not query:
        print("Usage: /memory search <query>")
        return False

    records = manager.search_memory_records(query, limit=5)
    if not records:
        print(f"No memory matching {query!r}.")
        return False

    for record in records:
        print(f"[{record.layer}] score={record.score:.3f}")
        print(record.block.strip())
        print()
    return False


def _add_memory(manager: MemoryManager, payload: str) -> bool:
    """解析单行结构化输入并写入指定记忆层级。"""
    layer = "project"
    value = payload
    first, separator, remainder = payload.partition(" ")
    if separator and first.lower() in {"project", "user"}:
        layer = first.lower()
        value = remainder.strip()

    fields = [field.strip() for field in value.split("|")]
    if len(fields) == 1 and fields[0]:
        title, context = _split_memory_shorthand(fields[0])
        fields = [title, context, context, ".", context]
    if len(fields) != 5 or any(not field for field in fields):
        print(
            "Usage: /memory add [project|user] "
            "<title> | <context> | <solution> | <files> | <takeaways>"
        )
        print("Short form: /memory add [project|user] <title>: <note>")
        return False

    title, context, solution, files, takeaways = fields
    block = (
        f"## {title}\n"
        f"- Context/Query: {context}\n"
        f"- Solution: {solution}\n"
        f"- Files: {files}\n"
        f"- Takeaways: {takeaways}\n"
    )
    memory_layer = cast(MemoryLayer, layer)
    if not manager.add_memory_block(block, source="repl", layer=memory_layer):
        print("Memory was rejected by validation or duplicate detection.")
        return False

    memory_file = (
        manager.memory_file if layer == "project" else manager.user_memory_file
    )
    print(f"Added {layer} memory: {title}")
    print(f"Path: {memory_file}")
    return False


def _split_memory_shorthand(text: str) -> tuple[str, str]:
    """将自然语言记忆简写拆成标题和正文。"""
    for separator in ("：", ":"):
        title, found, body = text.partition(separator)
        if found and title.strip() and body.strip():
            return title.strip(), body.strip()
    words = text.split(maxsplit=1)
    if len(words) == 2:
        return words[0].strip(), words[1].strip()
    return text.strip(), text.strip()


def cmd_exit(cmd: str, ctx: CommandContext) -> bool:
    """退出 REPL。"""
    return True


@dataclass
class _ContextSummary:
    categories: list[tuple[str, int]]
    total: int
    context_window: int
    model_name: str
    spent: float
    free: int
    memory_text: str
    skill_count: int


def _format_token(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _count_output_tokens(messages: list[object]) -> int:
    """Sum output tokens from AssistantMessage.usage across history."""
    total = 0
    for msg in messages:
        usage = getattr(msg, "usage", None) or {}
        if isinstance(usage, dict):
            ct = usage.get("completion_tokens") or usage.get("output_tokens", 0)
            total += ct if isinstance(ct, int) else 0
        elif hasattr(usage, "output"):
            ct = getattr(usage, "output", 0)
            total += ct if isinstance(ct, int) else 0
    return total


def _get_context_window(model_name: str) -> int:
    from xcode.ai.registry import get_providers, get_models

    for provider in get_providers():
        for model in get_models(provider):
            if model.id == model_name:
                return model.context_window
    return 0


def _get_model_cost(model_name: str) -> object | None:
    from xcode.ai.registry import get_providers, get_models

    for provider in get_providers():
        for model in get_models(provider):
            if model.id == model_name:
                return model.cost
    return None


def _compute_context_summary(
    agent: object, project_root: Path, state: ReplState
) -> _ContextSummary:
    """计算分类 token 用量，并更新 state 供底栏使用。"""
    from xcode.agent.compaction import estimate_tokens, estimate_message_tokens
    from xcode.harness.agent_runtime.prompting.identity import (
        CORE_IDENTITY,
        TOOL_DISCIPLINE,
        SEARCH_STRATEGY,
    )

    categories: list[tuple[str, int]] = []

    system_text = f"{CORE_IDENTITY}\n\n{TOOL_DISCIPLINE}\n\n{SEARCH_STRATEGY}"
    categories.append(("System prompt", estimate_tokens(system_text)))

    registry = getattr(agent, "registry", None)
    if registry is not None:
        snap = registry.snapshot() if hasattr(registry, "snapshot") else registry
        from xcode.harness.skills import build_tool_prompt, build_tool_guidelines

        parts = ["Available tools:\n" + build_tool_prompt(snap)]
        guidelines = build_tool_guidelines(snap)
        if guidelines:
            parts.append("Guidelines:\n" + guidelines)
        categories.append(("System tools", estimate_tokens("\n\n".join(parts))))

    history_messages = getattr(agent, "history_messages", None)
    messages = history_messages() if history_messages is not None else []
    categories.append(("Messages", estimate_message_tokens(messages)))

    memory_manager = MemoryManager(project_root)
    memory_text = "\n".join(memory_manager.read_memory_blocks())
    if memory_text:
        categories.append(("Memory files", estimate_tokens(memory_text)))

    skill_count = 0
    runtime = getattr(agent, "_runtime", None)
    skill_registry = getattr(runtime, "skill_registry", None) if runtime else None
    if skill_registry is not None and hasattr(skill_registry, "list_summaries"):
        summaries = skill_registry.list_summaries()
        if summaries:
            skill_count = len(summaries)
            lines = [
                "<skill-activation>\n"
                "When the user task clearly matches a skill description below, "
                "call load_skill with that exact name before performing the task. "
                "Do not load a skill when no description clearly matches.\n"
                "</skill-activation>",
                "<available-skills>",
            ]
            for s in summaries:
                desc = s.description
                if len(desc) > 768:
                    desc = desc[:765] + "..."
                lines.append(f"  <skill name={s.name}>{desc}</skill>")
            lines.append("</available-skills>")
            categories.append(("Skills", estimate_tokens("\n".join(lines))))

    provider = getattr(agent, "provider", None)
    inner = getattr(provider, "active_provider", provider)
    model_name = getattr(inner, "model", "unknown") if inner else "unknown"
    context_window = _get_context_window(model_name)
    cost = _get_model_cost(model_name)

    total = sum(t for _, t in categories)
    free = max(0, context_window - total) if context_window > 0 else 0
    cost_input_rate = getattr(cost, "input", 0) if cost else 0
    cost_output_rate = getattr(cost, "output", 0) if cost else 0

    input_cost = (total / 1_000_000) * cost_input_rate if cost_input_rate else 0
    history = getattr(agent, "history_messages", lambda: [])()
    output_tokens = _count_output_tokens(history)
    output_cost = (
        (output_tokens / 1_000_000) * cost_output_rate if cost_output_rate else 0
    )
    spent = input_cost + output_cost

    context_str = (
        f"{_format_token(total)}/{_format_token(context_window)}"
        f" ({total / context_window * 100:.1f}%)"
        if context_window > 0
        else f"{_format_token(total)} tokens"
    )
    cost_str = f"${spent:.2f}" if spent > 0 else ""

    state.last_dir = str(project_root)
    state.model_name = model_name
    state.context_usage = context_str
    state.context_cost = cost_str

    return _ContextSummary(
        categories=categories,
        total=total,
        context_window=context_window,
        model_name=model_name,
        spent=spent,
        free=free,
        memory_text=memory_text,
        skill_count=skill_count,
    )


def cmd_context(cmd: str, ctx: CommandContext) -> bool:
    """显示当前会话上下文使用情况，按分类展示 token 用量。"""
    agent = getattr(ctx.app, "agent", None)
    if agent is None:
        print("No agent available.")
        return False

    summary = _compute_context_summary(agent, ctx.project_root, ctx.state)

    cost_str = f"${summary.spent:.2f}" if summary.spent > 0 else ""
    parts = [f"cwd: {ctx.project_root}"]
    parts.append(f"model: {summary.model_name}")
    parts.append(f"mode: {ctx.state.mode}")
    parts.append(f"context: {ctx.state.context_usage}")
    if cost_str:
        parts.append(f"cost: {cost_str}")
    print("  ".join(parts))

    print(" Estimated usage by category")
    for name, tokens in summary.categories:
        pct = (
            (tokens / summary.context_window * 100) if summary.context_window > 0 else 0
        )
        print(f"   \u26c1 {name}: {_format_token(tokens)} tokens ({pct:.1f}%)")

    if summary.context_window > 0:
        free_pct = summary.free / summary.context_window * 100
        print(f"   \u26f6 Free space: {_format_token(summary.free)} ({free_pct:.1f}%)")

    if summary.memory_text:
        block_count = max(1, summary.memory_text.count("## "))
        print("\n Memory files \u00b7 /memory")
        print(
            f" \u2514 {block_count} blocks"
            f" \u00b7 {_format_token(len(summary.memory_text))} chars"
        )

    if summary.skill_count > 0:
        skill_token = next((t for n, t in summary.categories if n == "Skills"), 0)
        print("\n Skills \u00b7 /skills")
        print(
            f" \u2514 {summary.skill_count} skills"
            f" \u00b7 {_format_token(skill_token)} tokens"
        )

    return False


def cmd_btw(cmd: str, ctx: CommandContext) -> bool:
    """Ask a quick side question without interrupting the main conversation."""
    parts = cmd.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        print("Usage: /btw <question>")
        return False

    question = parts[1].strip()

    from xcode.harness.agent_runtime.events import TextDeltaStructuredEvent

    sys.stdout.write("\033[90m[side question]\033[0m\n")
    sys.stdout.flush()

    for event in ctx.app.ask_stream(question, mode=ctx.state.mode):
        if isinstance(event, TextDeltaStructuredEvent):
            sys.stdout.write(event.data)
            sys.stdout.flush()

    print()

    sync_agent_history(ctx.app, ctx.store)

    return False


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
        group=COMMAND_GROUP_SESSION_LIFECYCLE,
    ),
    "/continue": CommandEntry(
        handler=cmd_continue,
        desc="Resume the latest session for this project.",
        group=COMMAND_GROUP_SESSION_LIFECYCLE,
    ),
    "/new": CommandEntry(
        handler=cmd_clear,
        desc="Start a new session transcript.",
        group=COMMAND_GROUP_SESSION_LIFECYCLE,
    ),
    "/fork": CommandEntry(
        handler=cmd_fork,
        desc="Fork current session into an independent branch.",
        args_desc="[explore|verify|isolate]",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION_BRANCH,
    ),
    "/rewind": CommandEntry(
        handler=cmd_rewind,
        desc="Remove the last N user turns from the transcript.",
        args_desc="N",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION_ROLLBACK,
    ),
    "/resume": CommandEntry(
        handler=cmd_resume,
        desc="Choose a recent conversation to resume.",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION_LIFECYCLE,
    ),
    "/sessions": CommandEntry(
        handler=cmd_sessions,
        desc="List and resume recent conversations.",
        group=COMMAND_GROUP_SESSION_LIFECYCLE,
    ),
    "/tree": CommandEntry(
        handler=cmd_tree,
        desc="Show session fork tree.",
        group=COMMAND_GROUP_SESSION_BRANCH,
    ),
    "/branch": CommandEntry(
        handler=cmd_branch,
        desc="List or switch session branches.",
        args_desc="list|tree|<id|title>",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION_BRANCH,
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
        group=COMMAND_GROUP_MODEL,
    ),
    "/thinking": CommandEntry(
        handler=cmd_thinking,
        desc="Show current thinking state (on/off).",
        args_desc="on|off",
        accepts_args=True,
        group=COMMAND_GROUP_MODEL,
    ),
    "/config": CommandEntry(
        handler=cmd_config,
        desc="Manage provider profiles interactively.",
        args_desc="[list|add <name>|edit <name>|delete <name>|set <profile> <field> <value>|reload]",
        accepts_args=True,
        group=COMMAND_GROUP_MODEL,
    ),
    "/plan": CommandEntry(
        handler=cmd_plan,
        desc="Enter Plan Mode: read-only inspection tools, no edits or shell.",
        args_desc="[prompt]",
        accepts_args=True,
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
        group=COMMAND_GROUP_SESSION_ROLLBACK,
    ),
    "/permissions": CommandEntry(
        handler=cmd_permissions,
        desc="List or clear active permission rules and grants.",
        accepts_args=True,
        group=COMMAND_GROUP_INFO,
    ),
    "/hooks": CommandEntry(
        handler=cmd_hooks,
        desc="Show external hook sources and recent status.",
        group=COMMAND_GROUP_INFO,
    ),
    "/mcp": CommandEntry(
        handler=cmd_mcp,
        desc="Show MCP server status or reload .local/mcp_config.json.",
        args_desc="status|reload",
        accepts_args=True,
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
    "/memory": CommandEntry(
        handler=cmd_memory,
        desc="List, search, or add project and user memory.",
        args_desc="list [all|project|user] | search <query> | add ...",
        accepts_args=True,
        group=COMMAND_GROUP_INFO,
    ),
    "/rename": CommandEntry(
        handler=cmd_rename,
        desc="Rename the current session.",
        args_desc="<title>",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION_LIFECYCLE,
    ),
    "/undo": CommandEntry(
        handler=cmd_undo,
        desc="Undo file changes from the last N user turns (via snapshot restore).",
        args_desc="[N|--list]",
        accepts_args=True,
        group=COMMAND_GROUP_SESSION_ROLLBACK,
    ),
    "/exit": CommandEntry(
        handler=cmd_exit, desc="Exit the REPL.", group=COMMAND_GROUP_EXIT
    ),
    "/context": CommandEntry(
        handler=cmd_context,
        desc="Show context usage (token count, messages, etc.).",
        group=COMMAND_GROUP_INFO,
    ),
    "/btw": CommandEntry(
        handler=cmd_btw,
        desc="Ask a quick side question without interrupting the main conversation.",
        args_desc="<question>",
        accepts_args=True,
        group=COMMAND_GROUP_INFO,
    ),
    "/quit": CommandEntry(
        handler=cmd_exit,
        desc="Alias for /exit.",
        visible=False,
        group=COMMAND_GROUP_EXIT,
        canonical="/exit",
    ),
    "/revert": CommandEntry(
        handler=cmd_undo,
        desc="Alias for /undo.",
        args_desc="[N|--list]",
        accepts_args=True,
        visible=False,
        group=COMMAND_GROUP_SESSION_ROLLBACK,
        canonical="/undo",
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
            if entry.canonical is not None:
                canonical_entry = COMMAND_REGISTRY[entry.canonical]
                command = entry.canonical + command[len(prefix) :]
                print(command)
                return canonical_entry.handler(command, ctx)
            return entry.handler(command, ctx)
    print(f"Unknown command: {command}")
    return False
