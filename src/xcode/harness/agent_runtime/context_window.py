"""可丢弃上下文窗口与活动工作集管理。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from xcode.agent._context_window import estimate_tokens
from xcode.agent.config import ContextWindowResetReason
from xcode.agent.types import ToolInput, ToolSpec

from ..skill_activation import activated_skill_names, is_skill_activation_content

_RESET_TAG = "<context-window-reset"


class ContextWindowController:
    """在工具调用或 UI 与 agent 循环之间传递一次换窗请求。"""

    def __init__(self) -> None:
        self._reason: ContextWindowResetReason | None = None

    def request(self, reason: ContextWindowResetReason) -> bool:
        self._reason = reason
        return True

    def consume(self) -> ContextWindowResetReason | None:
        reason = self._reason
        self._reason = None
        return reason


class ContextWindowRollover:
    """关闭旧窗口，只把启动上下文、技能与当前工作回合带入新窗口。"""

    def __init__(
        self,
        keep_recent_tool_results: int = 2,
        max_tool_result_chars: int = 100,
        fallback_recent_messages: int = 8,
        fallback_recent_tokens: int = 20_000,
        large_tool_output_chars: int = 20_000,
        large_tool_output_head_chars: int = 10_000,
        large_tool_output_tail_chars: int = 10_000,
        active_window_token_threshold: int = 32_000,
        tool_trim_trigger_ratio: float = 0.5,
    ) -> None:
        self.keep_recent_tool_results = keep_recent_tool_results
        self.max_tool_result_chars = max_tool_result_chars
        self.fallback_recent_messages = fallback_recent_messages
        self.fallback_recent_tokens = fallback_recent_tokens
        self.large_tool_output_chars = large_tool_output_chars
        self.large_tool_output_head_chars = large_tool_output_head_chars
        self.large_tool_output_tail_chars = large_tool_output_tail_chars
        self.active_window_token_threshold = active_window_token_threshold
        self.tool_trim_trigger_ratio = tool_trim_trigger_ratio
        self.last_window_id: str | None = None

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_active_turn: bool = True,
    ) -> list[dict[str, Any]]:
        """执行无摘要换窗；原始历史由 session JSONL 保持不变。"""
        archived = deepcopy(messages)
        window_id = uuid4().hex[:12]
        self.last_window_id = window_id

        leading_system_end = _leading_system_end(archived)
        recent_start = (
            _active_turn_start(
                archived,
                leading_system_end=leading_system_end,
                fallback_recent_messages=self.fallback_recent_messages,
                fallback_recent_tokens=self.fallback_recent_tokens,
            )
            if preserve_active_turn
            else len(archived)
        )
        initial_context = [
            deepcopy(message)
            for message in archived[:leading_system_end]
            if not _is_reset_notice(message)
        ]
        protected_skills = _activated_skill_context_messages(
            archived[leading_system_end:recent_start]
        )
        active_window = [
            *initial_context,
            {"role": "system", "content": render_context_window_reset(window_id)},
            *protected_skills,
            *deepcopy(archived[recent_start:]),
        ]
        active_window = stale_snip_file_reads(active_window)
        preserved_results = latest_read_file_tool_result_ids(active_window)
        preserved_results.update(activated_skill_tool_result_ids(active_window))
        active_window = budget_large_tool_outputs(
            active_window,
            large_tool_output_chars=self.large_tool_output_chars,
            large_tool_output_head_chars=self.large_tool_output_head_chars,
            large_tool_output_tail_chars=self.large_tool_output_tail_chars,
            active_window_token_threshold=self.active_window_token_threshold,
            tool_trim_trigger_ratio=self.tool_trim_trigger_ratio,
            preserve_tool_result_ids=preserved_results,
        )
        return trim_old_tool_results(
            active_window,
            keep_recent=self.keep_recent_tool_results,
            max_content_chars=self.max_tool_result_chars,
            preserve_tool_result_ids=preserved_results,
        )


def build_new_context_tool(
    controller: ContextWindowController,
    project_root: Path,
) -> tuple[ToolSpec, ...]:
    """构建模型主动换窗工具；NOTE.md 是强制交接点。"""

    def request_new_context(
        data: ToolInput,
        _on_update: Callable[[str], None] | None = None,
    ) -> str:
        reason = str(data.get("reason", "")).strip()
        note_path = project_root / "NOTE.md"
        try:
            note = note_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            note = ""
        if not note:
            return (
                "Context window not changed. Write NOTE.md with the current goal, "
                "confirmed decisions, completed verification, unresolved issues, "
                "and the exact next action; then call new_context again."
            )
        controller.request("model")
        return (
            "Fresh context window scheduled before the next inference. No summary "
            f"will be generated. Reason: {reason}"
        )

    return (
        ToolSpec(
            name="new_context",
            description=(
                "Close the current model context and continue in a fresh window "
                "without generating a summary. First update NOTE.md with the "
                "execution frontier; older exact details remain in history."
            ),
            input_hint='JSON: {"reason":"the active window is stale or noisy"}',
            handler=request_new_context,
            schema={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why a fresh working window is useful now.",
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
            prompt_snippet=(
                "Use new_context when the current working set is stale or near its "
                "token budget. Update NOTE.md first; no summary is generated."
            ),
        ),
    )


def render_context_window_reset(window_id: str) -> str:
    """渲染新窗口的最小恢复协议。"""
    return (
        f'<context-window-reset id="{window_id}">\n'
        "The previous context window was closed without a summary. NOTE.md "
        "contains explicit working state; the lossless session transcript is "
        "authoritative. Use history list_windows/search/read/around to retrieve "
        "older details before relying on memory.\n"
        "</context-window-reset>"
    )


def stale_snip_file_reads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """在活动窗口内裁剪同一文件的旧读取结果。"""
    window = deepcopy(messages)
    tool_use_id_to_path = _read_file_tool_paths(window)
    path_to_results: dict[str, list[dict[str, Any]]] = {}
    for part in _iter_tool_results(window):
        path = tool_use_id_to_path.get(str(part.get("tool_use_id", "")))
        if path:
            path_to_results.setdefault(path, []).append(part)
    for results in path_to_results.values():
        for part in results[:-1]:
            part["content"] = "[Content snipped - re-read if needed]"
    return window


def latest_read_file_tool_result_ids(messages: list[dict[str, Any]]) -> set[str]:
    """返回每个文件最新一次 read_file 对应的工具结果 ID。"""
    tool_use_id_to_path = _read_file_tool_paths(messages)
    path_to_latest: dict[str, str] = {}
    for part in _iter_tool_results(messages):
        tool_use_id = str(part.get("tool_use_id", ""))
        path = tool_use_id_to_path.get(tool_use_id)
        if path:
            path_to_latest[path] = tool_use_id
    return set(path_to_latest.values())


def activated_skill_tool_result_ids(messages: list[dict[str, Any]]) -> set[str]:
    """返回包含完整技能激活内容的工具结果 ID。"""
    return {
        str(part.get("tool_use_id", ""))
        for part in _iter_tool_results(messages)
        if is_skill_activation_content(part.get("content", ""))
        and str(part.get("tool_use_id", ""))
    }


def budget_large_tool_outputs(
    messages: list[dict[str, Any]],
    large_tool_output_chars: int = 20_000,
    large_tool_output_head_chars: int = 10_000,
    large_tool_output_tail_chars: int = 10_000,
    active_window_token_threshold: int = 32_000,
    tool_trim_trigger_ratio: float = 0.5,
    preserve_tool_result_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """活动窗口达到预算压力时，对超大工具结果保留头尾。"""
    window = deepcopy(messages)
    protected = set(preserve_tool_result_ids or ())
    protected.update(activated_skill_tool_result_ids(window))
    trigger = active_window_token_threshold * tool_trim_trigger_ratio
    if estimate_message_tokens(window) <= trigger:
        return window
    for part in _iter_tool_results(window):
        if str(part.get("tool_use_id", "")) in protected:
            continue
        content = part.get("content", "")
        if not isinstance(content, str) or len(content) <= large_tool_output_chars:
            continue
        if content.startswith("[") and content.endswith("]"):
            continue
        if len(content) <= large_tool_output_head_chars + large_tool_output_tail_chars:
            continue
        removed = (
            len(content) - large_tool_output_head_chars - large_tool_output_tail_chars
        )
        part["content"] = (
            f"{content[:large_tool_output_head_chars]}\n\n"
            f"[... {removed} characters omitted from active window; retrieve "
            f"exact output from history ...]\n\n"
            f"{content[-large_tool_output_tail_chars:]}"
        )
    return window


def trim_old_tool_results(
    messages: list[dict[str, Any]],
    keep_recent: int = 2,
    max_content_chars: int = 100,
    preserve_tool_result_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """裁剪活动窗口中的旧工具结果，不修改持久化 transcript。"""
    window = deepcopy(messages)
    results = list(_iter_tool_results(window))
    cutoff = max(len(results) - max(keep_recent, 0), 0)
    for part in results[:cutoff]:
        tool_use_id = str(part.get("tool_use_id", ""))
        if preserve_tool_result_ids and tool_use_id in preserve_tool_result_ids:
            continue
        content = str(part.get("content", ""))
        if len(content) > max_content_chars:
            part["content"] = (
                f"[Tool result omitted from active window; {len(content)} chars "
                "remain available through history]"
            )
    return window


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(
        estimate_tokens(json.dumps(message, ensure_ascii=False, default=str))
        for message in messages
    )


def _active_turn_start(
    messages: list[dict[str, Any]],
    *,
    leading_system_end: int,
    fallback_recent_messages: int,
    fallback_recent_tokens: int,
) -> int:
    for index in range(len(messages) - 1, leading_system_end - 1, -1):
        if messages[index].get("role") == "user":
            return index
    recent_count = min(
        fallback_recent_messages,
        _compute_recent_count_from_tokens(messages, fallback_recent_tokens),
    )
    raw_start = max(leading_system_end, len(messages) - recent_count)
    return _find_turn_boundary(messages, raw_start)


def _leading_system_end(messages: list[dict[str, Any]]) -> int:
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        index += 1
    return index


def _is_reset_notice(message: dict[str, Any]) -> bool:
    return message.get("role") == "system" and _RESET_TAG in str(
        message.get("content", "")
    )


def _compute_recent_count_from_tokens(
    messages: list[dict[str, Any]], token_budget: int
) -> int:
    accumulated = 0
    count = 0
    for message in reversed(messages):
        tokens = estimate_message_tokens([message])
        if accumulated + tokens > token_budget and count > 0:
            break
        accumulated += tokens
        count += 1
    return max(count, 1)


def _find_turn_boundary(messages: list[dict[str, Any]], raw_index: int) -> int:
    if not messages:
        return 0
    index = min(max(raw_index, 0), len(messages) - 1)
    while index > 0:
        if messages[index].get("role") in {"user", "assistant"}:
            return index
        index -= 1
    return index


def _read_file_tool_paths(messages: list[dict[str, Any]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_use_id, tool_name, tool_input in _iter_tool_uses(message):
            if not tool_use_id or tool_name != "read_file":
                continue
            path = str(tool_input.get("path", "")).strip()
            if path:
                paths[tool_use_id] = Path(path).as_posix()
    return paths


def _iter_tool_uses(
    message: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "tool_use"):
                continue
            tool_input = part.get("input", {})
            if isinstance(tool_input, dict):
                yield (
                    str(part.get("id", "")).strip(),
                    str(part.get("name", "")).strip(),
                    tool_input,
                )
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if isinstance(arguments, dict):
            yield (
                str(call.get("id", "")).strip(),
                str(function.get("name", "")).strip(),
                arguments,
            )


def _iter_tool_results(messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                yield part


def _activated_skill_context_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把已激活技能重注入为显式启动上下文，避免伪造旧工具调用链。"""
    by_name: dict[str, str] = {}
    for message in messages:
        for text in _message_text_fragments(message):
            for name in activated_skill_names(text):
                by_name[name] = text
    return [{"role": "user", "content": by_name[name]} for name in sorted(by_name)]


def _message_text_fragments(message: dict[str, Any]) -> Iterator[str]:
    content = message.get("content")
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            value = part.get("content") if part.get("type") == "tool_result" else None
            if isinstance(value, str):
                yield value
