"""会话级上下文管理器。

把历史、world state、预算、压缩生命周期和 provider 统计放在同一个可追踪对象中。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._compaction import estimate_message_tokens
from ._hygiene import repair_tool_pairing
from .context import ContextState
from .messages import AgentMessage

if TYPE_CHECKING:
    from .request import RequestAssembly


@dataclass
class ContextTokenUsage:
    """当前 context window 的估算和 provider 实测 token。"""

    estimated_prompt_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_budget: int = 0
    budget_remaining: int = 0
    last_prompt_tokens: int | None = None


@dataclass
class ContextCompactionState:
    """压缩与 context window 生命周期状态。"""

    context_window_id: int = 0
    compaction_count: int = 0
    last_reason: str | None = None
    last_messages_before: int = 0
    last_messages_after: int = 0


@dataclass
class PromptCacheMetadata:
    """最近一次请求的 prompt/cache fingerprint。"""

    prompt_sha256: str = ""
    request_sha256: str = ""
    system_prompt_bytes: int = 0
    request_count: int = 0


@dataclass
class ContextManager:
    """会话上下文的唯一可变状态边界。"""

    history: list[AgentMessage] = field(default_factory=list)
    context_state: ContextState = field(default_factory=ContextState)
    history_version: int = 0
    token_usage: ContextTokenUsage = field(default_factory=ContextTokenUsage)
    compaction: ContextCompactionState = field(default_factory=ContextCompactionState)
    prompt_cache: PromptCacheMetadata = field(default_factory=PromptCacheMetadata)
    provider_usage: dict[str, int] = field(default_factory=dict)

    def history_messages(self) -> list[AgentMessage]:
        return list(self.history)

    def replace_history(
        self,
        messages: Sequence[AgentMessage],
        *,
        normalize: bool = True,
    ) -> list[AgentMessage]:
        """替换历史并在进入 session surface 前修复工具配对。"""
        replacement = list(messages)
        if normalize:
            replacement = repair_tool_pairing(replacement)
        self.history[:] = replacement
        self.history_version += 1
        self.token_usage.estimated_prompt_tokens = estimate_message_tokens(self.history)
        return self.history_messages()

    def normalize_messages(
        self,
        messages: Sequence[AgentMessage],
    ) -> list[AgentMessage]:
        """在 provider 请求前返回已修复工具配对的消息投影。"""
        return repair_tool_pairing(list(messages))

    def append(self, messages: Sequence[AgentMessage]) -> None:
        if not messages:
            return
        self.history.extend(messages)
        self.history[:] = repair_tool_pairing(self.history)
        self.history_version += 1
        self.token_usage.estimated_prompt_tokens = estimate_message_tokens(self.history)

    def complete_compaction(
        self,
        messages: Sequence[AgentMessage],
        *,
        reason: str = "token_limit",
        before_messages: int | None = None,
    ) -> list[AgentMessage]:
        """提交压缩结果、开启新的 context window 并重置 world state。"""
        before = len(self.history) if before_messages is None else before_messages
        replacement = self.replace_history(messages)
        self.context_state.reset()
        self.token_usage.last_prompt_tokens = None
        self.compaction.context_window_id += 1
        self.compaction.compaction_count += 1
        self.compaction.last_reason = reason
        self.compaction.last_messages_before = before
        self.compaction.last_messages_after = len(replacement)
        return replacement

    def clear(self) -> None:
        self.history.clear()
        self.context_state.reset()
        self.history_version += 1
        self.token_usage = ContextTokenUsage()
        self.compaction = ContextCompactionState(
            context_window_id=self.compaction.context_window_id + 1
        )
        self.provider_usage.clear()
        self.prompt_cache = PromptCacheMetadata()

    def set_last_prompt_tokens(self, value: int | None) -> None:
        self.token_usage.last_prompt_tokens = value

    def record_request(self, assembly: RequestAssembly) -> None:
        """记录组装后的请求预算和 prompt/cache fingerprint。"""
        self.token_usage.estimated_prompt_tokens = assembly.estimated_tokens
        self.token_usage.context_budget = assembly.token_budget
        self.token_usage.budget_remaining = assembly.budget_remaining
        wire_messages = list(assembly.wire_messages)
        system_prompt = "\n\n".join(
            str(message.get("content", ""))
            for message in wire_messages
            if message.get("role") == "system"
        )
        prompt_bytes = system_prompt.encode("utf-8")
        request_payload = {
            "messages": wire_messages,
            "tools": [_json_value(tool) for tool in assembly.tools],
        }
        request_bytes = json.dumps(
            request_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        self.prompt_cache = PromptCacheMetadata(
            prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
            request_sha256=hashlib.sha256(request_bytes).hexdigest(),
            system_prompt_bytes=len(prompt_bytes),
            request_count=self.prompt_cache.request_count + 1,
        )

    def record_provider_usage(self, usage: Mapping[str, object] | None) -> None:
        """累加 provider 返回的输入、输出和缓存统计。"""
        if not usage:
            return
        numeric: dict[str, int] = {}
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                numeric[str(key)] = value
        for key, value in numeric.items():
            self.provider_usage[key] = self.provider_usage.get(key, 0) + value

        prompt_tokens = numeric.get("prompt_tokens")
        completion_tokens = numeric.get("completion_tokens")
        if prompt_tokens is not None:
            self.token_usage.prompt_tokens += prompt_tokens
            self.token_usage.last_prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            self.token_usage.completion_tokens += completion_tokens
        total_tokens = numeric.get("total_tokens")
        if total_tokens is not None:
            self.token_usage.total_tokens += total_tokens
        else:
            self.token_usage.total_tokens += prompt_tokens or 0
            self.token_usage.total_tokens += completion_tokens or 0


def _json_value(value: object) -> Any:
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value
