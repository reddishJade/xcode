from __future__ import annotations


from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
from collections.abc import Awaitable, Callable
from typing import Any

from xcode.agent.messages import BRANCH_SUMMARY_PREFIX, SUMMARY_SUFFIX
from xcode.agent.config import CompactInstructions
from ..skills import ToolSpec

"""分层上下文压缩工具。"""


class CompactController:
    def __init__(self) -> None:
        self._requested = False

    def request(self) -> str:
        self._requested = True
        return "manual compact requested"

    def consume(self) -> bool:
        requested = self._requested
        self._requested = False
        return requested


class LayeredCompactor:
    def __init__(
        self,
        transcript_dir: Path | None = None,
        keep_recent_tool_results: int = 2,
        max_tool_result_chars: int = 100,
        max_recent_messages: int = 8,
        large_tool_output_chars: int = 20_000,
        large_tool_output_head_chars: int = 10_000,
        large_tool_output_tail_chars: int = 10_000,
        compact_token_threshold: int = 32_000,
        budget_trigger_token_ratio: float = 0.5,
        on_compact: Callable[[str], None] | None = None,
        compact_instructions: CompactInstructions | None = None,
        summarize_fn: SummarizeFn | None = None,
        active_branch_id: str | None = None,
    ) -> None:
        self.transcript_dir = transcript_dir
        self.compact_instructions = compact_instructions
        self.summarize_fn = summarize_fn
        self.active_branch_id = active_branch_id
        self.keep_recent_tool_results = keep_recent_tool_results
        self.max_tool_result_chars = max_tool_result_chars
        self.max_recent_messages = max_recent_messages
        self.large_tool_output_chars = large_tool_output_chars
        self.large_tool_output_head_chars = large_tool_output_head_chars
        self.large_tool_output_tail_chars = large_tool_output_tail_chars
        self.compact_token_threshold = compact_token_threshold
        self.budget_trigger_token_ratio = budget_trigger_token_ratio
        self.on_compact = on_compact
        self.last_transcript_path: Path | None = None

    def __call__(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compacted = stale_snip_file_reads(messages)
        preserved_tool_results = latest_read_file_tool_result_ids(compacted)
        compacted = budget_large_tool_outputs(
            compacted,
            large_tool_output_chars=self.large_tool_output_chars,
            large_tool_output_head_chars=self.large_tool_output_head_chars,
            large_tool_output_tail_chars=self.large_tool_output_tail_chars,
            compact_token_threshold=self.compact_token_threshold,
            budget_trigger_token_ratio=self.budget_trigger_token_ratio,
            preserve_tool_result_ids=preserved_tool_results,
        )
        micro = micro_compact_tool_results(
            compacted,
            keep_recent=self.keep_recent_tool_results,
            max_content_chars=self.max_tool_result_chars,
            preserve_tool_result_ids=preserved_tool_results,
        )
        branch_compacted = summarize_inactive_branches(
            micro,
            active_branch_id=self.active_branch_id,
            compact_token_threshold=self.compact_token_threshold,
            budget_trigger_token_ratio=self.budget_trigger_token_ratio,
            summarize_fn=self.summarize_fn,
        )
        if self.transcript_dir is not None:
            self.last_transcript_path = save_transcript(
                branch_compacted, self.transcript_dir
            )

        final_messages = summarize_messages(
            branch_compacted,
            max_recent_messages=self.max_recent_messages,
            compact_instructions=self.compact_instructions,
            summarize_fn=self.summarize_fn,
        )
        if len(final_messages) > 1:
            second_msg = final_messages[1]
            if second_msg.get("role") == "user" and str(
                second_msg.get("content")
            ).startswith("[Compressed]"):
                content_val = second_msg.get("content")
                assert isinstance(content_val, str)
                cleaned_content = context_collapse_clean(content_val)
                second_msg["content"] = cleaned_content
                if self.on_compact is not None:
                    self.on_compact(cleaned_content)

        return final_messages


def context_collapse_clean(content: str) -> str:
    """提取 <summary> 标签中的纯净摘要，剥离思考区与分析块。"""
    content_str = str(content).strip()

    # 提取 <summary> 标签中的内容
    summary_match = re.search(
        r"<summary>(.*?)</summary>", content_str, re.DOTALL | re.IGNORECASE
    )
    if summary_match:
        prefix = "[Compressed]\n" if content_str.startswith("[Compressed]") else ""
        return prefix + summary_match.group(1).strip()

    # 移除 <analysis> 和 <think> 标签及其内容
    cleaned = re.sub(
        r"<analysis>.*?</analysis>", "", content_str, flags=re.DOTALL | re.IGNORECASE
    )
    cleaned = re.sub(
        r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    return cleaned.strip()


def stale_snip_file_reads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """裁剪旧的 read_file 工具结果，仅保留最新一次读取。

    两阶段算法设计：
    第一阶段：建立 tool_use_id → file_path 映射
    - 遍历 assistant 消息中的 tool_use，记录 read_file 调用的文件路径

    第二阶段：按文件路径分组，裁剪旧读取结果
    - 收集每个文件的所有 tool_result
    - 保留最新一次结果，旧结果内容替换为 "[Content snipped - re-read if needed]"

    设计原因：
    避免上下文被重复的文件内容污染，特别是大文件多次读取时。
    """
    compacted = deepcopy(messages)
    tool_use_id_to_path: dict[str, str] = {}

    # 第一阶段：建立 tool_use_id → file_path 映射
    for message in compacted:
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "tool_use":
                        tool_use_id = part.get("id")
                        tool_name = part.get("name")
                        tool_input = part.get("input", {})
                        if tool_use_id and tool_name == "read_file":
                            path_val = ""
                            if isinstance(tool_input, dict):
                                path_val = str(tool_input.get("path", "")).strip()
                            if path_val:
                                norm_path = Path(path_val).as_posix()
                                tool_use_id_to_path[tool_use_id] = norm_path

    # 第二阶段：按文件路径分组，收集所有 tool_result
    path_to_results: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
    for message_index, message in enumerate(compacted):
        content = message.get("content")
        if isinstance(content, list):
            for part_index, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_use_id = part.get("tool_use_id")
                    if tool_use_id and tool_use_id in tool_use_id_to_path:
                        path = tool_use_id_to_path[tool_use_id]
                        if path not in path_to_results:
                            path_to_results[path] = []
                        path_to_results[path].append((message_index, part_index, part))

    # 裁剪：保留每个文件的最新读取结果
    for path, results in path_to_results.items():
        if len(results) > 1:
            for _msg_idx, _part_idx, part in results[:-1]:
                part["content"] = "[Content snipped - re-read if needed]"
    return compacted


def latest_read_file_tool_result_ids(messages: list[dict[str, Any]]) -> set[str]:
    """返回每个文件最新一次 read_file 对应的 tool_result id。"""
    tool_use_id_to_path = _read_file_tool_paths(messages)
    path_to_latest: dict[str, str] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "tool_result"):
                continue
            tool_use_id = str(part.get("tool_use_id", ""))
            path = tool_use_id_to_path.get(tool_use_id)
            if path:
                path_to_latest[path] = tool_use_id
    return set(path_to_latest.values())


def _read_file_tool_paths(messages: list[dict[str, Any]]) -> dict[str, str]:
    """提取所有 read_file 工具调用的 tool_use_id → 文件路径映射。"""
    tool_use_id_to_path: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "tool_use"):
                continue
            tool_use_id = part.get("id")
            tool_name = part.get("name")
            tool_input = part.get("input", {})
            if not tool_use_id or tool_name != "read_file":
                continue
            path_val = ""
            if isinstance(tool_input, dict):
                path_val = str(tool_input.get("path", "")).strip()
            if path_val:
                tool_use_id_to_path[str(tool_use_id)] = Path(path_val).as_posix()
    return tool_use_id_to_path


def budget_large_tool_outputs(
    messages: list[dict[str, Any]],
    large_tool_output_chars: int = 20_000,
    large_tool_output_head_chars: int = 10_000,
    large_tool_output_tail_chars: int = 10_000,
    compact_token_threshold: int = 32_000,
    budget_trigger_token_ratio: float = 0.5,
    preserve_tool_result_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """在 token 压力大时，对超大工具结果进行头尾保留裁剪。"""
    compacted = deepcopy(messages)
    current_tokens = estimate_message_tokens(compacted)
    trigger_threshold = compact_token_threshold * budget_trigger_token_ratio

    if current_tokens <= trigger_threshold:
        return compacted

    for message in compacted:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_use_id = str(part.get("tool_use_id", ""))
                    if (
                        preserve_tool_result_ids
                        and tool_use_id in preserve_tool_result_ids
                    ):
                        continue
                    tool_content = part.get("content", "")
                    if (
                        isinstance(tool_content, str)
                        and len(tool_content) > large_tool_output_chars
                    ):
                        if tool_content.startswith("[") and tool_content.endswith("]"):
                            continue
                        if (
                            len(tool_content)
                            > large_tool_output_head_chars
                            + large_tool_output_tail_chars
                        ):
                            head = tool_content[:large_tool_output_head_chars]
                            tail = tool_content[-large_tool_output_tail_chars:]
                            truncated_len = (
                                len(tool_content)
                                - large_tool_output_head_chars
                                - large_tool_output_tail_chars
                            )
                            part["content"] = (
                                f"{head}\n\n[... truncated {truncated_len} characters due to token budget ...]\n\n{tail}"
                            )
    return compacted


def micro_compact_tool_results(
    messages: list[dict[str, Any]],
    keep_recent: int = 2,
    max_content_chars: int = 100,
    preserve_tool_result_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    compacted = deepcopy(messages)
    locations: list[tuple[int, int, dict[str, Any]]] = []
    for message_index, message in enumerate(compacted):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "tool_result":
                locations.append((message_index, part_index, part))
    for _message_index, _part_index, part in locations[:-keep_recent]:
        tool_use_id = str(part.get("tool_use_id", ""))
        if preserve_tool_result_ids and tool_use_id in preserve_tool_result_ids:
            continue
        content = str(part.get("content", ""))
        if len(content) > max_content_chars:
            part["content"] = (
                f"[Previous tool_result compacted; {len(content)} chars removed]"
            )
    return compacted


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(
        estimate_text_tokens(json.dumps(message, ensure_ascii=False, default=str))
        for message in messages
    )


def estimate_text_tokens(text: str) -> int:
    try:
        import tiktoken
    except ImportError:
        return max(1, len(text) // 4)
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


type SummarizeFn = Callable[[list[dict[str, Any]]], str | Awaitable[str]]
"""可选的 LLM 摘要函数。接收旧消息列表，返回摘要文本。"""


SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the following conversation turns into a concise summary. "
    "Preserve architecture decisions, file changes, and TODO items. "
    "Output only the summary, no preamble."
)


def build_summarize_prompt(older: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建 LLM 摘要的消息列表。与 claude.md 描述的 fork → Summarize 模式对应。"""
    return [
        {"role": "user", "content": SUMMARIZE_SYSTEM_PROMPT},
        *older,
    ]


def summarize_messages(
    messages: list[dict[str, Any]],
    max_recent_messages: int = 8,
    compact_instructions: CompactInstructions | None = None,
    summarize_fn: SummarizeFn | None = None,
) -> list[dict[str, Any]]:
    if len(messages) <= max_recent_messages + 1:
        return messages

    if compact_instructions and compact_instructions.priorities:
        preserved_count = _preserved_recent_count(
            messages, compact_instructions, max_recent_messages
        )
        effective_recent = max(max_recent_messages, preserved_count)
    else:
        effective_recent = max_recent_messages

    recent_start = max(1, len(messages) - effective_recent)
    older = messages[1:recent_start]

    if not older:
        return messages

    if summarize_fn:
        raw = summarize_fn(older)
        if isinstance(raw, Awaitable):
            import asyncio
            raw = asyncio.get_event_loop().run_until_complete(raw)
        summary_content = str(raw).strip()
        if not summary_content.startswith("[Compressed]"):
            summary_content = "[Compressed]\n" + summary_content
    else:
        summary_lines = []
        for message in older:
            content = _content_preview(message.get("content"))
            summary_lines.append(f"- {message.get('role')}: {content}")
        summary_content = "[Compressed]\n" + "\n".join(summary_lines)

    if compact_instructions and compact_instructions.frozen_identifiers:
        summary_content = _protect_identifiers(
            summary_content, compact_instructions.frozen_identifiers
        )

    summary = {
        "role": "user",
        "content": summary_content,
    }
    return [messages[0], summary, *messages[recent_start:]]


def summarize_inactive_branches(
    messages: list[dict[str, Any]],
    active_branch_id: str | None = None,
    compact_token_threshold: int = 32_000,
    budget_trigger_token_ratio: float = 0.5,
    summarize_fn: SummarizeFn | None = None,
) -> list[dict[str, Any]]:
    """在上下文紧张时用分支摘要替换非活跃分支消息。"""
    if not _branch_summary_should_run(
        messages, compact_token_threshold, budget_trigger_token_ratio
    ):
        return messages

    result: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        branch_id = _inactive_branch_id(message, active_branch_id)
        if branch_id is None:
            result.append(message)
            index += 1
            continue

        branch_messages = [message]
        index += 1
        while index < len(messages):
            next_message = messages[index]
            if _inactive_branch_id(next_message, active_branch_id) != branch_id:
                break
            branch_messages.append(next_message)
            index += 1
        result.append(
            build_branch_summary_message(
                branch_id,
                branch_messages,
                summarize_fn=summarize_fn,
            )
        )
    return result


def build_branch_summary_message(
    branch_id: str,
    messages: list[dict[str, Any]],
    summarize_fn: SummarizeFn | None = None,
) -> dict[str, Any]:
    """构建 LLM 可见的分支摘要消息。"""
    summary = _summarize_branch_messages(messages, summarize_fn)
    content = BRANCH_SUMMARY_PREFIX + summary + SUMMARY_SUFFIX
    return {
        "role": "user",
        "content": [{"type": "text", "text": content}],
        "metadata": {
            "type": "branch_summary",
            "branch_id": branch_id,
            "source_message_count": len(messages),
        },
    }


def _branch_summary_should_run(
    messages: list[dict[str, Any]],
    compact_token_threshold: int,
    budget_trigger_token_ratio: float,
) -> bool:
    if compact_token_threshold <= 0:
        return False
    trigger_threshold = compact_token_threshold * budget_trigger_token_ratio
    return estimate_message_tokens(messages) > trigger_threshold


def _inactive_branch_id(
    message: dict[str, Any],
    active_branch_id: str | None,
) -> str | None:
    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("type") == "branch_summary":
        return None
    branch_id = str(metadata.get("branch_id") or "").strip()
    if not branch_id:
        return None
    if bool(metadata.get("active_branch", False)):
        return None
    if active_branch_id is not None and branch_id == active_branch_id:
        return None
    return branch_id


def _summarize_branch_messages(
    messages: list[dict[str, Any]],
    summarize_fn: SummarizeFn | None,
) -> str:
    if summarize_fn is not None:
        raw = summarize_fn(messages)
        if isinstance(raw, Awaitable):
            import asyncio

            raw = asyncio.get_event_loop().run_until_complete(raw)
        summary = context_collapse_clean(str(raw).strip())
        if summary:
            return summary

    summary_lines = []
    for message in messages:
        content = _content_preview(message.get("content"))
        summary_lines.append(f"- {message.get('role')}: {content}")
    return "\n".join(summary_lines)


def _preserved_recent_count(
    messages: list[dict[str, Any]],
    instructions: CompactInstructions,
    base_count: int,
) -> int:
    """根据优先级估算需要额外保留的消息数。"""
    extra = 0
    has_top = "architecture_decision" in instructions.priorities
    has_high = "modified_file" in instructions.priorities
    if has_top and len(messages) > 20:
        extra += 2
    if has_high and len(messages) > 15:
        extra += 1
    return extra


def _protect_identifiers(
    summary: str,
    frozen: list[str],
) -> str:
    """标记不可变标识符，防止后续 LLM 摘要修改它们。"""
    if not frozen:
        return summary
    markers = [f"__FROZEN_{i}__" for i in range(len(frozen))]
    for marker, value in zip(markers, frozen):
        if value in summary:
            summary = summary.replace(value, marker)
    for marker, value in zip(markers, frozen):
        if marker in summary:
            summary = summary.replace(marker, f"`{value}`")
    return summary


def save_transcript(messages: list[dict[str, Any]], transcript_dir: Path) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = (
        transcript_dir
        / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    )
    with path.open("w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    return path


def build_compact_tool(controller: CompactController) -> ToolSpec:
    return ToolSpec(
        name="compact",
        description="Request manual context compaction before the next model call.",
        input_hint="empty",
        handler=lambda _input: controller.request(),
        risk="low",
    )


def _content_preview(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, list):
        rendered = []
        for part in content:
            if isinstance(part, dict):
                rendered.append(str(part.get("type", "block")))
            else:
                rendered.append(str(part))
        text = " ".join(rendered)
    else:
        text = str(content)
    return " ".join(text.split())[:180]
