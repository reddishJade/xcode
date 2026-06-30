from __future__ import annotations


import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from xcode.agent.compaction import estimate_tokens
from xcode.agent.message_converter import BRANCH_SUMMARY_PREFIX, SUMMARY_SUFFIX
from xcode.agent.config import CompactInstructions
from ..skill_activation import is_skill_activation_content
from ..skills import ToolSpec

"""分层上下文压缩工具。"""


# ── 会话树条目类型（用于无损历史追踪）──


class CompactionEntry:
    """压缩条目，记录一次压缩操作的元数据。

    当 LayeredCompactor 执行压缩时，创建此条目并追加到会话树中。
    完整的消息历史可通过 transcript JSONL + entries 重建。
    """

    def __init__(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        read_files: set[str] | None = None,
        modified_files: set[str] | None = None,
        parent_id: str | None = None,
        entry_id: str | None = None,
    ) -> None:
        self.id: str = entry_id or uuid4().hex[:12]
        self.parent_id: str | None = parent_id
        self.type: str = "compaction"
        self.timestamp: float = datetime.now(timezone.utc).timestamp()
        self.summary: str = summary
        self.first_kept_entry_id: str = first_kept_entry_id
        self.tokens_before: int = tokens_before
        self.read_files: tuple[str, ...] = tuple(sorted(read_files or set()))
        self.modified_files: tuple[str, ...] = tuple(sorted(modified_files or set()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "first_kept_entry_id": self.first_kept_entry_id,
            "tokens_before": self.tokens_before,
            "read_files": list(self.read_files),
            "modified_files": list(self.modified_files),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactionEntry:
        entry = cls.__new__(cls)
        entry.id = str(data.get("id", uuid4().hex[:12]))
        entry.parent_id = data.get("parent_id")
        entry.type = "compaction"
        entry.timestamp = float(data.get("timestamp", 0))
        entry.summary = str(data.get("summary", ""))
        entry.first_kept_entry_id = str(data.get("first_kept_entry_id", ""))
        entry.tokens_before = int(data.get("tokens_before", 0))
        entry.read_files = tuple(data.get("read_files", []) or [])
        entry.modified_files = tuple(data.get("modified_files", []) or [])
        return entry


# 文件操作工具名集合（用于累积文件跟踪）
_READ_TOOLS: frozenset[str] = frozenset({"read_file", "read"})
_MODIFY_TOOLS: frozenset[str] = frozenset({"edit_file", "write_file", "write", "edit", "apply_patch"})


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
        checkpoint_dir: Path | None = None,
        keep_recent_tool_results: int = 2,
        max_tool_result_chars: int = 100,
        max_recent_messages: int = 8,
        keep_recent_tokens: int = 20000,
        large_tool_output_chars: int = 20_000,
        large_tool_output_head_chars: int = 10_000,
        large_tool_output_tail_chars: int = 10_000,
        compact_token_threshold: int = 32_000,
        budget_trigger_token_ratio: float = 0.5,
        on_compact: Callable[[str], None] | None = None,
        compact_instructions: CompactInstructions | None = None,
        summarize_fn: SummarizeFn | None = None,
        active_branch_id: str | None = None,
        reserve_tokens: int = 16384,
    ) -> None:
        self.transcript_dir = transcript_dir
        self.checkpoint_dir = checkpoint_dir
        self.compact_instructions = compact_instructions
        self.summarize_fn = summarize_fn
        self.active_branch_id = active_branch_id
        self.keep_recent_tool_results = keep_recent_tool_results
        self.max_tool_result_chars = max_tool_result_chars
        self.max_recent_messages = max_recent_messages
        self.keep_recent_tokens = keep_recent_tokens
        self.large_tool_output_chars = large_tool_output_chars
        self.large_tool_output_head_chars = large_tool_output_head_chars
        self.large_tool_output_tail_chars = large_tool_output_tail_chars
        self.compact_token_threshold = compact_token_threshold
        self.budget_trigger_token_ratio = budget_trigger_token_ratio
        self.reserve_tokens = reserve_tokens
        self.on_compact = on_compact
        self.last_transcript_path: Path | None = None
        # 累积文件跟踪（跨压缩轮次）
        self._cumulative_read_files: set[str] = set()
        self._cumulative_modified_files: set[str] = set()
        # 会话树条目（用于无损历史追踪）
        self.entries: list[CompactionEntry] = []
        self._last_entry_id: str | None = None

    def _accumulate_file_ops(self, messages: list[dict[str, Any]]) -> None:
        """从消息中提取文件操作，合并到累积跟踪状态。"""
        reads, modifies = _extract_file_ops_from_messages(messages)
        self._cumulative_read_files.update(reads)
        self._cumulative_modified_files.update(modifies)

    def _write_checkpoint(self, summary: str) -> None:
        """将当前会话状态写入 checkpoint.md。"""
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "checkpoint.md"

        # 解析结构化摘要
        sections = self._parse_checkpoint_sections(summary)

        read_list = "\n".join(
            f"  - {f}" for f in sorted(self._cumulative_read_files)
        ) or "  (none)"
        modified_list = "\n".join(
            f"  - {f}" for f in sorted(self._cumulative_modified_files)
        ) or "  (none)"

        content = (
            "# Session checkpoint\n"
            "_Generated by context compaction. Captures session state up to this point._\n"
            "\n"
            "## Goal\n"
            f"{sections.get('goal', '(not specified)')}\n"
            "\n"
            "## Progress\n"
            f"{sections.get('progress', '(not specified)')}\n"
            "\n"
            "## Key Decisions\n"
            f"{sections.get('key decisions', '(none)')}\n"
            "\n"
            "## Files\n"
            "### Read\n"
            f"{read_list}\n"
            "\n"
            "### Modified\n"
            f"{modified_list}\n"
            "\n"
            "## Next Steps\n"
            f"{sections.get('next steps', '(not specified)')}\n"
        )
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _parse_checkpoint_sections(summary: str) -> dict[str, str]:
        """从结构化压缩摘要中提取 sections。"""
        content = summary.removeprefix("[Compressed]").strip()
        sections: dict[str, str] = {}
        current_key = None
        current_lines: list[str] = []
        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("## "):
                if current_key is not None:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = stripped[3:].strip().lower()
                current_lines = []
            elif current_key is not None:
                current_lines.append(stripped)
        if current_key is not None:
            sections[current_key] = "\n".join(current_lines).strip()
        return sections

    def __call__(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compacted = stale_snip_file_reads(messages)
        preserved_tool_results = latest_read_file_tool_result_ids(compacted)
        preserved_tool_results.update(activated_skill_tool_result_ids(compacted))
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

        # 在最终摘要前，从待摘要消息中提取文件操作
        if len(branch_compacted) > self.max_recent_messages + 1:
            recent_count = _compute_recent_count_from_tokens(
                branch_compacted, self.keep_recent_tokens
            )
            effective_recent = min(self.max_recent_messages, recent_count)
            older_start = max(1, len(branch_compacted) - effective_recent)
            older_msgs = branch_compacted[1:older_start]
            self._accumulate_file_ops(older_msgs)

        final_messages = summarize_messages(
            branch_compacted,
            max_recent_messages=self.max_recent_messages,
            keep_recent_tokens=self.keep_recent_tokens,
            compact_instructions=self.compact_instructions,
            summarize_fn=self.summarize_fn,
            read_files=self._cumulative_read_files,
            modified_files=self._cumulative_modified_files,
        )
        if len(final_messages) > 1:
            second_msg = final_messages[1]
            if second_msg.get("role") == "user" and str(
                second_msg.get("content")
            ).startswith("[Compressed]"):
                content_val = second_msg.get("content")
                assert isinstance(content_val, str)
                cleaned_content = context_collapse_clean(content_val)
                # 当实际发生压缩时，创建 CompactionEntry 追加到会话树
                # 使用系统提示的 id 作为 first_kept_entry_id 锚点
                first_kept_idx = len(final_messages) - 1
                # 从保留的消息中找第一个 user 或 assistant 消息作为锚点
                for i in range(1, len(final_messages)):
                    role = final_messages[i].get("role", "")
                    if role in ("user", "assistant"):
                        first_kept_idx = i
                        break

                entry = CompactionEntry(
                    summary=cleaned_content,
                    first_kept_entry_id=f"idx:{first_kept_idx}",
                    tokens_before=len(messages),
                    read_files=set(self._cumulative_read_files),
                    modified_files=set(self._cumulative_modified_files),
                    parent_id=self._last_entry_id,
                )
                self.entries.append(entry)
                self._last_entry_id = entry.id

                second_msg["content"] = cleaned_content
                if self.on_compact is not None:
                    self.on_compact(cleaned_content)

                # 写入 checkpoint.md
                self._write_checkpoint(cleaned_content)

        return final_messages

    def save_entries(self, path: Path) -> None:
        """将会话树条目持久化到 JSONL 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def load_entries(self, path: Path) -> None:
        """从 JSONL 文件加载会话树条目。"""
        if not path.exists():
            return
        loaded: list[CompactionEntry] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        loaded.append(CompactionEntry.from_dict(data))
                    except (json.JSONDecodeError, KeyError):
                        continue
        self.entries = loaded
        if loaded:
            self._last_entry_id = loaded[-1].id

    def entries_dicts(self) -> list[dict[str, Any]]:
        """返回所有条目的 dict 列表，用于序列化。"""
        return [e.to_dict() for e in self.entries]


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


def activated_skill_tool_result_ids(messages: list[dict[str, Any]]) -> set[str]:
    """返回包含完整技能激活内容的 tool_result id。"""
    result_ids: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "tool_result"):
                continue
            if is_skill_activation_content(part.get("content", "")):
                result_ids.add(str(part.get("tool_use_id", "")))
    return {result_id for result_id in result_ids if result_id}


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
    protected_result_ids = set(preserve_tool_result_ids or ())
    protected_result_ids.update(activated_skill_tool_result_ids(compacted))
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
                    if tool_use_id in protected_result_ids:
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


def estimate_text_tokens(text: str) -> int:
    return estimate_tokens(text)


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(
        estimate_text_tokens(json.dumps(message, ensure_ascii=False, default=str))
        for message in messages
    )


type SummarizeFn = Callable[[list[dict[str, Any]]], str | Awaitable[str]]
"""可选的 LLM 摘要函数。接收旧消息列表，返回摘要文本。"""


# 结构化摘要格式（镜像 pi 的摘要设计，便于 LLM 理解和消费）
SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the following conversation turns into a structured markdown summary. "
    "Preserve architecture decisions, file changes, and TODO items. "
    "Use the following format:\n\n"
    "## Goal\n"
    "[What the user is trying to accomplish]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks]\n\n"
    "### In Progress\n"
    "- [ ] [Current work]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Rationale]\n\n"
    "## Next Steps\n"
    "1. [What should happen next]\n"
    "Output only the summary, no preamble."
)


# 结构化摘要文本标签（用于纯文本 fallback）
_STRUCTURED_FALLBACK_TEMPLATE = """## Goal
{goal}

## Progress
### Done
{done}

### In Progress
{in_progress}

## Key Decisions
{decisions}

## Next Steps
{next_steps}"""


def build_summarize_prompt(older: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建 LLM 摘要的消息列表。"""
    return [
        {"role": "user", "content": SUMMARIZE_SYSTEM_PROMPT},
        *older,
    ]


def _compute_recent_count_from_tokens(
    messages: list[dict[str, Any]],
    keep_recent_tokens: int,
) -> int:
    """从后向前扫描消息，累计 token 直到 keep_recent_tokens，返回应保留的消息数。

    始终保留至少 1 条消息（以及系统提示）。
    """
    accumulated = 0
    count = 0
    # 从最后一条消息开始反向扫描（跳过系统提示 message[0]）
    for i in range(len(messages) - 1, 0, -1):
        tokens = estimate_message_tokens([messages[i]])
        if accumulated + tokens > keep_recent_tokens and count > 0:
            break
        accumulated += tokens
        count += 1
    return max(count, 1)


def _find_turn_boundary(
    messages: list[dict[str, Any]],
    raw_index: int,
) -> int:
    """将切分索引调整到最近的 turn 边界。

    Turn 规则：
    - 一个 turn 从 user 消息开始
    - 之后跟随 assistant 和 tool 消息，直到下一个 user 消息
    - 有效切分点：user 消息、assistant 消息
    - 永不切在 tool_result 消息上（必须与对应的 tool_call 在同一 turn 内）
    - 也永不切在 role='tool' 的消息上

    调整策略：
    1. 如果 raw_index 指向 tool 消息，向前回溯到最近的 assistant 或 user
    2. 如果 raw_index 指向 assistant 消息（且它前面是 tool/assistant），
       检查是否可以把前面的所有 tool 结果一起保留
    """
    index = raw_index
    # 确保不超出范围
    if index <= 1:
        return 1
    if index >= len(messages):
        return len(messages) - 1

    # 从 raw_index 开始回溯，找到安全的切分点
    while index > 1:
        role = messages[index].get("role", "")
        if role == "user":
            return index  # user 消息永远是安全的切分点
        if role == "assistant":
            return index  # assistant 消息也是安全的切分点
        # tool / tool_result → 必须与对应的 assistant 消息一起保留
        index -= 1

    return index  # fallback


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    """判断消息是否为 tool result 消息。"""
    role = message.get("role", "")
    if role == "tool":
        return True
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                return True
    return False


def summarize_messages(
    messages: list[dict[str, Any]],
    max_recent_messages: int = 8,
    keep_recent_tokens: int = 20000,
    compact_instructions: CompactInstructions | None = None,
    summarize_fn: SummarizeFn | None = None,
    read_files: set[str] | None = None,
    modified_files: set[str] | None = None,
) -> list[dict[str, Any]]:
    if len(messages) <= max_recent_messages + 1:
        return messages

    # token 驱动的保留计数：min 确保不超过 max_recent_messages 上限
    token_count = _compute_recent_count_from_tokens(messages, keep_recent_tokens)
    effective_recent = min(max_recent_messages, token_count)

    raw_recent_start = max(1, len(messages) - effective_recent)
    # 将切分点对齐到 turn 边界，避免切在 tool result 中间
    recent_start = _find_turn_boundary(messages, raw_recent_start)
    protected_indices = _activated_skill_message_indices(messages)
    protected_older = [
        message
        for index, message in enumerate(messages[1:recent_start], start=1)
        if index in protected_indices
    ]
    older = [
        message
        for index, message in enumerate(messages[1:recent_start], start=1)
        if index not in protected_indices
    ]

    if not older:
        return [messages[0], *protected_older, *messages[recent_start:]]

    if summarize_fn:
        raw = summarize_fn(older)
        if isinstance(raw, Awaitable):
            raw = asyncio.get_event_loop().run_until_complete(raw)
        summary_content = str(raw).strip()
        if not summary_content.startswith("[Compressed]"):
            summary_content = "[Compressed]\n" + summary_content
    else:
        summary_content = "[Compressed]\n" + _fallback_structured_summary(older)

    if compact_instructions and compact_instructions.frozen_identifiers:
        summary_content = _protect_identifiers(
            summary_content, compact_instructions.frozen_identifiers
        )

    summary = {
        "role": "user",
        "content": summary_content,
    }
    # 注入累积文件操作信息到摘要
    if read_files or modified_files:
        tracked = _render_file_tracking(
            read_files or set(), modified_files or set()
        )
        if tracked:
            summary_content += "\n\n" + tracked

    return [messages[0], summary, *protected_older, *messages[recent_start:]]


def _activated_skill_message_indices(messages: list[dict[str, Any]]) -> set[int]:
    """返回需成对保留的 load_skill tool_use 与激活结果消息索引。"""
    activation_ids: set[str] = set()
    protected_indices: set[int] = set()
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "tool_result"):
                continue
            if not is_skill_activation_content(part.get("content", "")):
                continue
            tool_use_id = str(part.get("tool_use_id", ""))
            if tool_use_id:
                activation_ids.add(tool_use_id)
                protected_indices.add(index)

    if not activation_ids:
        return protected_indices
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict)
            and part.get("type") == "tool_use"
            and str(part.get("id", "")) in activation_ids
            for part in content
        ):
            protected_indices.add(index)
    return protected_indices


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
        schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )


def _extract_file_ops_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """从消息列表中提取所有被读取和修改的文件路径。

    Returns:
        (read_files, modified_files) 两组集合
    """
    read_files: set[str] = set()
    modified_files: set[str] = set()

    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "tool_use"):
                continue
            tool_name = str(part.get("name", ""))
            tool_input = part.get("input", {})
            if not isinstance(tool_input, dict):
                continue
            path = str(tool_input.get("path", "") or "").strip()
            if not path:
                continue
            if tool_name in _READ_TOOLS:
                read_files.add(path)
            elif tool_name in _MODIFY_TOOLS:
                modified_files.add(path)

    return read_files, modified_files


def _render_file_tracking(read_files: set[str], modified_files: set[str]) -> str:
    """将文件跟踪信息渲染为 XML 标签块。"""
    parts: list[str] = []
    if read_files:
        paths = "\n".join(sorted(read_files))
        parts.append(f"<read-files>\n{paths}\n</read-files>")
    if modified_files:
        paths = "\n".join(sorted(modified_files))
        parts.append(f"<modified-files>\n{paths}\n</modified-files>")
    return "\n".join(parts)


def _fallback_structured_summary(older: list[dict[str, Any]]) -> str:
    """生成结构化摘要的纯文本 fallback 版本。

    当没有 LLM summarize_fn 时使用，输出的格式与 LLM 摘要一致。
    """
    # 提取各角色消息预览
    goals: list[str] = []
    done_items: list[str] = []
    decisions: list[str] = []
    next_steps: list[str] = []

    for message in older:
        preview = _content_preview(message.get("content"))
        role = message.get("role", "")
        entry = f"- {role}: {preview}"
        if role == "user":
            goals.append(entry)
        elif role == "assistant":
            done_items.append(entry)
        else:
            next_steps.append(entry)

    return _STRUCTURED_FALLBACK_TEMPLATE.format(
        goal="\n".join(goals) if goals else "Continue previous work",
        done="\n".join(done_items) if done_items else "Work in progress",
        in_progress="(see recent messages)",
        decisions="\n".join(decisions) if decisions else "(see context)",
        next_steps="\n".join(next_steps) if next_steps else "Continue from summary",
    )


def _content_preview(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, list):
        rendered = []
        for part in content:
            rendered.append(str(part.get("type", "block")))
        text = " ".join(rendered)
    else:
        text = str(content)
    return " ".join(text.split())[:180]
