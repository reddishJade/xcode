from __future__ import annotations

from typing import Any

"""消息转换层。将自定义消息角色转为 LLM 能理解的 user/assistant/toolResult 格式。

基于 TS pi/packages/agent/src/harness/messages.ts。
"""

COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
BRANCH_SUMMARY_SUFFIX = "\n</summary>"


def bash_execution_to_text(msg: Any) -> str:
    """将 BashExecutionMessage 转为纯文本描述。"""
    command = getattr(msg, "command", "") or ""
    output = getattr(msg, "output", "") or ""
    exit_code = getattr(msg, "exit_code", None)
    cancelled = getattr(msg, "cancelled", False)
    truncated = getattr(msg, "truncated", False)

    text = f"Ran `{command}`\n"
    if output:
        text += f"```\n{output}\n```"
    else:
        text += "(no output)"
    if cancelled:
        text += "\n\n(command cancelled)"
    elif exit_code is not None and exit_code != 0:
        text += f"\n\nCommand exited with code {exit_code}"
    if truncated:
        text += "\n\n[Output truncated]"
    return text


def convert_to_llm(messages: list[Any]) -> list[dict[str, Any]]:
    """将 AgentMessage list 转换成 LLM 兼容的 dict message list。"""
    result: list[dict[str, Any]] = []
    for m in messages:
        converted = _convert_one(m)
        if converted is not None:
            result.append(converted)
    return result


def _convert_one(m: Any) -> dict[str, Any] | None:
    role = getattr(m, "role", "") or ""

    if role == "system":
        return {"role": "system", "content": str(getattr(m, "content", ""))}

    if role == "user":
        return {"role": "user", "content": getattr(m, "content", "")}

    if role == "assistant":
        return _convert_assistant(m)

    if role == "toolResult":
        return {
            "role": "tool",
            "tool_call_id": getattr(m, "tool_call_id", ""),
            "content": getattr(m, "content", ""),
        }

    if role == "bashExecution":
        exclude = getattr(m, "exclude_from_context", False)
        if exclude:
            return None
        return {
            "role": "user",
            "content": [{"type": "text", "text": bash_execution_to_text(m)}],
        }

    if role == "custom":
        content = getattr(m, "content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        return {"role": "user", "content": content}

    if role == "branchSummary":
        summary = getattr(m, "summary", "")
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": BRANCH_SUMMARY_PREFIX + summary + BRANCH_SUMMARY_SUFFIX,
                }
            ],
        }

    if role == "compactionSummary":
        summary = getattr(m, "summary", "")
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": COMPACTION_SUMMARY_PREFIX
                    + summary
                    + COMPACTION_SUMMARY_SUFFIX,
                }
            ],
        }

    return None


def _convert_assistant(m: Any) -> dict[str, Any]:
    content = getattr(m, "content", [])
    content_blocks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        bt = getattr(block, "type", "")
        if bt == "text":
            content_blocks.append({"type": "text", "text": getattr(block, "text", "")})
        elif bt == "toolCall":
            tool_calls.append(
                {
                    "id": getattr(block, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": getattr(block, "arguments", {}),
                    },
                }
            )
        elif bt == "thinking":
            content_blocks.append(
                {"type": "text", "text": getattr(block, "thinking", "")}
            )

    result: dict[str, Any] = {"role": "assistant"}
    reasoning_content = getattr(m, "reasoning_content", None)
    if reasoning_content is not None:
        result["reasoning_content"] = reasoning_content
    if content_blocks:
        result["content"] = (
            "".join(
                b.get("text", "") for b in content_blocks if b.get("type") == "text"
            )
            or None
        )
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result
