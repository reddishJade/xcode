from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any

from ...harness.agent_runtime.events import (
    FinalMessage,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallReady,
    UsageUpdate,
)

"""OpenAI tool schema and stream delta codecs consolidated."""


def make_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    import copy

    s = copy.deepcopy(schema)

    def process(node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        t = node.get("type")
        if t == "object" or "properties" in node:
            properties = node.get("properties", {})
            if properties:
                node["required"] = list(properties.keys())
                node["additionalProperties"] = False

        node.pop("minLength", None)
        node.pop("maxLength", None)
        node.pop("minItems", None)
        node.pop("maxItems", None)

        if "properties" in node:
            node["properties"] = {k: process(v) for k, v in node["properties"].items()}
        if "items" in node:
            node["items"] = process(node["items"])
        if "anyOf" in node:
            node["anyOf"] = [process(item) for item in node["anyOf"]]
        if "allOf" in node:
            node["allOf"] = [process(item) for item in node["allOf"]]
        if "oneOf" in node:
            node["oneOf"] = [process(item) for item in node["oneOf"]]

        return node

    return process(s)


def to_chat_tool(
    name: str, description: str, schema: dict | None, strict: bool = False
) -> dict[str, Any]:
    resolved = schema or {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": description,
            }
        },
    }
    if strict:
        resolved = make_schema_strict(resolved)
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": resolved,
                "strict": True,
            },
        }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": resolved,
        },
    }


def to_responses_tool(
    name: str, description: str, schema: dict | None
) -> dict[str, Any]:
    resolved = schema or {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": description,
            }
        },
    }
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": resolved,
    }


def to_openai_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if "content" in message:
            if role == "tool" and content is None:
                result_content: str | None = ""
            else:
                result_content = None if content is None else str(content)
        else:
            result_content = ""

        if isinstance(content, list):
            converted.extend(
                _content_blocks_to_openai_messages(
                    role,
                    content,
                    reasoning_content=message.get("reasoning_content"),
                    prefix=message.get("prefix"),
                )
            )
        else:
            result: dict[str, Any] = {"role": role, "content": result_content}
            if "tool_calls" in message:
                result["tool_calls"] = _normalize_chat_tool_calls(message["tool_calls"])
            if (
                "reasoning_content" in message
                and message["reasoning_content"] is not None
            ):
                result["reasoning_content"] = message["reasoning_content"]
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "prefix" in message:
                result["prefix"] = message["prefix"]
            converted.append(result)
    return converted


def _normalize_chat_tool_calls(tool_calls: object) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(tool_calls, list):
        return normalized
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fixed_call = dict(call)
        function = fixed_call.get("function")
        if isinstance(function, dict):
            fixed_function = dict(function)
            arguments = fixed_function.get("arguments", "")
            if not isinstance(arguments, str):
                fixed_function["arguments"] = json.dumps(
                    arguments,
                    ensure_ascii=False,
                )
            fixed_call["function"] = fixed_function
        normalized.append(fixed_call)
    return normalized


def parse_tool_arguments(raw_arguments: str) -> object:
    try:
        return json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {"input": raw_arguments}


def _content_blocks_to_openai_messages(
    role: str,
    content: list,
    reasoning_content: str | None = None,
    prefix: bool | None = None,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    assistant_tool_calls = []
    text_parts = []
    for part in content:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        if part.get("type") == "tool_result":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": str(part.get("tool_use_id", "")),
                    "content": str(part.get("content", "")),
                }
            )
        elif part.get("type") == "tool_use":
            assistant_tool_calls.append(
                {
                    "id": str(part.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(part.get("name", "")),
                        "arguments": json.dumps(
                            part.get("input", {}),
                            ensure_ascii=False,
                        ),
                    },
                }
            )
        elif part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))
    if assistant_tool_calls:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts) or None,
            "tool_calls": assistant_tool_calls,
        }
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if prefix is not None:
            msg["prefix"] = prefix
        converted.insert(0, msg)
    elif text_parts:
        msg = {"role": role, "content": "".join(text_parts)}
        if reasoning_content is not None and role == "assistant":
            msg["reasoning_content"] = reasoning_content
        if prefix is not None and role == "assistant":
            msg["prefix"] = prefix
        converted.insert(0, msg)
    return converted


def tool_call_from_chat(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    return {
        "id": str(getattr(call, "id", "")),
        "name": str(getattr(function, "name", "")),
        "input": parse_tool_arguments(str(getattr(function, "arguments", "{}"))),
    }


def tool_call_from_response_item(item: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(item, "call_id", None) or getattr(item, "id", "")),
        "name": str(getattr(item, "name", "")),
        "input": parse_tool_arguments(str(getattr(item, "arguments", "{}"))),
    }


def chat_stream_to_events(
    stream: Iterable[object],
) -> Iterator[TextDelta | ReasoningDelta | ToolCallReady | UsageUpdate]:
    """Yields provider events dicts: TextDelta, ReasoningDelta, ToolCallReady, UsageUpdate."""
    calls: dict[int, dict[str, str]] = defaultdict(
        lambda: {"id": "", "name": "", "arguments": ""}
    )
    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            yield UsageUpdate(input_tokens, output_tokens)

        choices = getattr(chunk, "choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield ReasoningDelta(str(reasoning))
        text = getattr(delta, "content", None)
        if text:
            yield TextDelta(str(text))
        for call in getattr(delta, "tool_calls", None) or []:
            index = int(getattr(call, "index", 0) or 0)
            current = calls[index]
            call_id = getattr(call, "id", None)
            if call_id:
                current["id"] = str(call_id)
            function = getattr(call, "function", None)
            name = getattr(function, "name", None)
            if name:
                current["name"] = str(name)
            arguments = getattr(function, "arguments", None)
            if arguments:
                current["arguments"] += str(arguments)

    ready = [
        ToolCall(
            id=current["id"],
            name=current["name"],
            input=parse_tool_arguments(current["arguments"]),
        )
        for index, current in sorted(calls.items())
    ]
    if ready:
        yield ToolCallReady(ready)


def responses_stream_to_events(
    stream: Iterable[object],
) -> Iterator[TextDelta | FinalMessage | ToolCallReady]:  # noqa: C901
    for event in stream:
        event_type = str(getattr(event, "type", ""))
        if event_type.endswith(".delta"):
            text = getattr(event, "delta", None)
            if text:
                yield TextDelta(str(text))
        elif event_type.endswith(".completed"):
            response = getattr(event, "response", None)
            if response is not None:
                for provider_event in responses_to_events(response):
                    if not isinstance(provider_event, TextDelta):
                        yield provider_event


def responses_to_events(
    response: object,
) -> list[TextDelta | FinalMessage | ToolCallReady]:
    events: list[TextDelta | FinalMessage | ToolCallReady] = []
    text = getattr(response, "output_text", None)
    if text:
        events.append(TextDelta(str(text)))
    calls = []
    for item in getattr(response, "output", []) or []:
        item_type = str(getattr(item, "type", ""))
        if item_type == "message":
            for content in getattr(item, "content", []) or []:
                content_text = getattr(content, "text", None)
                if content_text:
                    events.append(TextDelta(str(content_text)))
        elif item_type in {"function_call", "tool_call"}:
            calls.append(tool_call_from_response_item(item))
    if calls:
        events.append(ToolCallReady([ToolCall(**c) for c in calls]))
    else:
        events.append(FinalMessage(str(text or ""), "end_turn"))
    return events
