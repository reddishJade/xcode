"""持久化 session 事件的版本化编码。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel

from xcode.harness.agent_runtime.events import (
    AgentHarnessEvent,
    AssistantEventBlock,
    AssistantStructuredEvent,
    AssistantTextBlock,
    CompactionStructuredEvent,
    MessageStartStructuredEvent,
    ReasoningDeltaStructuredEvent,
    TextDeltaStructuredEvent,
    ToolResultStructuredEvent,
    ToolUpdateStructuredEvent,
    ToolUseStructuredEvent,
    TurnEndStructuredEvent,
)


SESSION_EVENT_SCHEMA_VERSION = 1


def encode_session_event(event: AgentHarnessEvent) -> dict[str, Any]:
    """将运行时事件编码为稳定的持久化 envelope。"""
    return {
        "schema_version": SESSION_EVENT_SCHEMA_VERSION,
        "type": event.type,
        "step": event.step,
        "data": _event_payload(event),
        "correlation": asdict(event.correlation),
    }


def _event_payload(event: AgentHarnessEvent) -> object:
    if isinstance(event, (TextDeltaStructuredEvent, ReasoningDeltaStructuredEvent)):
        return event.data
    if isinstance(event, MessageStartStructuredEvent):
        if isinstance(event.data, BaseModel):
            return event.data.model_dump()
        return None
    if isinstance(event, TurnEndStructuredEvent):
        return {
            "tool_results": [
                {"tool_call_id": result.tool_call_id, "content": result.content}
                for result in event.data.tool_results
            ]
        }
    if isinstance(event, AssistantStructuredEvent):
        return [_assistant_block_payload(block) for block in event.data]
    if isinstance(event, ToolUseStructuredEvent):
        return {
            "id": event.data.id,
            "name": event.data.name,
            "input": event.data.input,
        }
    if isinstance(event, ToolUpdateStructuredEvent):
        return {
            "tool_call_id": event.data.tool_call_id,
            "tool_name": event.data.tool_name,
            "partial_result": event.data.partial_result,
        }
    if isinstance(event, ToolResultStructuredEvent):
        return {
            "tool_use_id": event.data.tool_use_id,
            "content": event.data.content,
            "status": event.data.status,
            "permission_notice": event.data.permission_notice,
            "type": "tool_result",
        }
    if isinstance(event, CompactionStructuredEvent):
        return {
            "messages_removed": event.data.messages_removed,
            "messages_after": event.data.messages_after,
            "summary_token_estimate": event.data.summary_token_estimate,
            "trigger": event.data.trigger,
        }
    return {
        "answer": event.data.answer,
        "steps": event.data.steps,
        "tool_calls": [
            {"id": call.id, "name": call.name, "input": call.input}
            for call in event.data.tool_calls
        ],
        "termination_reason": event.data.termination_reason.value,
        "metrics": event.data.metrics,
        "watchdog_reason": event.data.watchdog_reason,
        "error_detail": event.data.error_detail,
        "needs_follow_up": event.data.needs_follow_up,
        "last_agent": event.data.last_agent,
        "run_state": (
            event.data.run_state.to_dict()
            if event.data.run_state is not None
            else None
        ),
    }


def _assistant_block_payload(block: AssistantEventBlock) -> dict[str, object]:
    if isinstance(block, AssistantTextBlock):
        return {"type": "text", "text": block.text}
    return {
        "type": "tool_use",
        "id": block.id,
        "name": block.name,
        "input": block.input,
    }
