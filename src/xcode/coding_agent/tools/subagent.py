from __future__ import annotations

from asyncio import run as async_run
from collections.abc import Callable
from typing import Any

from xcode.agent.agent import Agent
from xcode.ai.providers.base import ModelProvider
from xcode.agent.types import AgentTool, CancellationSignal, ToolSpec, ToolSpecAdapter


BUILD_SUBAGENT_PROMPTS: dict[str, str] = {
    "coding": (
        "You are an expert software engineer with full file system access. "
        "Complete the assigned task using available tools.\n\n"
        "- Use read/grep/glob/list_dir to explore before making changes.\n"
        "- Use bash to run tests, linters, or build commands.\n"
        "- Use edit/write to make changes.\n"
        "- Prefer small, focused changes and verify them.\n"
        "- Return a concise summary of what you did."
    ),
    "research": (
        "You are a thorough research assistant with file system and web access.\n\n"
        "- Use read/grep/glob/list_dir to inspect local files.\n"
        "- Use websearch/webfetch to gather current external information.\n"
        "- Cite sources and files when possible.\n"
        "- Return a structured summary of your findings."
    ),
    "default": (
        "You are a helpful AI assistant with file system access. "
        "Complete the assigned task and return a concise summary."
    ),
}


def build_subagent_tool(
    model: ModelProvider,
    coding_tools: list[ToolSpec],
    research_tools: list[ToolSpec],
    cancellation_token: CancellationSignal | None = None,
) -> ToolSpec:
    def handler(
        data: dict[str, Any],
        on_update: Callable[[str], None] | None = None,
    ) -> str:
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return "Error: prompt is required"
        subagent_type = str(data.get("subagent_type", "coding")).strip()
        system_prompt = BUILD_SUBAGENT_PROMPTS.get(subagent_type, BUILD_SUBAGENT_PROMPTS["default"])
        raw_tools = coding_tools if subagent_type == "coding" else research_tools

        async def _run() -> str:
            adapted: list[AgentTool] = [ToolSpecAdapter(s) for s in raw_tools]
            agent = Agent(tools=adapted, model=model, system_prompt=system_prompt)
            return await agent.prompt(
                prompt,
                signal=cancellation_token,
                on_update=on_update,
            )

        return async_run(_run())

    return ToolSpec(
        name="subagent",
        description=(
            "Launch a subagent to perform a self-contained task. "
            "The subagent runs independently with file system and web access.\n\n"
            "Available subagent types:\n"
            "- coding: Expert software engineer (default)\n"
            "- research: Research assistant with web access\n"
            "- default: General-purpose assistant"
        ),
        input_hint='JSON: {"description":"short label","prompt":"...", "subagent_type":"coding"}',
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short 3-7 word label for the delegated task",
                },
                "prompt": {
                    "type": "string",
                    "description": "Complete task prompt for the subagent",
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["coding", "research", "default"],
                    "description": "Type of subagent (default: coding)",
                },
            },
            "required": ["description", "prompt"],
            "additionalProperties": False,
        },
        prompt_snippet="Delegate tasks to a subagent with full tool access",
        prompt_guidelines=(
            "Use subagent for substantial independent investigation or implementation tasks.",
            "Include all necessary context in the prompt; subagent does not inherit conversation history.",
            "Do not poll delegated tasks; subagent returns the final result when done.",
        ),
    )
