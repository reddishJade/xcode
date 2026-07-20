"""通用运行时的稳定 prompt 片段。"""

from __future__ import annotations

from xcode.harness.config import DEFAULT_PROMPT_MODULES

TOOL_DISCIPLINE = """<tool-discipline>
Tools must serve the current response. Use tools for external facts, workspace
evidence, code changes, file operations, diagnostics, and validation.
</tool-discipline>"""

CITATION_INSTRUCTION = """<citation-instruction>
When tool output provides source markers, preserve those markers in the answer.
</citation-instruction>"""

SEARCH_STRATEGY = """<search-strategy>
Use lexical search for discovery, then read complete relevant files before edits.
Identify the root cause before changing code and verify changed behavior.
</search-strategy>"""

SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "<system-prompt-dynamic-boundary />"
STABLE_PROMPT_MODULE_ORDER: tuple[str, ...] = (
    "identity",
    "tool_discipline",
    "citations",
    "tools",
    "search_strategy",
)
DYNAMIC_PROMPT_MODULE_ORDER: tuple[str, ...] = ("environment", "cwd")
VOLATILE_PROMPT_MODULE_ORDER: tuple[str, ...] = (
    "git_preflight",
    "contextual_retrieval",
    "notices",
)


def prompt_version() -> str:
    """返回通用 prompt 模块的版本。"""
    return repr(
        (
            TOOL_DISCIPLINE,
            CITATION_INSTRUCTION,
            SEARCH_STRATEGY,
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
            STABLE_PROMPT_MODULE_ORDER,
            DYNAMIC_PROMPT_MODULE_ORDER,
            VOLATILE_PROMPT_MODULE_ORDER,
            DEFAULT_PROMPT_MODULES,
        )
    )
