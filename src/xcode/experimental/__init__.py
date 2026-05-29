"""Experimental extensions for Xcode.

Modules here are preserved as teaching / research assets but are not
part of the default coding agent harness path. They are opt-in via
`tools.enabled_groups` or imported explicitly when needed.

Modules:
- speculation: SpeculationPlanner (UI warmup hints, no side effects)
- tasks: TaskStore (JSON-file task persistence)
- worktree: WorktreeTaskRunner, build_worktree_tools (git worktree isolation)
- mcp: Model Context Protocol (MCP) clients and integration
"""
