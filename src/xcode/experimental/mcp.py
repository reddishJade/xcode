"""MCP 工具注册与配置。

根据 mcp_config.json 加载 MCP 服务器工具，支持 defer_loading 延迟加载、
缓存持久化和显式风险声明。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from xcode.harness.skills import ToolInput, ToolSpec

from . import mcp_client as _mcp_mod

# ── 配置与风险声明 ──


def compute_config_hash(server_config: dict[str, Any]) -> str:
    data = {
        "command": server_config.get("command"),
        "args": server_config.get("args", []),
        "env": server_config.get("env", {}),
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_mcp_tool_risk(server_config: dict[str, Any], tool_name: str) -> str:
    """从 server overrides 读取工具风险；未审核工具默认 high。"""
    overrides = server_config.get("overrides", {})
    risk = None
    if isinstance(overrides, dict) and tool_name in overrides:
        override = overrides[tool_name]
        risk = (
            override.get("risk")
            if isinstance(override, dict)
            else override
            if isinstance(override, str)
            else None
        )
    if risk in {"low", "medium", "high"}:
        return risk
    return "high"


# ── 缓存工具 ──


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            logging.warning(
                "failed to load MCP cache from %s", cache_path, exc_info=True
            )
    return {"servers": {}}


def _save_cache(cache_path: Path, cache_data: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── 引导工具 ──


def build_fetch_tools_tool(
    project_root: Path, server_name: str, server_config: dict[str, Any]
) -> ToolSpec:
    """创建用于冷启动延迟加载服务器并拉取工具列表的引导工具。"""

    def handler(_args: ToolInput) -> str:
        config_hash = compute_config_hash(server_config)
        try:
            command = [server_config["command"]] + server_config.get("args", [])
            env = server_config.get("env")
            client = _mcp_mod.McpClient(command, env)
            client.start()
            tools_list = client.list_tools()
            client.stop()

            for tool in tools_list:
                tool["risk"] = resolve_mcp_tool_risk(server_config, tool["name"])

            cache_path = project_root / ".local" / "mcp_cache.json"
            cache_data = _load_cache(cache_path)
            cache_data.setdefault("servers", {})[server_name] = {
                "config_hash": config_hash,
                "tools": tools_list,
            }
            _save_cache(cache_path, cache_data)
            return f"Successfully fetched {len(tools_list)} tools from server '{server_name}' and populated the cache. Please use mcp_tool_search to retrieve their schemas now!"
        except Exception as e:
            return f"Error fetching tools from server '{server_name}': {e}"

    return ToolSpec(
        name=f"mcp__{server_name}__fetch_tools",
        description=f"Bootstrap and fetch tool list for deferred server '{server_name}' to populate the cache.",
        input_hint="{}",
        handler=handler,
        risk="low",
        schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        read_only=True,
        group="mcp",
    )


def _format_tool_match(server_name: str, tool: dict[str, Any]) -> str:
    name = tool["name"]
    desc = tool.get("description", "")
    schema = tool.get("inputSchema", {})
    required = schema.get("required", [])
    props = schema.get("properties", {})
    param_lines = [
        f"    - **{p_name}** ({p_info.get('type', 'any')}): "
        f"{p_info.get('description', '')}"
        f"{' (required)' if p_name in required else ''}"
        for p_name, p_info in props.items()
    ]
    params_str = (
        "\n".join(param_lines) if param_lines else "    - No parameters required."
    )
    return (
        f"- **Tool Name**: `mcp__{server_name}__{name}`\n"
        f"  **Description**: {desc}\n"
        f"  **Schema / Parameters**:\n{params_str}"
    )


def build_mcp_tool_search(project_root: Path, deferred_servers: set[str]) -> ToolSpec:
    """创建用于搜索和获取延迟加载工具完整参数 Schema 的工具。"""

    def handler(args: ToolInput) -> str:
        query = str(args.get("query", "")).strip().lower()
        if not query:
            return "Please provide a query to search."

        cache_data = _load_cache(project_root / ".local" / "mcp_cache.json")
        results = []

        for server_name in deferred_servers:
            server_entry = cache_data.get("servers", {}).get(server_name, {})
            tools = server_entry.get("tools", [])
            if not tools:
                results.append(
                    f"### Server '{server_name}'\n"
                    f"Schema not yet loaded. Please invoke the fetch_tools tool "
                    f"(e.g. `mcp__{server_name}__fetch_tools`) to trigger schema fetch."
                )
                continue

            matched = [
                t
                for t in tools
                if query in t.get("name", "").lower()
                or query in t.get("description", "").lower()
                or query == "all"
            ]
            if matched:
                results.append(f"### Server '{server_name}' matched tools:")
                for tool in matched:
                    results.append(_format_tool_match(server_name, tool))

        return (
            "\n\n".join(results)
            if results
            else f"No tools found matching query: {query}"
        )

    return ToolSpec(
        name="mcp_tool_search",
        description="Search and retrieve full schema, descriptions, and parameters for deferred MCP tools.",
        input_hint='{"query": "keyword_or_all"}',
        handler=handler,
        risk="low",
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword to search within tool names or descriptions, or 'all' to list all tools.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        read_only=True,
        group="mcp",
    )


# ── 工具执行包装 ──


def _make_handler(
    ref: _mcp_mod.LazyClientRef,
    tool_name: str,
    deferred: bool,
    server_name: str,
    cache_path: Path,
) -> Any:
    def handler(args: ToolInput) -> str:
        if deferred:
            try:
                cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
                tool_entry = next(
                    (
                        cached_tool
                        for cached_tool in cache_data.get("servers", {})
                        .get(server_name, {})
                        .get("tools", [])
                        if cached_tool["name"] == tool_name
                    ),
                    None,
                )
                if tool_entry:
                    real_schema = tool_entry.get("inputSchema", {})
                    missing = [
                        field_name
                        for field_name in real_schema.get("required", [])
                        if field_name not in args
                    ]
                    if missing:
                        raise ValueError(
                            f"Missing required parameters for deferred tool {tool_name}: {', '.join(missing)}"
                        )
            except Exception as validation_error:
                if isinstance(validation_error, ValueError):
                    raise

        client_instance = ref.get_or_create()
        response = client_instance.call_tool(tool_name, args)
        if "content" in response:
            parts = [
                block.get("text", "")
                for block in response["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content_str = "\n".join(parts)
            if response.get("isError", False):
                raise RuntimeError(content_str)
            return content_str
        return str(response)

    return handler


# ── 主入口 ──


def build_mcp_tools(project_root: Path) -> tuple[ToolSpec, ...]:
    """根据配置文件加载和构建所有 MCP 工具，支持 defer_loading 延迟加载。"""
    config_path = _mcp_config_path(project_root)
    if config_path is None:
        return ()

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: Failed to load MCP config {config_path}: {e}")
        return ()

    mcp_servers = config_data.get("mcpServers", {})
    if not mcp_servers:
        return ()

    cache_path = project_root / ".local" / "mcp_cache.json"
    cache_data = _load_cache(cache_path)
    tools_to_register: list[ToolSpec] = []
    deferred_servers: set[str] = set()

    for server_name, server_config in mcp_servers.items():
        is_deferred = bool(server_config.get("defer_loading", False))
        if is_deferred:
            deferred_servers.add(server_name)

        tools_list = _tools_for_server(
            project_root,
            cache_path,
            cache_data,
            server_name,
            server_config,
            is_deferred,
            tools_to_register,
        )
        lazy_ref = _mcp_mod.LazyClientRef(server_name, server_config)
        for tool in tools_list:
            tools_to_register.append(
                _build_registered_mcp_tool(
                    server_name,
                    server_config,
                    tool,
                    lazy_ref,
                    is_deferred,
                    cache_path,
                )
            )

    if deferred_servers:
        tools_to_register.append(build_mcp_tool_search(project_root, deferred_servers))

    return tuple(tools_to_register)


def _mcp_config_path(project_root: Path) -> Path | None:
    local_config = project_root / ".local" / "mcp_config.json"
    root_config = project_root / "mcp_config.json"
    if local_config.exists():
        return local_config
    if root_config.exists():
        return root_config
    return None


def _tools_for_server(
    project_root: Path,
    cache_path: Path,
    cache_data: dict[str, Any],
    server_name: str,
    server_config: dict[str, Any],
    is_deferred: bool,
    tools_to_register: list[ToolSpec],
) -> list[dict[str, Any]]:
    config_hash = compute_config_hash(server_config)
    cached_entry = cache_data.get("servers", {}).get(server_name, {})
    if cached_entry.get("config_hash") == config_hash:
        cached_tools = cached_entry.get("tools")
        if isinstance(cached_tools, list):
            return cached_tools

    if is_deferred:
        tools_to_register.append(
            build_fetch_tools_tool(project_root, server_name, server_config)
        )
        return []

    return _query_server_tools(
        cache_path,
        cache_data,
        server_name,
        server_config,
        config_hash,
        cached_entry,
    )


def _query_server_tools(
    cache_path: Path,
    cache_data: dict[str, Any],
    server_name: str,
    server_config: dict[str, Any],
    config_hash: str,
    cached_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        command = [server_config["command"]] + server_config.get("args", [])
        client = _mcp_mod.McpClient(command, server_config.get("env"))
        client.start()
        tools_list = client.list_tools()
        client.stop()
        for tool in tools_list:
            tool["risk"] = resolve_mcp_tool_risk(server_config, tool["name"])
        cache_data.setdefault("servers", {})[server_name] = {
            "config_hash": config_hash,
            "tools": tools_list,
        }
        _save_cache(cache_path, cache_data)
        return tools_list
    except Exception as e:
        print(f"Error querying tools from MCP server '{server_name}': {e}")
        cached_tools = cached_entry.get("tools")
        return cached_tools if isinstance(cached_tools, list) else []


def _build_registered_mcp_tool(
    server_name: str,
    server_config: dict[str, Any],
    tool: dict[str, Any],
    lazy_ref: _mcp_mod.LazyClientRef,
    is_deferred: bool,
    cache_path: Path,
) -> ToolSpec:
    name = tool["name"]
    risk = resolve_mcp_tool_risk(server_config, name)
    tool_schema, tool_description = _mcp_tool_schema_and_description(
        server_name, tool, is_deferred
    )
    return ToolSpec(
        name=f"mcp__{server_name}__{name}",
        description=tool_description,
        input_hint=_mcp_tool_input_hint(tool.get("inputSchema", {})),
        handler=_make_handler(lazy_ref, name, is_deferred, server_name, cache_path),
        risk=risk,
        schema=tool_schema,
        read_only=(risk == "low"),
        group="mcp",
    )


def _mcp_tool_schema_and_description(
    server_name: str,
    tool: dict[str, Any],
    is_deferred: bool,
) -> tuple[dict[str, Any], str]:
    desc = tool.get("description", "")
    if is_deferred:
        return (
            {"type": "object", "properties": {}, "additionalProperties": True},
            f"[Deferred] {desc} Parameters unknown until searched. Call mcp_tool_search first to retrieve the required schema before invoking this tool. [mcp: {server_name}]",
        )
    return (
        tool.get("inputSchema", {}),
        f"{desc} [mcp: {server_name}]" if desc else f"[mcp: {server_name}]",
    )


def _mcp_tool_input_hint(input_schema: dict[str, Any]) -> str:
    props = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    hints = [
        f"{p_name}: {p_info.get('type', 'any')}{' (required)' if p_name in required else ''}"
        for p_name, p_info in props.items()
    ]
    return ", ".join(hints) if hints else "no arguments"
