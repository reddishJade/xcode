"""MCP 工具注册与配置。

Step 9 canonicalized MCP integration:
- Config schema validation with McpServerConfig
- Single canonical config path (.local/mcp_config.json only)
- overrides → skip server with warning
- enabled/timeout config support
- sanitize() naming + collision detection
- ActionExtractor MCP branch support via ToolSpec metadata
- isError structured handling
- stderr redaction/truncation
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xcode.harness.skills import ToolInput, ToolSpec

from . import client as _mcp_mod

_log = logging.getLogger(__name__)


# ── 数据模型 ──


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    command: tuple[str, ...]
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    enabled: bool = True
    timeout: float | None = None


@dataclass(frozen=True)
class McpToolMetadata:
    """Metadata preserved on ToolSpec for MCP tools."""

    server_name: str
    server_slug: str
    tool_name: str
    tool_slug: str
    host_tool_id: str


# ── 名称清理 ──


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _warn(msg: str) -> None:
    _log.warning("MCP: %s", msg)


# ── 配置校验 ──


def _validate_server_config(name: str, raw: object) -> McpServerConfig | None:
    if not isinstance(raw, dict):
        _warn(f"server {name!r} config is not a dict; skipped")
        return None

    if "overrides" in raw:
        _warn(f"server {name!r} uses unsupported overrides; skipped")
        return None

    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        _warn(f"server {name!r} has no valid command; skipped")
        return None

    args_list: tuple[str, ...] = ()
    raw_args = raw.get("args", [])
    if isinstance(raw_args, list):
        args_list = tuple(str(a) for a in raw_args if isinstance(a, str))

    env = raw.get("env")
    if env is not None and not isinstance(env, dict):
        env = None

    enabled = bool(raw.get("enabled", True))
    timeout = raw.get("timeout")
    if timeout is not None and not isinstance(timeout, (int, float)):
        timeout = None

    slug = _sanitize(name)
    if not slug:
        _warn(f"server {name!r} produces empty slug after sanitize; skipped")
        return None

    return McpServerConfig(
        name=name,
        command=(command,),
        args=args_list,
        env=env,
        enabled=enabled,
        timeout=float(timeout) if timeout is not None else None,
    )


# ── 配置与缓存路径 ──


CANONICAL_CONFIG_PATH = ".local" / Path("mcp_config.json")


def _mcp_config_path(project_root: Path) -> Path | None:
    canonical = project_root / ".local" / "mcp_config.json"
    if canonical.exists():
        return canonical
    legacy = project_root / "mcp_config.json"
    if legacy.exists():
        _warn(
            "found mcp_config.json at project root; canonical location is "
            ".local/mcp_config.json. Move the file and remove the root copy."
        )
        return None
    return None


def _cache_path(project_root: Path) -> Path:
    return project_root / ".local" / "mcp_cache.json"


# ── 缓存工具 ──


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            _log.warning("failed to load MCP cache from %s", cache_path, exc_info=True)
    return {"servers": {}}


def _save_cache(cache_path: Path, cache_data: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── 配置哈希 ──


def compute_config_hash(server_config: dict[str, Any]) -> str:
    data = {
        "command": server_config.get("command"),
        "args": server_config.get("args", []),
        "env": server_config.get("env", {}),
    }
    timeout = server_config.get("timeout")
    if timeout is not None:
        data["timeout"] = timeout
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_metadata(client: _mcp_mod.McpClient) -> dict[str, Any]:
    """提取可用于验证缓存来源的协商元数据。"""
    protocol_version = client.protocol_version
    if protocol_version is None:
        raise RuntimeError("MCP client has no negotiated protocol version")
    return {
        "protocol_version": protocol_version,
        "server_info": client.server_info,
    }


def _compatible_cached_tools(
    cached_entry: object,
    config_hash: str,
) -> list[dict[str, Any]] | None:
    """返回来源可验证且配置未变化的缓存工具列表。"""
    if not isinstance(cached_entry, dict):
        return None
    if cached_entry.get("config_hash") != config_hash:
        return None
    if cached_entry.get("protocol_version") not in (
        _mcp_mod.SUPPORTED_PROTOCOL_VERSIONS
    ):
        return None
    server_info = cached_entry.get("server_info")
    if not isinstance(server_info, dict):
        return None
    server_name = server_info.get("name")
    server_version = server_info.get("version")
    if not isinstance(server_name, str) or not server_name.strip():
        return None
    if not isinstance(server_version, str) or not server_version.strip():
        return None
    tools = cached_entry.get("tools")
    return tools if isinstance(tools, list) else None


# ── 引导工具 ──


def build_fetch_tools_tool(
    project_root: Path,
    server_name: str,
    validated: McpServerConfig,
) -> ToolSpec:
    """创建用于冷启动延迟加载服务器并拉取工具列表的引导工具。"""

    def handler(_args: ToolInput) -> str:
        config_hash = compute_config_hash(
            {
                "command": validated.command[0],
                "args": list(validated.args),
                "env": validated.env,
                "timeout": validated.timeout,
            }
        )
        client: _mcp_mod.McpClient | None = None
        try:
            command = [validated.command[0]] + list(validated.args)
            client = _mcp_mod.McpClient(
                command, validated.env, timeout=validated.timeout
            )
            client.start()
            tools_list = client.list_tools()
            cache_metadata = _cache_metadata(client)

            cache_path = _cache_path(project_root)
            cache_data = _load_cache(cache_path)
            cache_data.setdefault("servers", {})[server_name] = {
                "config_hash": config_hash,
                "tools": tools_list,
                **cache_metadata,
            }
            _save_cache(cache_path, cache_data)
            return (
                f"Successfully fetched {len(tools_list)} tools from server "
                f"'{server_name}' and populated the cache. Please use "
                f"mcp_tool_search to retrieve their schemas now!"
            )
        except Exception as e:
            redacted = str(e)
            return f"Error fetching tools from server '{server_name}': {redacted}"
        finally:
            if client is not None:
                client.stop()

    slug = _sanitize(server_name)
    return ToolSpec(
        name=f"mcp__{slug}__fetch_tools",
        description=(
            f"Bootstrap and fetch tool list for deferred server "
            f"'{server_name}' to populate the cache."
        ),
        input_hint="{}",
        handler=handler,
        schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        read_only=True,
        group="mcp",
    )


def _format_tool_match(server_name: str, server_slug: str, tool: dict[str, Any]) -> str:
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
        f"- **Tool Name**: `mcp__{server_slug}__{_sanitize(name)}`\n"
        f"  **Description**: {desc}\n"
        f"  **Schema / Parameters**:\n{params_str}"
    )


def build_mcp_tool_search(
    project_root: Path,
    deferred_servers: set[str],
    slug_map: dict[str, str],
) -> ToolSpec:
    """创建用于搜索和获取延迟加载工具完整参数 Schema 的工具。"""

    def handler(args: ToolInput) -> str:
        query = str(args.get("query", "")).strip().lower()
        if not query:
            return "Please provide a query to search."

        cache_data = _load_cache(_cache_path(project_root))
        results = []

        for server_name in deferred_servers:
            server_slug = slug_map.get(server_name, _sanitize(server_name))
            server_entry = cache_data.get("servers", {}).get(server_name, {})
            tools = server_entry.get("tools", [])
            if not tools:
                results.append(
                    f"### Server '{server_name}'\n"
                    f"Schema not yet loaded. Please invoke the fetch_tools tool "
                    f"(e.g. `mcp__{server_slug}__fetch_tools`) "
                    f"to trigger schema fetch."
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
                    results.append(_format_tool_match(server_name, server_slug, tool))

        return (
            "\n\n".join(results)
            if results
            else f"No tools found matching query: {query}"
        )

    return ToolSpec(
        name="mcp_tool_search",
        description=(
            "Search and retrieve full schema, descriptions, and parameters "
            "for deferred MCP tools."
        ),
        input_hint='{"query": "keyword_or_all"}',
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keyword to search within tool names or descriptions, "
                        "or 'all' to list all tools."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        read_only=True,
        group="mcp",
    )


# ── 工具执行包装 ──


def _redact_and_truncate(text: str, max_len: int = 200) -> str:
    return _mcp_mod.truncate_redact(text, max_len=max_len)


def _make_handler(
    ref: _mcp_mod.LazyClientRef,
    original_tool_name: str,
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
                        if cached_tool["name"] == original_tool_name
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
                            "Missing required parameters for deferred "
                            f"tool {original_tool_name}: "
                            f"{', '.join(missing)}"
                        )
            except Exception as validation_error:
                if isinstance(validation_error, ValueError):
                    raise

        client_instance = ref.get_or_create()
        response = client_instance.call_tool(
            original_tool_name, args, timeout=ref.config.get("timeout")
        )
        if "content" in response:
            text_parts: list[str] = []
            non_text_hint: str | None = None
            for block in response.get("content", []):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                else:
                    if non_text_hint is None:
                        non_text_hint = f"[unsupported content type: {block_type}]"
            content_str = "\n".join(text_parts)
            if non_text_hint:
                if content_str:
                    content_str += f"\n{non_text_hint}"
                else:
                    content_str = non_text_hint
            if response.get("isError", False):
                redacted = _redact_and_truncate(content_str, max_len=200)
                raise _MCPToolError(redacted, is_error=True)
            return content_str
        return str(response)

    return handler


class _MCPToolError(Exception):
    """Structured MCP tool error with isError flag."""

    def __init__(self, message: str, is_error: bool = True) -> None:
        super().__init__(message)
        self.is_error = is_error


# ── 碰撞检测 ──


def _detect_collisions(
    tools_by_server: dict[str, list[dict[str, Any]]],
    validated_servers: dict[str, McpServerConfig],
) -> set[str]:
    """Detect host_tool_id collisions and return set of (server, tool) to disable.

    Returns set of "server_name:original_tool_name" strings for all
    tools involved in collisions.
    """
    host_ids: dict[str, list[tuple[str, str]]] = {}
    for server_name, tool_list in tools_by_server.items():
        server_slug = _sanitize(server_name)
        for tool in tool_list:
            tool_slug = _sanitize(tool["name"])
            host_id = f"mcp__{server_slug}__{tool_slug}"
            host_ids.setdefault(host_id, []).append((server_name, tool["name"]))

    disabled: set[str] = set()
    for host_id, entries in host_ids.items():
        if len(entries) > 1:
            identities = "; ".join(f"{s}:{t}" for s, t in entries)
            _warn(
                f"tool ID collision: {host_id!r} produced by "
                f"{identities}; all conflicting tools disabled"
            )
            for s, t in entries:
                disabled.add(f"{s}:{t}")

    return disabled


# ── 主入口 ──


def build_mcp_tools(project_root: Path) -> tuple[ToolSpec, ...]:
    """根据配置文件加载和构建所有 MCP 工具。"""
    config_path = _mcp_config_path(project_root)
    if config_path is None:
        return ()

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        _warn(f"failed to load MCP config {config_path}: {e}")
        return ()

    mcp_servers = config_data.get("mcpServers", {})
    if not mcp_servers:
        return ()

    # Validate all servers first
    validated_servers: dict[str, McpServerConfig] = {}
    for server_name, raw in mcp_servers.items():
        validated = _validate_server_config(server_name, raw)
        if validated is not None:
            validated_servers[server_name] = validated

    tools_to_register: list[ToolSpec] = []
    deferred_servers: set[str] = set()
    slug_map: dict[str, str] = {}
    all_tools_by_server: dict[str, list[dict[str, Any]]] = {}

    cache_path = _cache_path(project_root)
    cache_data = _load_cache(cache_path)

    for server_name, validated in validated_servers.items():
        if not validated.enabled:
            _log.info("MCP server %r is disabled; skipping", server_name)
            continue

        is_deferred = bool(mcp_servers.get(server_name, {}).get("defer_loading", False))
        if is_deferred:
            deferred_servers.add(server_name)
            slug_map[server_name] = _sanitize(server_name)

        tools_list = _tools_for_server(
            project_root,
            cache_path,
            cache_data,
            server_name,
            mcp_servers.get(server_name, {}),
            validated,
            is_deferred,
            tools_to_register,
        )
        if tools_list:
            all_tools_by_server[server_name] = tools_list

    # Collision detection
    disabled = _detect_collisions(all_tools_by_server, validated_servers)

    # Build ToolSpecs for non-colliding tools
    for server_name, tools_list in all_tools_by_server.items():
        validated = validated_servers[server_name]
        raw_config = mcp_servers.get(server_name, {})
        lazy_ref = _mcp_mod.LazyClientRef(server_name, raw_config)
        server_slug = _sanitize(server_name)

        for tool in tools_list:
            tool_key = f"{server_name}:{tool['name']}"
            if tool_key in disabled:
                continue

            tool_slug = _sanitize(tool["name"])
            host_id = f"mcp__{server_slug}__{tool_slug}"
            metadata = McpToolMetadata(
                server_name=server_name,
                server_slug=server_slug,
                tool_name=tool["name"],
                tool_slug=tool_slug,
                host_tool_id=host_id,
            )

            tools_to_register.append(
                _build_registered_mcp_tool(
                    validated,
                    tool,
                    lazy_ref,
                    is_deferred=(server_name in deferred_servers),
                    cache_path=cache_path,
                    metadata=metadata,
                )
            )

    if deferred_servers:
        tools_to_register.append(
            build_mcp_tool_search(project_root, deferred_servers, slug_map)
        )

    return tuple(tools_to_register)


def _tools_for_server(
    project_root: Path,
    cache_path: Path,
    cache_data: dict[str, Any],
    server_name: str,
    raw_config: dict[str, Any],
    validated: McpServerConfig,
    is_deferred: bool,
    tools_to_register: list[ToolSpec],
) -> list[dict[str, Any]]:
    config_hash = compute_config_hash(raw_config)
    cached_entry = cache_data.get("servers", {}).get(server_name, {})
    cached_tools = _compatible_cached_tools(cached_entry, config_hash)
    if cached_tools is not None:
        return cached_tools

    if is_deferred:
        tools_to_register.append(
            build_fetch_tools_tool(project_root, server_name, validated)
        )
        return []

    return _query_server_tools(
        cache_path,
        cache_data,
        server_name,
        raw_config,
        validated,
        config_hash,
        cached_entry,
    )


def _query_server_tools(
    cache_path: Path,
    cache_data: dict[str, Any],
    server_name: str,
    raw_config: dict[str, Any],
    validated: McpServerConfig,
    config_hash: str,
    cached_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    client: _mcp_mod.McpClient | None = None
    try:
        command = [validated.command[0]] + list(validated.args)
        client = _mcp_mod.McpClient(command, validated.env, timeout=validated.timeout)
        client.start()
        tools_list = client.list_tools()
        cache_metadata = _cache_metadata(client)

        cache_data.setdefault("servers", {})[server_name] = {
            "config_hash": config_hash,
            "tools": tools_list,
            **cache_metadata,
        }
        _save_cache(cache_path, cache_data)
        return tools_list
    except Exception as e:
        redacted = _redact_and_truncate(str(e), max_len=200)
        _warn(f"error querying tools from MCP server {server_name!r}: {redacted}")
        cached_tools = _compatible_cached_tools(cached_entry, config_hash)
        return cached_tools or []
    finally:
        if client is not None:
            client.stop()


def _build_registered_mcp_tool(
    validated: McpServerConfig,
    tool: dict[str, Any],
    lazy_ref: _mcp_mod.LazyClientRef,
    is_deferred: bool,
    cache_path: Path,
    metadata: McpToolMetadata,
) -> ToolSpec:
    tool_schema, tool_description = _mcp_tool_schema_and_description(
        validated.name, tool, is_deferred
    )
    return ToolSpec(
        name=metadata.host_tool_id,
        description=tool_description,
        input_hint=_mcp_tool_input_hint(tool.get("inputSchema", {})),
        handler=_make_handler(
            lazy_ref,
            tool["name"],
            is_deferred,
            validated.name,
            cache_path,
        ),
        schema=tool_schema,
        group="mcp",
        builtin={
            "mcp_metadata": {
                "server": metadata.server_name,
                "server_slug": metadata.server_slug,
                "tool": metadata.tool_name,
                "tool_slug": metadata.tool_slug,
            }
        },
    )


def _mcp_tool_schema_and_description(
    server_name: str,
    tool: dict[str, Any],
    is_deferred: bool,
) -> tuple[dict[str, Any], str]:
    desc = tool.get("description", "")
    if is_deferred:
        return (
            {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
            (
                f"[Deferred] {desc} Parameters unknown until searched. "
                f"Call mcp_tool_search first to retrieve the required "
                f"schema before invoking this tool. [mcp: {server_name}]"
            ),
        )
    return (
        tool.get("inputSchema", {}),
        f"{desc} [mcp: {server_name}]" if desc else f"[mcp: {server_name}]",
    )


def _mcp_tool_input_hint(input_schema: dict[str, Any]) -> str:
    props = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    hints = [
        f"{p_name}: {p_info.get('type', 'any')}"
        f"{' (required)' if p_name in required else ''}"
        for p_name, p_info in props.items()
    ]
    return ", ".join(hints) if hints else "no arguments"
