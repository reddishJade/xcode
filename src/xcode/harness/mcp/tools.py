"""MCP 工具注册与配置。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xcode.harness.skills import ToolInput, ToolSpec

from . import client as _mcp_mod
from .results import convert_mcp_tool_result

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


@dataclass(frozen=True)
class McpToolOverride:
    """宿主侧允许覆盖的 MCP 工具元数据。"""

    risk: str | None = None
    read_only: bool | None = None
    concurrency_safe: bool | None = None
    enabled: bool | None = None
    description: str | None = None


@dataclass(frozen=True)
class McpServerStatus:
    """供 REPL 与诊断读取的 MCP server 运行时状态。"""

    server_name: str
    state: str
    enabled: bool
    deferred: bool
    tool_count: int
    protocol_version: str | None = None
    server_info: dict[str, Any] | None = None
    last_error: str | None = None


class McpRuntimeRegistry:
    """协调 MCP 动态工具快照、订阅者和持久客户端生命周期。"""

    def __init__(self) -> None:
        """初始化空的运行时注册状态。"""
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[tuple[ToolSpec, ...]], None]] = []
        self._client_refs: dict[str, _mcp_mod.LazyClientRef] = {}
        self._reload_callback: Callable[[], tuple[ToolSpec, ...]] | None = None
        self._workspace_roots: tuple[Path, ...] = ()
        self._cancel_event: threading.Event | None = None
        self._raw_servers: dict[str, dict[str, Any]] = {}
        self._validated_servers: dict[str, McpServerConfig] = {}
        self._deferred_servers: set[str] = set()
        self._tool_counts: dict[str, int] = {}
        self._cache_metadata: dict[str, dict[str, Any]] = {}
        self._server_errors: dict[str, str] = {}

    def configure_runtime(
        self,
        *,
        workspace_roots: tuple[Path, ...],
        cancel_event: threading.Event | None,
    ) -> None:
        """配置与当前宿主运行绑定的共享上下文。"""
        with self._lock:
            self._workspace_roots = tuple(
                path.resolve() for path in workspace_roots if path.exists()
            )
            self._cancel_event = cancel_event

    def subscribe(
        self,
        callback: Callable[[tuple[ToolSpec, ...]], None],
    ) -> None:
        """注册工具快照更新回调。"""
        with self._lock:
            self._callbacks.append(callback)

    def publish(self, tools: tuple[ToolSpec, ...]) -> None:
        """向所有订阅者发布完整 MCP 工具快照。"""
        with self._lock:
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            callback(tools)

    def set_reload_callback(
        self,
        callback: Callable[[], tuple[ToolSpec, ...]],
    ) -> None:
        """注册手动 reload 入口。"""
        with self._lock:
            self._reload_callback = callback

    def reload(self) -> tuple[ToolSpec, ...]:
        """重新读取 MCP 配置并发布新的工具快照。"""
        with self._lock:
            callback = self._reload_callback
        if callback is None:
            return ()
        tools = callback()
        self.publish(tools)
        return tools

    def update_runtime_snapshot(
        self,
        *,
        raw_servers: dict[str, dict[str, Any]],
        validated_servers: dict[str, McpServerConfig],
        deferred_servers: set[str],
        tool_counts: dict[str, int],
        cache_metadata: dict[str, dict[str, Any]],
        server_errors: dict[str, str],
    ) -> None:
        """保存当前配置与发现结果，供状态接口读取。"""
        with self._lock:
            self._raw_servers = dict(raw_servers)
            self._validated_servers = dict(validated_servers)
            self._deferred_servers = set(deferred_servers)
            self._tool_counts = dict(tool_counts)
            self._cache_metadata = dict(cache_metadata)
            self._server_errors = dict(server_errors)

    def status_snapshot(self) -> tuple[McpServerStatus, ...]:
        """返回所有已配置 server 的只读运行时状态。"""
        with self._lock:
            raw_servers = dict(self._raw_servers)
            validated_servers = dict(self._validated_servers)
            deferred_servers = set(self._deferred_servers)
            tool_counts = dict(self._tool_counts)
            cache_metadata = dict(self._cache_metadata)
            server_errors = dict(self._server_errors)
            client_refs = dict(self._client_refs)
        statuses: list[McpServerStatus] = []
        for server_name, raw in raw_servers.items():
            validated = validated_servers.get(server_name)
            enabled = bool(raw.get("enabled", True))
            last_error = server_errors.get(server_name)
            ref = client_refs.get(server_name)
            protocol_version = None
            server_info = None
            state = "configured"
            if (
                ref is not None
                and ref.client is not None
                and ref.client.status == "connected"
            ):
                state = "connected"
                protocol_version = ref.client.protocol_version
                server_info = ref.client.server_info
                last_error = ref.last_error or last_error
            else:
                cached = cache_metadata.get(server_name, {})
                protocol_version = cached.get("protocol_version")
                server_info = cached.get("server_info")
                if last_error or (ref is not None and ref.last_error):
                    state = "failed"
                    if ref is not None and ref.last_error:
                        last_error = ref.last_error
                elif (
                    server_name in deferred_servers
                    and tool_counts.get(server_name, 0) == 0
                ):
                    state = "deferred"
            if validated is None:
                state = "failed"
            statuses.append(
                McpServerStatus(
                    server_name=server_name,
                    state=state,
                    enabled=enabled,
                    deferred=server_name in deferred_servers,
                    tool_count=tool_counts.get(server_name, 0),
                    protocol_version=protocol_version,
                    server_info=server_info,
                    last_error=last_error,
                )
            )
        return tuple(sorted(statuses, key=lambda item: item.server_name))

    @property
    def workspace_roots(self) -> tuple[Path, ...]:
        with self._lock:
            return self._workspace_roots

    @property
    def cancel_event(self) -> threading.Event | None:
        with self._lock:
            return self._cancel_event

    def update_workspace_roots(self, workspace_roots: tuple[Path, ...]) -> None:
        """更新 roots 并通知所有已连接 server。"""
        normalized = tuple(path.resolve() for path in workspace_roots if path.exists())
        with self._lock:
            self._workspace_roots = normalized
            refs = tuple(self._client_refs.values())
        for ref in refs:
            ref.update_workspace_roots(normalized)

    def track_client_ref(self, server_name: str, ref: _mcp_mod.LazyClientRef) -> None:
        """登记需要随应用关闭的延迟客户端。"""
        with self._lock:
            self._client_refs[server_name] = ref

    def drop_client_ref(self, server_name: str) -> None:
        """关闭并移除被删除或替换的 server client。"""
        with self._lock:
            ref = self._client_refs.pop(server_name, None)
        if ref is not None:
            ref.stop()

    def close(self) -> None:
        """关闭所有已启动的 MCP 持久客户端。"""
        with self._lock:
            refs = tuple(self._client_refs.values())
            self._client_refs.clear()
            self._callbacks.clear()
        for ref in refs:
            ref.stop()


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


def _parse_overrides(raw: object) -> dict[str, McpToolOverride]:
    """解析宿主侧 tool overrides，仅接受显式白名单字段。"""
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, McpToolOverride] = {}
    for tool_name, tool_override in raw.items():
        if not isinstance(tool_name, str) or not isinstance(tool_override, dict):
            continue
        enabled = tool_override.get("enabled")
        read_only = tool_override.get("read_only")
        concurrency_safe = tool_override.get("concurrency_safe")
        description = tool_override.get("description")
        risk = tool_override.get("risk")
        overrides[tool_name] = McpToolOverride(
            risk=risk if isinstance(risk, str) else None,
            read_only=read_only if isinstance(read_only, bool) else None,
            concurrency_safe=(
                concurrency_safe if isinstance(concurrency_safe, bool) else None
            ),
            enabled=enabled if isinstance(enabled, bool) else None,
            description=description if isinstance(description, str) else None,
        )
    return overrides


def _tool_override(
    raw_server: dict[str, Any],
    tool_name: str,
) -> McpToolOverride:
    overrides = _parse_overrides(raw_server.get("overrides"))
    default_override = overrides.get("*", McpToolOverride())
    specific_override = overrides.get(tool_name, McpToolOverride())
    return McpToolOverride(
        risk=specific_override.risk or default_override.risk,
        read_only=(
            specific_override.read_only
            if specific_override.read_only is not None
            else default_override.read_only
        ),
        concurrency_safe=(
            specific_override.concurrency_safe
            if specific_override.concurrency_safe is not None
            else default_override.concurrency_safe
        ),
        enabled=(
            specific_override.enabled
            if specific_override.enabled is not None
            else default_override.enabled
        ),
        description=specific_override.description or default_override.description,
    )


# ── 配置与缓存路径 ──


CANONICAL_CONFIG_PATH = ".local" / Path("mcp_config.json")


def _mcp_config_path(project_root: Path) -> Path | None:
    canonical = project_root / CANONICAL_CONFIG_PATH
    if canonical.exists():
        return canonical
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
    output_schema: object,
    runtime_registry: McpRuntimeRegistry,
) -> Any:
    def handler(
        args: ToolInput,
        on_update: Callable[[str], None] | None = None,
    ) -> str:
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
        progress_updates: list[dict[str, object]] = []

        def record_progress(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            update = {
                "progress": progress,
                "total": total,
                "message": message,
            }
            progress_updates.append(update)
            if on_update is not None:
                on_update(_format_progress_update(progress, total, message))

        response = client_instance.call_tool(
            original_tool_name,
            args,
            timeout=ref.config.get("timeout"),
            progress_callback=record_progress,
            cancel_event=runtime_registry.cancel_event,
        )
        result = convert_mcp_tool_result(response, output_schema)
        if not progress_updates:
            return result
        metadata = getattr(result, "metadata", {})
        merged = dict(metadata)
        merged["mcp_progress"] = progress_updates
        return type(result)(str(result), metadata=merged, is_error=result.is_error)

    return handler


def _format_progress_update(
    progress: float,
    total: float | None,
    message: str | None,
) -> str:
    """将 MCP progress notification 转为稳定的工具更新文本。"""
    position = f"{progress:g}"
    if total is not None:
        position = f"{position}/{total:g}"
    suffix = f" {message}" if message else ""
    return f"MCP progress {position}{suffix}"


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


def _prune_removed_or_replaced_servers(
    project_root: Path,
    runtime_registry: McpRuntimeRegistry,
    raw_servers: dict[str, dict[str, Any]],
) -> None:
    """关闭被删除或配置已变化的 server，并清理失效缓存。"""
    previous = runtime_registry._raw_servers
    removed_or_changed = {
        server_name
        for server_name, old_raw in previous.items()
        if server_name not in raw_servers
        or compute_config_hash(old_raw) != compute_config_hash(raw_servers[server_name])
    }
    if not removed_or_changed:
        return
    for server_name in removed_or_changed:
        runtime_registry.drop_client_ref(server_name)
    cache_path = _cache_path(project_root)
    cache_data = _load_cache(cache_path)
    servers = cache_data.get("servers")
    if not isinstance(servers, dict):
        return
    changed = False
    for server_name in removed_or_changed:
        if server_name in servers:
            servers.pop(server_name, None)
            changed = True
    if changed:
        _save_cache(cache_path, cache_data)


# ── 主入口 ──


def build_mcp_tools(
    project_root: Path,
    runtime_registry: McpRuntimeRegistry | None = None,
) -> tuple[ToolSpec, ...]:
    """根据配置文件加载和构建所有 MCP 工具。"""
    runtime_registry = runtime_registry or McpRuntimeRegistry()
    runtime_registry.set_reload_callback(
        lambda: _load_mcp_tools_snapshot(project_root, runtime_registry)
    )
    return _load_mcp_tools_snapshot(project_root, runtime_registry)


def _load_mcp_tools_snapshot(
    project_root: Path,
    runtime_registry: McpRuntimeRegistry,
) -> tuple[ToolSpec, ...]:
    """读取当前配置并构建完整 MCP 工具快照。"""
    config_path = _mcp_config_path(project_root)
    if config_path is None:
        runtime_registry.update_runtime_snapshot(
            raw_servers={},
            validated_servers={},
            deferred_servers=set(),
            tool_counts={},
            cache_metadata={},
            server_errors={},
        )
        return ()

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _warn(f"failed to load MCP config {config_path}: {exc}")
        runtime_registry.update_runtime_snapshot(
            raw_servers={},
            validated_servers={},
            deferred_servers=set(),
            tool_counts={},
            cache_metadata={},
            server_errors={"<config>": _redact_and_truncate(str(exc))},
        )
        return ()

    mcp_servers = {
        name: raw
        for name, raw in config_data.get("mcpServers", {}).items()
        if isinstance(name, str) and isinstance(raw, dict)
    }
    validated_servers: dict[str, McpServerConfig] = {}
    server_errors: dict[str, str] = {}
    for server_name, raw in mcp_servers.items():
        validated = _validate_server_config(server_name, raw)
        if validated is not None:
            validated_servers[server_name] = validated
        else:
            server_errors[server_name] = "invalid config"

    _prune_removed_or_replaced_servers(project_root, runtime_registry, mcp_servers)

    tools_to_register: list[ToolSpec] = []
    deferred_servers: set[str] = set()
    slug_map: dict[str, str] = {}
    all_tools_by_server: dict[str, list[dict[str, Any]]] = {}
    cache_metadata_by_server: dict[str, dict[str, Any]] = {}
    refresh_lock = threading.Lock()
    cache_path = _cache_path(project_root)
    cache_data = _load_cache(cache_path)

    for server_name, validated in validated_servers.items():
        is_deferred = bool(mcp_servers.get(server_name, {}).get("defer_loading", False))
        if is_deferred:
            deferred_servers.add(server_name)
            slug_map[server_name] = _sanitize(server_name)
        if not validated.enabled:
            continue
        tools_list = _tools_for_server(
            project_root,
            cache_path,
            cache_data,
            server_name,
            mcp_servers[server_name],
            validated,
            is_deferred,
            tools_to_register,
            runtime_registry.workspace_roots,
            cache_metadata_by_server,
            server_errors,
        )
        if tools_list:
            all_tools_by_server[server_name] = tools_list

    bootstrap_tools = tuple(tools_to_register)
    search_tools = (
        (build_mcp_tool_search(project_root, deferred_servers, slug_map),)
        if deferred_servers
        else ()
    )

    def refresh_server_tools(server_name: str, client: _mcp_mod.McpClient) -> None:
        validated = validated_servers[server_name]
        tools_list = client.list_tools(timeout=validated.timeout)
        cache_metadata = _cache_metadata(client)
        config_hash = compute_config_hash(mcp_servers[server_name])
        with refresh_lock:
            current_cache = _load_cache(cache_path)
            current_cache.setdefault("servers", {})[server_name] = {
                "config_hash": config_hash,
                "tools": tools_list,
                **cache_metadata,
            }
            _save_cache(cache_path, current_cache)
            all_tools_by_server[server_name] = tools_list
            cache_metadata_by_server[server_name] = cache_metadata
            refreshed = _build_runtime_mcp_tools(
                all_tools_by_server=all_tools_by_server,
                validated_servers=validated_servers,
                raw_servers=mcp_servers,
                deferred_servers=deferred_servers,
                cache_path=cache_path,
                runtime_registry=runtime_registry,
                refresh_server_tools=refresh_server_tools,
            )
            runtime_registry.update_runtime_snapshot(
                raw_servers=mcp_servers,
                validated_servers=validated_servers,
                deferred_servers=deferred_servers,
                tool_counts={
                    name: len(tools) for name, tools in all_tools_by_server.items()
                },
                cache_metadata=cache_metadata_by_server,
                server_errors=server_errors,
            )
        runtime_registry.publish(bootstrap_tools + refreshed + search_tools)

    registered_tools = _build_runtime_mcp_tools(
        all_tools_by_server=all_tools_by_server,
        validated_servers=validated_servers,
        raw_servers=mcp_servers,
        deferred_servers=deferred_servers,
        cache_path=cache_path,
        runtime_registry=runtime_registry,
        refresh_server_tools=refresh_server_tools,
    )
    runtime_registry.update_runtime_snapshot(
        raw_servers=mcp_servers,
        validated_servers=validated_servers,
        deferred_servers=deferred_servers,
        tool_counts={name: len(tools) for name, tools in all_tools_by_server.items()},
        cache_metadata=cache_metadata_by_server,
        server_errors=server_errors,
    )
    return bootstrap_tools + registered_tools + search_tools


def _build_runtime_mcp_tools(
    *,
    all_tools_by_server: dict[str, list[dict[str, Any]]],
    validated_servers: dict[str, McpServerConfig],
    raw_servers: dict[str, dict[str, Any]],
    deferred_servers: set[str],
    cache_path: Path,
    runtime_registry: McpRuntimeRegistry,
    refresh_server_tools: Callable[[str, _mcp_mod.McpClient], None],
) -> tuple[ToolSpec, ...]:
    """根据最新服务器 schema 构建无冲突的运行时 MCP 工具。"""
    disabled = _detect_collisions(all_tools_by_server, validated_servers)
    registered: list[ToolSpec] = []

    for server_name, tools_list in all_tools_by_server.items():
        validated = validated_servers[server_name]
        lazy_ref = runtime_registry._client_refs.get(server_name)
        if lazy_ref is None:

            def on_tools_changed(
                client: _mcp_mod.McpClient,
                current_server: str = server_name,
            ) -> None:
                """将客户端通知转交给对应服务器的注册表刷新。"""
                refresh_server_tools(current_server, client)

            lazy_ref = _mcp_mod.LazyClientRef(
                server_name,
                raw_servers[server_name],
                tools_changed_callback=on_tools_changed,
                workspace_roots=runtime_registry.workspace_roots,
            )
            runtime_registry.track_client_ref(server_name, lazy_ref)

        server_slug = _sanitize(server_name)
        for tool in tools_list:
            tool_key = f"{server_name}:{tool['name']}"
            if tool_key in disabled:
                continue
            tool_override = _tool_override(raw_servers[server_name], tool["name"])
            if tool_override.enabled is False:
                continue

            tool_slug = _sanitize(tool["name"])
            metadata = McpToolMetadata(
                server_name=server_name,
                server_slug=server_slug,
                tool_name=tool["name"],
                tool_slug=tool_slug,
                host_tool_id=f"mcp__{server_slug}__{tool_slug}",
            )
            registered.append(
                _build_registered_mcp_tool(
                    validated,
                    tool,
                    lazy_ref,
                    is_deferred=(server_name in deferred_servers),
                    cache_path=cache_path,
                    metadata=metadata,
                    tool_override=tool_override,
                    runtime_registry=runtime_registry,
                )
            )

    return tuple(registered)


def _tools_for_server(
    project_root: Path,
    cache_path: Path,
    cache_data: dict[str, Any],
    server_name: str,
    raw_config: dict[str, Any],
    validated: McpServerConfig,
    is_deferred: bool,
    tools_to_register: list[ToolSpec],
    workspace_roots: tuple[Path, ...],
    cache_metadata_by_server: dict[str, dict[str, Any]],
    server_errors: dict[str, str],
) -> list[dict[str, Any]]:
    config_hash = compute_config_hash(raw_config)
    cached_entry = cache_data.get("servers", {}).get(server_name, {})
    cached_tools = _compatible_cached_tools(cached_entry, config_hash)
    if cached_tools is not None:
        cache_metadata_by_server[server_name] = {
            "protocol_version": cached_entry.get("protocol_version"),
            "server_info": cached_entry.get("server_info"),
        }
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
        workspace_roots,
        cache_metadata_by_server,
        server_errors,
    )


def _query_server_tools(
    cache_path: Path,
    cache_data: dict[str, Any],
    server_name: str,
    raw_config: dict[str, Any],
    validated: McpServerConfig,
    config_hash: str,
    cached_entry: dict[str, Any],
    workspace_roots: tuple[Path, ...],
    cache_metadata_by_server: dict[str, dict[str, Any]],
    server_errors: dict[str, str],
) -> list[dict[str, Any]]:
    client: _mcp_mod.McpClient | None = None
    try:
        command = [validated.command[0]] + list(validated.args)
        client = _mcp_mod.McpClient(
            command,
            validated.env,
            timeout=validated.timeout,
            workspace_roots=workspace_roots,
        )
        client.start()
        tools_list = client.list_tools()
        cache_metadata = _cache_metadata(client)
        cache_metadata_by_server[server_name] = cache_metadata

        cache_data.setdefault("servers", {})[server_name] = {
            "config_hash": config_hash,
            "tools": tools_list,
            **cache_metadata,
        }
        _save_cache(cache_path, cache_data)
        return tools_list
    except Exception as e:
        redacted = _redact_and_truncate(str(e), max_len=200)
        server_errors[server_name] = redacted
        _warn(f"error querying tools from MCP server {server_name!r}: {redacted}")
        cached_tools = _compatible_cached_tools(cached_entry, config_hash)
        if cached_tools is not None:
            cache_metadata_by_server[server_name] = {
                "protocol_version": cached_entry.get("protocol_version"),
                "server_info": cached_entry.get("server_info"),
            }
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
    tool_override: McpToolOverride,
    runtime_registry: McpRuntimeRegistry,
) -> ToolSpec:
    tool_schema, tool_description = _mcp_tool_schema_and_description(
        validated.name, tool, is_deferred
    )
    annotations = tool.get("annotations")
    read_only_hint = isinstance(annotations, dict) and bool(
        annotations.get("readOnlyHint")
    )
    read_only = (
        tool_override.read_only
        if tool_override.read_only is not None
        else read_only_hint
    )
    concurrency_safe = bool(tool_override.concurrency_safe)
    description = tool_override.description or tool_description
    mcp_handler = _make_handler(
        lazy_ref,
        tool["name"],
        is_deferred,
        validated.name,
        cache_path,
        tool.get("outputSchema"),
        runtime_registry,
    )
    return ToolSpec(
        name=metadata.host_tool_id,
        description=description,
        input_hint=_mcp_tool_input_hint(tool.get("inputSchema", {})),
        handler=lambda args: mcp_handler(args),
        schema=tool_schema,
        group="mcp",
        read_only=read_only,
        concurrency_safe=concurrency_safe,
        builtin={
            "mcp_metadata": {
                "server": metadata.server_name,
                "server_slug": metadata.server_slug,
                "tool": metadata.tool_name,
                "tool_slug": metadata.tool_slug,
                "outputSchema": tool.get("outputSchema"),
                "annotations": tool.get("annotations"),
                "risk": tool_override.risk,
            }
        },
        streaming_handler=mcp_handler,
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
