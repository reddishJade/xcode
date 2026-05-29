from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO

from xcode.harness.skills import ToolSpec, parse_tool_input


class McpClient:
    """Standard Synchronous Model Context Protocol (MCP) Client over Stdio."""

    def __init__(self, command: list[str], env: dict[str, str] | None = None) -> None:
        self.command = command
        # Inherit system environment variables
        self.env = os.environ.copy()
        if env:
            self.env.update(env)
        self.process: subprocess.Popen | None = None
        self.next_id = 1
        self._lock = threading.Lock()
        self._pending_responses: dict[int | str, dict[str, Any]] = {}
        self._read_thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self.process is not None:
            return
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
            )
        except OSError as e:
            raise RuntimeError(f"Failed to start MCP server {self.command}: {e}")

        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        _active_clients.append(self)

        # 1. Initialize Handshake
        try:
            self.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "xcode-client", "version": "0.1.0"},
                },
            )
            # 2. Initialized Notification
            self.send_notification("notifications/initialized")
        except Exception as e:
            self.stop()
            raise RuntimeError(f"MCP handshake failed: {e}")

    def _read_loop(self) -> None:
        while self._running and self.process and self.process.stdout:
            msg = _read_jsonrpc_message(self.process.stdout)
            if msg is None:
                break
            if isinstance(msg, dict) and "id" in msg:
                msg_id = msg["id"]
                with self._lock:
                    self._pending_responses[msg_id] = msg

    def send_request(
        self, method: str, params: dict, timeout: float = 10.0
    ) -> dict[str, Any]:
        if not self._running or not self.process:
            raise RuntimeError("MCP client is not running.")

        with self._lock:
            req_id = self.next_id
            self.next_id += 1

        req = {
            "jsonrpc": "2.0",
            "method": method,
            "id": req_id,
            "params": params,
        }

        payload = _encode_jsonrpc_message(req)
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"Failed to write request to MCP server: {e}")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                err_content = ""
                if self.process.stderr:
                    try:
                        raw_err = self.process.stderr.read()
                        err_content = raw_err.decode("utf-8", errors="replace")
                    except Exception:
                        pass
                raise RuntimeError(
                    f"MCP server process exited unexpectedly. Stderr: {err_content}"
                )
            with self._lock:
                if req_id in self._pending_responses:
                    resp = self._pending_responses.pop(req_id)
                    if "error" in resp:
                        raise RuntimeError(f"MCP error response: {resp['error']}")
                    return resp.get("result", {})
            time.sleep(0.01)

        raise TimeoutError(f"Timeout waiting for response to {method} (id={req_id})")

    def send_notification(self, method: str, params: dict | None = None) -> None:
        if not self._running or not self.process:
            raise RuntimeError("MCP client is not running.")

        req = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            req["params"] = params

        payload = _encode_jsonrpc_message(req)
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"Failed to send notification to MCP server: {e}")

    def list_tools(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        result = self.send_request("tools/list", {}, timeout=timeout)
        return result.get("tools", [])

    def call_tool(
        self, name: str, arguments: dict, timeout: float = 30.0
    ) -> dict[str, Any]:
        return self.send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )

    def stop(self) -> None:
        self._running = False
        if self in _active_clients:
            try:
                _active_clients.remove(self)
            except ValueError:
                pass
        if self.process:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                if self.process.stdout:
                    self.process.stdout.close()
            except Exception:
                pass
            try:
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None


_active_clients: list[McpClient] = []


def _encode_jsonrpc_message(message: dict[str, Any]) -> bytes:
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _read_jsonrpc_message(stream: BinaryIO) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        text = line.decode("ascii", errors="replace").strip()
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        headers[key.lower()] = value.strip()
    length_text = headers.get("content-length")
    if not length_text:
        return None
    try:
        length = int(length_text)
    except ValueError:
        return None
    body = stream.read(length)
    if len(body) != length:
        return None
    try:
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _cleanup_clients() -> None:
    while _active_clients:
        client = _active_clients[0]
        try:
            client.stop()
        except Exception:
            pass


atexit.register(_cleanup_clients)


class LazyClientRef:
    """Manages lazy instantiation of MCP clients to avoid startup overhead."""

    def __init__(self, server_name: str, config: dict[str, Any]) -> None:
        self.server_name = server_name
        self.config = config
        self.client: McpClient | None = None

    def get_or_create(self) -> McpClient:
        if self.client is None:
            command = [self.config["command"]] + self.config.get("args", [])
            env = self.config.get("env")
            self.client = McpClient(command, env)
            self.client.start()
        return self.client

    def stop(self) -> None:
        if self.client is not None:
            self.client.stop()
            self.client = None


def compute_config_hash(server_config: dict[str, Any]) -> str:
    data = {
        "command": server_config.get("command"),
        "args": server_config.get("args", []),
        "env": server_config.get("env", {}),
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def get_mcp_tool_risk(name: str, description: str) -> str:
    name_lower = name.lower()
    desc_lower = description.lower()

    # Writing, deleting, executing, modifying is high risk
    high_risk_keywords = {
        "write",
        "delete",
        "remove",
        "update",
        "create",
        "edit",
        "run",
        "exec",
        "execute",
        "shell",
        "bash",
        "command",
        "modify",
    }
    if any(k in name_lower or k in desc_lower for k in high_risk_keywords):
        return "high"

    # Read-only query keywords
    read_keywords = {
        "read",
        "view",
        "get",
        "list",
        "search",
        "find",
        "show",
        "query",
        "info",
        "status",
        "check",
        "inspect",
    }
    if any(k in name_lower or k in desc_lower for k in read_keywords):
        return "low"

    # Default is medium
    return "medium"


def build_fetch_tools_tool(
    project_root: Path, server_name: str, server_config: dict[str, Any]
) -> ToolSpec:
    """创建用于冷启动延迟加载服务器并拉取工具列表的引导工具。"""

    def handler(action_input: str) -> str:
        config_hash = compute_config_hash(server_config)
        try:
            command = [server_config["command"]] + server_config.get("args", [])
            env = server_config.get("env")
            client = McpClient(command, env)
            client.start()
            tools_list = client.list_tools()
            client.stop()

            for tool in tools_list:
                tool["risk"] = get_mcp_tool_risk(
                    tool["name"], tool.get("description", "")
                )

            cache_path = project_root / ".local" / "mcp_cache.json"
            cache_data = {"servers": {}}
            if cache_path.exists():
                try:
                    cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            cache_data.setdefault("servers", {})[server_name] = {
                "config_hash": config_hash,
                "tools": tools_list,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(cache_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return f"Successfully fetched {len(tools_list)} tools from server '{server_name}' and populated the cache. Please use mcp_tool_search to retrieve their schemas now!"
        except Exception as e:
            return f"Error fetching tools from server '{server_name}': {e}"

    return ToolSpec(
        name=f"mcp__{server_name}__fetch_tools",
        description=f"Bootstrap and fetch tool list for deferred server '{server_name}' to populate the cache.",
        input_hint="{}",
        handler=handler,
        risk="low",
        schema={"type": "object"},
        read_only=True,
        group="mcp",
    )


def build_mcp_tool_search(project_root: Path, deferred_servers: set[str]) -> ToolSpec:
    """创建用于搜索和获取延迟加载工具完整参数 Schema 的工具。"""

    def handler(action_input: str) -> str:
        args = parse_tool_input(action_input)
        query = str(args.get("query", "")).strip().lower()
        if not query:
            return "Please provide a query to search."

        cache_path = project_root / ".local" / "mcp_cache.json"
        cache_data = {}
        if cache_path.exists():
            try:
                cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        results = []
        for server_name in deferred_servers:
            server_entry = cache_data.get("servers", {}).get(server_name, {})
            tools = server_entry.get("tools", [])
            if not tools:
                results.append(
                    f"### Server '{server_name}'\n"
                    f"Schema not yet loaded. Please invoke the fetch_tools tool (e.g. `mcp__{server_name}__fetch_tools`) to trigger schema fetch."
                )
                continue

            matched_tools = []
            for t in tools:
                name = t.get("name", "")
                desc = t.get("description", "")
                if query in name.lower() or query in desc.lower() or query == "all":
                    matched_tools.append(t)

            if matched_tools:
                results.append(f"### Server '{server_name}' matched tools:")
                for mt in matched_tools:
                    name = mt["name"]
                    desc = mt.get("description", "")
                    schema = mt.get("inputSchema", {})
                    required = schema.get("required", [])
                    props = schema.get("properties", {})

                    param_lines = []
                    for p_name, p_info in props.items():
                        req_str = " (required)" if p_name in required else ""
                        param_lines.append(
                            f"    - **{p_name}** ({p_info.get('type', 'any')}): {p_info.get('description', '')}{req_str}"
                        )
                    params_str = (
                        "\n".join(param_lines)
                        if param_lines
                        else "    - No parameters required."
                    )

                    results.append(
                        f"- **Tool Name**: `mcp__{server_name}__{name}`\n"
                        f"  **Description**: {desc}\n"
                        f"  **Schema / Parameters**:\n{params_str}"
                    )

        if not results:
            return f"No tools found matching query: {query}"

        return "\n\n".join(results)

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


def build_mcp_tools(project_root: Path) -> tuple[ToolSpec, ...]:
    """根据配置文件加载和构建所有 MCP 工具，支持 defer_loading 延迟加载。"""
    local_config = project_root / ".local" / "mcp_config.json"
    root_config = project_root / "mcp_config.json"
    config_path = (
        local_config
        if local_config.exists()
        else (root_config if root_config.exists() else None)
    )

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
    cache_data: dict[str, Any] = {"servers": {}}
    if cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    tools_to_register = []
    deferred_servers = set()

    for server_name, server_config in mcp_servers.items():
        is_deferred = bool(server_config.get("defer_loading", False))
        if is_deferred:
            deferred_servers.add(server_name)

        config_hash = compute_config_hash(server_config)
        cached_entry = cache_data.get("servers", {}).get(server_name, {})

        tools_list = None
        if cached_entry.get("config_hash") == config_hash:
            tools_list = cached_entry.get("tools")

        if tools_list is None:
            if is_deferred:
                # 延迟加载且无缓存时，静默跳过启动，注册引导工具
                tools_list = []
                tools_to_register.append(
                    build_fetch_tools_tool(project_root, server_name, server_config)
                )
            else:
                # 缓存失效或缺失，执行一次初始化连接来更新缓存
                try:
                    command = [server_config["command"]] + server_config.get("args", [])
                    env = server_config.get("env")
                    client = McpClient(command, env)
                    client.start()
                    tools_list = client.list_tools()
                    client.stop()

                    for tool in tools_list:
                        tool["risk"] = get_mcp_tool_risk(
                            tool["name"], tool.get("description", "")
                        )

                    cache_data.setdefault("servers", {})[server_name] = {
                        "config_hash": config_hash,
                        "tools": tools_list,
                    }
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(cache_data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception as e:
                    print(f"Error querying tools from MCP server '{server_name}': {e}")
                    tools_list = cached_entry.get("tools") or []

        lazy_ref = LazyClientRef(server_name, server_config)

        for tool in tools_list:
            name = tool["name"]
            desc = tool.get("description", "")
            input_schema = tool.get("inputSchema", {})

            props = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            hints = []
            for p_name, p_info in props.items():
                req_str = " (required)" if p_name in required else ""
                hints.append(f"{p_name}: {p_info.get('type', 'any')}{req_str}")
            input_hint = ", ".join(hints) if hints else "no arguments"

            # MCP 风险覆写与 Cache 持久化
            risk = None
            overrides = server_config.get("overrides", {})
            if isinstance(overrides, dict) and name in overrides:
                override_val = overrides[name]
                if isinstance(override_val, dict):
                    risk = override_val.get("risk")
                elif isinstance(override_val, str):
                    risk = override_val

            if not risk:
                risk = tool.get("risk")

            if not risk:
                risk = get_mcp_tool_risk(name, desc)
                tool["risk"] = risk

            # 根据是否延迟加载，重写描述与 Schema
            if is_deferred:
                tool_schema = {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                }
                tool_description = f"[Deferred] {desc} Parameters unknown until searched. Call mcp_tool_search first to retrieve the required schema before invoking this tool. [mcp: {server_name}]"
            else:
                tool_schema = input_schema
                tool_description = (
                    f"{desc} [mcp: {server_name}]" if desc else f"[mcp: {server_name}]"
                )

            # 统一 Handler 包装器
            def make_handler(
                ref: LazyClientRef, t_name: str, deferred: bool, s_name: str
            ) -> Any:
                def handler(action_input: str) -> str:
                    args = parse_tool_input(action_input)
                    if deferred:
                        # 延迟加载工具执行时， JIT 强校验 Required 参数
                        # 获取缓存的真实 Schema 进行 JIT 校验
                        try:
                            c_data = json.loads(cache_path.read_text(encoding="utf-8"))
                            t_entry = next(
                                (
                                    t
                                    for t in c_data.get("servers", {})
                                    .get(s_name, {})
                                    .get("tools", [])
                                    if t["name"] == t_name
                                ),
                                None,
                            )
                            if t_entry:
                                real_schema = t_entry.get("inputSchema", {})
                                req_fields = real_schema.get("required", [])
                                missing = [f for f in req_fields if f not in args]
                                if missing:
                                    raise ValueError(
                                        f"Missing required parameters for deferred tool {t_name}: {', '.join(missing)}"
                                    )
                        except Exception as val_exc:
                            if isinstance(val_exc, ValueError):
                                raise
                            # 缓存读取失败时，静默通过，由 MCP 服务自身在执行时报错
                            pass

                    client_instance = ref.get_or_create()
                    res = client_instance.call_tool(t_name, args)
                    if "content" in res:
                        parts = []
                        for block in res["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                parts.append(block.get("text", ""))
                        content_str = "\n".join(parts)
                        if res.get("isError", False):
                            raise RuntimeError(content_str)
                        return content_str
                    return str(res)

                return handler

            mcp_name = f"mcp__{server_name}__{name}"
            spec = ToolSpec(
                name=mcp_name,
                description=tool_description,
                input_hint=input_hint,
                handler=make_handler(lazy_ref, name, is_deferred, server_name),
                risk=risk,
                schema=tool_schema,
                read_only=(risk == "low"),
                group="mcp",
            )
            tools_to_register.append(spec)

    # 如果有延迟加载的服务器，注册统一搜索工具
    if deferred_servers:
        tools_to_register.append(build_mcp_tool_search(project_root, deferred_servers))

    return tuple(tools_to_register)
