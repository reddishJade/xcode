"""MCP 协议客户端。

基于 stdio 的 JSON-RPC 2.0 客户端，实现 Model Context Protocol 的
initialize 握手、tools/list 和 tools/call。
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, BinaryIO, cast

logger = logging.getLogger(__name__)

SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

_REDACT_PATTERNS: list[re.Pattern] = [
    re.compile(r"(Bearer\s+)[^\s]+", re.IGNORECASE),
    re.compile(r"(sk-)[^\s]+", re.IGNORECASE),
    re.compile(r"((?:api_key|token|secret)=)[^\s]+", re.IGNORECASE),
]


def redact_mcp_text(text: str) -> str:
    """脱敏 MCP 诊断文本中的常见凭据。"""
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(r"\1****", text)
    return text


def truncate_redact(text: str, max_len: int = 200) -> str:
    """先脱敏再按字符上限截断诊断文本。"""
    redacted = redact_mcp_text(text)
    if len(redacted) > max_len:
        redacted = redacted[:max_len] + "..."
    return redacted


def _is_valid_server_info(value: object) -> bool:
    """判断 serverInfo 是否包含稳定的名称和版本标识。"""
    if not isinstance(value, dict):
        return False
    name = value.get("name")
    version = value.get("version")
    return (
        isinstance(name, str)
        and bool(name.strip())
        and isinstance(version, str)
        and bool(version.strip())
    )


def _is_valid_message_id(value: object) -> bool:
    """判断 JSON-RPC id 是否为支持的字符串或整数。"""
    return isinstance(value, (int, str)) and not isinstance(value, bool)


class McpClient:
    """基于 stdio 的同步 MCP 客户端。"""

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        """保存进程配置并初始化连接状态。"""
        self.command = command
        self.env = os.environ.copy()
        if env:
            self.env.update(env)
        self.process: subprocess.Popen | None = None
        self.next_id = 1
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending_responses: dict[int | str, dict[str, Any]] = {}
        self._read_thread: threading.Thread | None = None
        self._running = False
        self._timeout = timeout
        self._status: str = "pending"
        self.protocol_version: str | None = None
        self.server_capabilities: dict[str, Any] = {}
        self.server_info: dict[str, Any] = {}
        self.instructions: str | None = None

    @property
    def status(self) -> str:
        """返回当前连接状态。"""
        return self._status

    def start(self) -> None:
        """启动服务器进程并完成 initialize 协商。"""
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
            self._status = "failed"
            raise RuntimeError(f"Failed to start MCP server {self.command}: {e}")

        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        _active_clients.append(self)

        try:
            initialize_result = self.send_request(
                "initialize",
                {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "xcode-client", "version": "0.1.0"},
                },
            )
            self._apply_initialize_result(initialize_result)
            self.send_notification("notifications/initialized")
        except Exception as e:
            self.stop()
            self._status = "failed"
            raise RuntimeError(f"MCP handshake failed: {e}")

        self._status = "connected"

    def _apply_initialize_result(self, result: dict[str, Any]) -> None:
        """校验 initialize 响应并保存协商后的服务器元数据。"""
        protocol_version = result.get("protocolVersion")
        if not isinstance(protocol_version, str):
            raise RuntimeError("MCP initialize response has no protocolVersion")
        if protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            supported = ", ".join(SUPPORTED_PROTOCOL_VERSIONS)
            raise RuntimeError(
                "Unsupported MCP protocol version "
                f"{protocol_version!r}; client supports: {supported}"
            )

        capabilities = result.get("capabilities")
        if not isinstance(capabilities, dict):
            raise RuntimeError("MCP initialize response has invalid capabilities")
        server_info = result.get("serverInfo")
        if not _is_valid_server_info(server_info):
            raise RuntimeError("MCP initialize response has invalid serverInfo")
        server_info = cast(dict[str, Any], server_info)
        instructions = result.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise RuntimeError("MCP initialize response has invalid instructions")

        self.protocol_version = protocol_version
        self.server_capabilities = capabilities
        self.server_info = server_info
        self.instructions = instructions

    def has_server_capability(self, capability: str) -> bool:
        """判断服务器是否在 initialize 响应中声明能力。"""
        return isinstance(self.server_capabilities.get(capability), dict)

    def _require_server_capability(self, capability: str, method: str) -> None:
        """拒绝调用未协商的服务器能力。"""
        if self.has_server_capability(capability):
            return
        raise RuntimeError(
            f"MCP server did not negotiate {capability!r}; cannot call {method}"
        )

    def _read_loop(self) -> None:
        """持续读取 JSON-RPC 消息并分派响应、请求和通知。"""
        while self._running and self.process and self.process.stdout:
            msg = _read_jsonrpc_message(cast(BinaryIO, self.process.stdout))
            if msg is None:
                break
            self._handle_incoming_message(msg)

    def _handle_incoming_message(self, message: dict[str, Any]) -> None:
        """按 JSON-RPC 形状分派单条入站消息。"""
        if "id" in message and ("result" in message or "error" in message):
            response_id = message.get("id")
            if not _is_valid_message_id(response_id):
                logger.warning("Ignoring MCP response with invalid id: %s", message)
                return
            response_id = cast(int | str, response_id)
            with self._lock:
                self._pending_responses[response_id] = message
            return

        method = message.get("method")
        if not isinstance(method, str):
            logger.warning("Ignoring malformed MCP JSON-RPC message: %s", message)
            return
        if "id" in message:
            request_id = message.get("id")
            if not _is_valid_message_id(request_id):
                logger.warning(
                    "Ignoring MCP server request with invalid id: %s", message
                )
                return
            request_id = cast(int | str, request_id)
            self._handle_server_request(request_id, method)
            return
        self._handle_server_notification(method)

    def _handle_server_request(self, request_id: int | str, method: str) -> None:
        """处理 server-to-client request。"""
        if method == "ping":
            self._send_jsonrpc_response(request_id, result={})
            return
        self._send_jsonrpc_response(
            request_id,
            error={
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        )

    def _handle_server_notification(self, method: str) -> None:
        """记录尚未支持的 server notification。"""
        logger.warning("Ignoring unsupported MCP server notification: %s", method)

    def _send_jsonrpc_response(
        self,
        request_id: int | str,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        """向服务器发送 JSON-RPC response。"""
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if error is not None:
            response["error"] = error
        else:
            response["result"] = result or {}
        try:
            self._write_message(response)
        except RuntimeError:
            logger.warning(
                "Failed to respond to MCP server request %r",
                request_id,
                exc_info=True,
            )

    def send_request(
        self, method: str, params: dict, timeout: float | None = None
    ) -> dict[str, Any]:
        """发送 JSON-RPC request 并同步等待对应响应。"""
        if not self._running or not self.process:
            raise RuntimeError("MCP client is not running.")

        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout is None:
            effective_timeout = 10.0

        with self._lock:
            req_id = self.next_id
            self.next_id += 1

        req = {"jsonrpc": "2.0", "method": method, "id": req_id, "params": params}
        self._write_message(req)

        start_time = time.time()
        while time.time() - start_time < effective_timeout:
            if self.process.poll() is not None:
                err_content = ""
                if self.process.stderr:
                    try:
                        raw_err = self.process.stderr.read()
                        raw_text = raw_err.decode("utf-8", errors="replace")
                        err_content = truncate_redact(raw_text, max_len=200)
                    except Exception:
                        logger.debug("failed to read MCP server stderr", exc_info=True)
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
        """发送不需要响应的 JSON-RPC notification。"""
        if not self._running or not self.process:
            raise RuntimeError("MCP client is not running.")

        req: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params

        self._write_message(req)

    def _write_message(self, message: dict[str, Any]) -> None:
        """串行写入单条 JSON-RPC 消息，避免多线程交错。"""
        if not self._running or not self.process:
            raise RuntimeError("MCP client is not running.")
        payload = _encode_jsonrpc_message(message)
        try:
            with self._write_lock:
                assert self.process.stdin is not None
                self.process.stdin.write(payload)
                self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to write to MCP server: {exc}") from exc

    def list_tools(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """列出服务器工具，要求已协商 tools capability。"""
        self._require_server_capability("tools", "tools/list")
        result = self.send_request("tools/list", {}, timeout=timeout)
        return result.get("tools", [])

    def call_tool(
        self, name: str, arguments: dict, timeout: float | None = None
    ) -> dict[str, Any]:
        """调用服务器工具，要求已协商 tools capability。"""
        self._require_server_capability("tools", "tools/call")
        return self.send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )

    def stop(self) -> None:
        """停止读取循环并终止服务器进程。"""
        self._running = False
        if self in _active_clients:
            try:
                _active_clients.remove(self)
            except ValueError:
                pass
        if self.process:
            for stream_attr in ("stdin", "stdout", "stderr"):
                stream = getattr(self.process, stream_attr, None)
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        logger.debug(
                            "failed to close MCP server stream %s",
                            stream_attr,
                            exc_info=True,
                        )
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    logger.debug("failed to kill MCP server process", exc_info=True)
            self.process = None
        self._status = "disabled"


# ── JSON-RPC 编解码 ──

_active_clients: list[McpClient] = []


def _encode_jsonrpc_message(message: dict[str, Any]) -> bytes:
    """将 JSON-RPC 消息编码为换行分隔 JSON。"""
    body = json.dumps(message, ensure_ascii=False)
    return (body + "\n").encode("utf-8")


def _read_jsonrpc_message(stream: BinaryIO) -> dict[str, Any] | None:
    """读取换行分隔 JSON，并兼容旧 Content-Length framing。"""
    line = stream.readline()
    if not line:
        return None

    text = line.decode("utf-8", errors="replace")

    # NDJSON: line starts with '{'
    stripped = text.strip()
    if stripped and stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    # Content-Length header format
    headers: dict[str, str] = {}
    # The first line we read may already be a header
    header_line = text.strip()
    if ":" in header_line:
        key, value = header_line.split(":", 1)
        headers[key.lower()] = value.strip()

    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        header_text = line.decode("ascii", errors="replace").strip()
        if ":" not in header_text:
            continue
        key, value = header_text.split(":", 1)
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
    """在进程退出时停止仍存活的 MCP 客户端。"""
    while _active_clients:
        client = _active_clients[0]
        try:
            client.stop()
        except Exception:
            logger.debug(
                "failed to stop MCP client during atexit cleanup", exc_info=True
            )


atexit.register(_cleanup_clients)


class LazyClientRef:
    """延迟创建并复用 MCP 客户端。"""

    def __init__(self, server_name: str, config: dict[str, Any]) -> None:
        """保存服务器名称和原始运行配置。"""
        self.server_name = server_name
        self.config = config
        self.client: McpClient | None = None

    def get_or_create(self) -> McpClient:
        """返回可用客户端，必要时完成启动和握手。"""
        if self.client is None or self.client.status == "failed":
            if self.client is not None:
                self.client.stop()
            command = [self.config["command"]] + self.config.get("args", [])
            env = self.config.get("env")
            timeout = self.config.get("timeout")
            self.client = McpClient(command, env, timeout=timeout)
            self.client.start()
        return self.client

    def stop(self) -> None:
        """停止并清除当前客户端。"""
        if self.client is not None:
            self.client.stop()
            self.client = None
