"""基于官方 Python SDK 的 MCP stdio 客户端。"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, IO, Literal, TextIO, TypeVar, cast

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.session import MessageHandlerFnT
from mcp.client.stdio import stdio_client
from mcp.shared.session import RequestResponder
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS as _SDK_PROTOCOL_VERSIONS
from pydantic import BaseModel

from xcode.harness.agent_runtime.async_worker import IsolatedAsyncWorker

logger = logging.getLogger(__name__)

LATEST_PROTOCOL_VERSION = types.LATEST_PROTOCOL_VERSION
SUPPORTED_PROTOCOL_VERSIONS = tuple(_SDK_PROTOCOL_VERSIONS)
MAX_TOOL_LIST_PAGES = 100
MAX_LAZY_CONNECT_ATTEMPTS = 2
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0

T = TypeVar("T")
_Operation = Literal["list_tools", "call_tool", "close"]

_REDACT_PATTERNS: list[re.Pattern[str]] = [
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
        return redacted[:max_len] + "..."
    return redacted


def _model_dict(model: BaseModel) -> dict[str, Any]:
    """将 SDK 模型转换为保留 MCP 字段别名的普通字典。"""
    return model.model_dump(by_alias=True, mode="json", exclude_none=True)


class _ErrorLog:
    """使用真实文件描述符捕获 Windows 子进程 stderr。"""

    def __init__(self) -> None:
        self.stream: IO[str] = tempfile.TemporaryFile(
            mode="w+",
            encoding="utf-8",
            errors="replace",
        )

    def snapshot(self) -> str:
        """返回当前 stderr 缓冲区内容。"""
        position = self.stream.tell()
        self.stream.flush()
        self.stream.seek(0)
        content = self.stream.read()
        self.stream.seek(position)
        return content

    def close(self) -> None:
        """关闭临时文件。"""
        self.stream.close()


@dataclass(frozen=True)
class _SdkCommand:
    """交给 session owner task 串行执行的操作。"""

    operation: _Operation
    future: concurrent.futures.Future[object]
    name: str | None = None
    arguments: dict[str, Any] | None = None
    timeout: float | None = None
    progress_callback: Callable[[float, float | None, str | None], None] | None = None
    cancel_event: threading.Event | None = None


class McpClient:
    """同步外观的官方 SDK stdio client。

    一个长期 owner task 同时负责 transport/session 的进入、请求和退出，
    满足 AnyIO cancel scope 必须由同一 task 管理的生命周期约束。
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        tools_changed_callback: Callable[[McpClient], None] | None = None,
        workspace_roots: tuple[Path, ...] = (),
    ) -> None:
        """保存连接配置并初始化 SDK session 状态。"""
        if not command:
            raise ValueError("MCP command must not be empty")
        self.command = command
        self.env = os.environ.copy()
        if env:
            self.env.update(env)
        self._timeout = timeout
        self._tools_changed_callback = tools_changed_callback
        self._workspace_roots = tuple(workspace_roots)
        self._worker = IsolatedAsyncWorker(name="xcode-mcp-client")
        self._commands: asyncio.Queue[_SdkCommand] | None = None
        self._owner_future: concurrent.futures.Future[None] | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._stderr = _ErrorLog()
        self._status = "pending"
        self._callback_lock = threading.Lock()
        self._callback_running = False
        self._callback_pending = False
        self._session: ClientSession | None = None
        self._latest_progress: dict[str, object] | None = None
        self.protocol_version: str | None = None
        self.server_capabilities: dict[str, Any] = {}
        self.server_info: dict[str, Any] = {}
        self.instructions: str | None = None

    @property
    def status(self) -> str:
        """返回当前连接状态。"""
        return self._status

    @property
    def latest_progress(self) -> dict[str, object] | None:
        """返回最近一次工具调用进度。"""
        return self._latest_progress

    def start(self) -> None:
        """启动 owner task、stdio server 并完成 SDK initialize。"""
        if self._status == "connected":
            return
        self._owner_future = self._worker.submit(self._session_owner())
        if not self._ready.wait(timeout=self._effective_timeout()):
            self._status = "failed"
            self._shutdown_worker()
            raise TimeoutError("Timeout waiting for MCP SDK initialization")
        if self._startup_error is not None:
            self._status = "failed"
            diagnostic = truncate_redact(self._stderr.snapshot())
            self._shutdown_worker()
            suffix = f" Stderr: {diagnostic}" if diagnostic else ""
            raise RuntimeError(
                f"MCP handshake failed: {self._startup_error}.{suffix}"
            ) from self._startup_error
        self._status = "connected"

    async def _session_owner(self) -> None:
        """在单一 task 内管理 SDK transport/session 的完整生命周期。"""
        try:
            params = StdioServerParameters(
                command=self.command[0],
                args=self.command[1:],
                env=self.env,
            )
            async with stdio_client(
                params,
                errlog=cast(TextIO, self._stderr.stream),
            ) as (read_stream, write_stream):
                session_kwargs: dict[str, object] = {
                    "read_timeout_seconds": self._timeout_delta(),
                    "message_handler": self._message_handler(),
                    "client_info": types.Implementation(
                        name="xcode-client",
                        version="0.1.1",
                    ),
                }
                if self._workspace_roots:
                    session_kwargs["list_roots_callback"] = self._list_roots_callback
                async with ClientSession(
                    read_stream,
                    write_stream,
                    **session_kwargs,
                ) as session:
                    result = await session.initialize()
                    self._apply_initialize_result(result)
                    self._session = session
                    self._commands = asyncio.Queue()
                    self._ready.set()
                    await self._serve_commands(session)
        except BaseException as exc:
            if not self._ready.is_set():
                self._startup_error = exc
                self._ready.set()
                return
            logger.warning("MCP SDK session owner stopped unexpectedly", exc_info=True)
            self._fail_pending_commands(exc)
        finally:
            self._session = None

    async def _list_roots_callback(
        self,
        _context: object,
    ) -> types.ListRootsResult:
        """向 server 暴露当前允许读取的工作区根目录。"""
        roots = [
            types.Root(
                uri=path.resolve().as_uri(),
                name=path.name or str(path.resolve()),
            )
            for path in self._workspace_roots
            if path.exists()
        ]
        return types.ListRootsResult(roots=roots)

    def _apply_initialize_result(self, result: types.InitializeResult) -> None:
        """保存 SDK 校验后的协商元数据。"""
        self.protocol_version = str(result.protocolVersion)
        self.server_capabilities = _model_dict(result.capabilities)
        self.server_info = _model_dict(result.serverInfo)
        self.instructions = result.instructions

    async def _serve_commands(self, session: ClientSession) -> None:
        """串行执行宿主请求，close 命令返回后退出 context。"""
        assert self._commands is not None
        while True:
            command = await self._commands.get()
            if command.operation == "close":
                command.future.set_result(None)
                return
            try:
                if command.operation == "list_tools":
                    result = await self._list_tools(session, command.timeout)
                else:
                    assert command.name is not None
                    result = await self._call_tool(
                        session,
                        command.name,
                        command.arguments or {},
                        command.timeout,
                        command.progress_callback,
                    )
            except BaseException as exc:
                if not command.future.cancelled():
                    command.future.set_exception(self._normalize_error(exc))
            else:
                if not command.future.cancelled():
                    command.future.set_result(result)

    async def _list_tools(
        self,
        session: ClientSession,
        timeout: float | None,
    ) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(MAX_TOOL_LIST_PAGES):
            result = await asyncio.wait_for(
                session.list_tools(cursor=cursor),
                timeout=self._effective_timeout(timeout),
            )
            tools.extend(_model_dict(tool) for tool in result.tools)
            next_cursor = result.nextCursor
            if next_cursor is None:
                return tools
            if not next_cursor:
                raise RuntimeError("MCP tools/list response has invalid nextCursor")
            if next_cursor in seen_cursors:
                raise RuntimeError(f"MCP tools/list repeated cursor: {next_cursor!r}")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        raise RuntimeError(f"MCP tools/list exceeded {MAX_TOOL_LIST_PAGES} pages")

    async def _call_tool(
        self,
        session: ClientSession,
        name: str,
        arguments: dict[str, Any],
        timeout: float | None,
        progress_callback: Callable[[float, float | None, str | None], None] | None,
    ) -> dict[str, Any]:
        async def on_progress(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            self._latest_progress = {
                "progress": progress,
                "total": total,
                "message": message,
            }
            if progress_callback is not None:
                progress_callback(progress, total, message)

        result = await session.call_tool(
            name,
            arguments,
            read_timeout_seconds=self._timeout_delta(timeout),
            progress_callback=on_progress,
        )
        self._latest_progress = None
        return _model_dict(result)

    def _message_handler(self) -> MessageHandlerFnT:
        """创建仅消费 Xcode 所需通知的 SDK 消息处理器。"""

        async def handler(
            message: RequestResponder[types.ServerRequest, types.ClientResult]
            | types.ServerNotification
            | Exception,
        ) -> None:
            if isinstance(message, types.ServerNotification) and isinstance(
                message.root,
                types.ToolListChangedNotification,
            ):
                self._schedule_tools_refresh()
            await asyncio.sleep(0)

        return handler

    def has_server_capability(self, capability: str) -> bool:
        """判断 initialize 是否声明指定 server capability。"""
        return isinstance(self.server_capabilities.get(capability), dict)

    def set_tools_changed_callback(
        self,
        callback: Callable[[McpClient], None] | None,
    ) -> None:
        """设置工具列表变化后的宿主刷新回调。"""
        self._tools_changed_callback = callback

    def list_tools(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """列出服务器工具并聚合 SDK 分页结果。"""
        self._require_connected("tools/list")
        self._require_tools_capability("tools/list")
        return cast(
            list[dict[str, Any]],
            self._execute(
                _SdkCommand("list_tools", concurrent.futures.Future(), timeout=timeout)
            ),
        )

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
        progress_callback: Callable[[float, float | None, str | None], None]
        | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """调用服务器工具并返回普通 MCP 结果字典。"""
        self._require_connected("tools/call")
        self._require_tools_capability("tools/call")
        return cast(
            dict[str, Any],
            self._execute(
                _SdkCommand(
                    "call_tool",
                    concurrent.futures.Future(),
                    name=name,
                    arguments=arguments,
                    timeout=timeout,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            ),
        )

    def update_workspace_roots(self, workspace_roots: tuple[Path, ...]) -> None:
        """更新 roots 回调使用的工作区，并在必要时通知 server。"""
        normalized = tuple(path.resolve() for path in workspace_roots if path.exists())
        if normalized == self._workspace_roots:
            return
        self._workspace_roots = normalized
        if self._status == "connected":
            self.send_roots_list_changed()

    def send_roots_list_changed(self) -> None:
        """向已连接 server 发送 roots/list_changed。"""
        if not self.has_server_capability("roots"):
            return

        async def notify() -> None:
            session = self._session
            if session is not None:
                await session.send_roots_list_changed()

        self._worker.submit(notify()).result(timeout=self._effective_timeout())

    def stop(self) -> None:
        """通知 owner task 退出并关闭 transport、session 与 worker。"""
        if self._status == "disabled":
            return
        if self._commands is not None and self._owner_future is not None:
            command = _SdkCommand("close", concurrent.futures.Future())
            try:
                self._execute(command)
                self._owner_future.result(timeout=self._effective_timeout())
            except Exception:
                logger.warning("Failed to close MCP SDK session", exc_info=True)
        self._shutdown_worker()
        self._status = "disabled"

    def _execute(self, command: _SdkCommand) -> object:
        commands = self._commands
        if commands is None:
            raise RuntimeError("MCP client is not connected")

        async def enqueue() -> None:
            await commands.put(command)

        self._worker.submit(enqueue()).result(timeout=self._effective_timeout())
        deadline = time.monotonic() + self._effective_timeout(command.timeout)
        while True:
            if command.cancel_event is not None and command.cancel_event.is_set():
                self._interrupt_active_command(command, "cancelled")
                raise RuntimeError(f"MCP SDK {command.operation} cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._interrupt_active_command(command, "timeout")
                raise TimeoutError(
                    f"Timeout waiting for MCP SDK {command.operation}"
                ) from None
            try:
                return command.future.result(timeout=min(0.1, remaining))
            except concurrent.futures.TimeoutError:
                continue

    def _shutdown_worker(self) -> None:
        self._worker.close()
        self._stderr.close()

    def _interrupt_active_command(self, command: _SdkCommand, reason: str) -> None:
        """统一回收挂起请求的底层会话与子进程。"""
        logger.info("Interrupting MCP command %s due to %s", command.operation, reason)
        command.future.cancel()
        self._status = "disabled"
        self._shutdown_worker()

    def _effective_timeout(self, timeout: float | None = None) -> float:
        if timeout is not None:
            return timeout
        if self._timeout is not None:
            return self._timeout
        return DEFAULT_REQUEST_TIMEOUT_SECONDS

    def _timeout_delta(self, timeout: float | None = None) -> timedelta:
        return timedelta(seconds=self._effective_timeout(timeout))

    def _require_connected(self, method: str) -> None:
        if self._status == "connected" and self._commands is not None:
            return
        raise RuntimeError(f"MCP client is not connected; cannot call {method}")

    def _require_tools_capability(self, method: str) -> None:
        if self.has_server_capability("tools"):
            return
        raise RuntimeError(
            f"MCP server did not negotiate 'tools'; cannot call {method}"
        )

    def _normalize_error(self, exc: BaseException) -> Exception:
        if isinstance(exc, TimeoutError):
            return exc
        diagnostic = truncate_redact(self._stderr.snapshot())
        suffix = f" Stderr: {diagnostic}" if diagnostic else ""
        return RuntimeError(f"{truncate_redact(str(exc))}.{suffix}")

    def _fail_pending_commands(self, exc: BaseException) -> None:
        commands = self._commands
        if commands is None:
            return
        while not commands.empty():
            command = commands.get_nowait()
            if not command.future.done():
                command.future.set_exception(self._normalize_error(exc))

    def _schedule_tools_refresh(self) -> None:
        """在线程中合并 tools/list_changed，避免阻塞 SDK receive loop。"""
        if self._tools_changed_callback is None:
            return
        with self._callback_lock:
            self._callback_pending = True
            if self._callback_running:
                return
            self._callback_running = True
        threading.Thread(
            target=self._run_tools_refresh,
            name="mcp-tools-list-changed",
            daemon=True,
        ).start()

    def _run_tools_refresh(self) -> None:
        while True:
            with self._callback_lock:
                if not self._callback_pending:
                    self._callback_running = False
                    return
                self._callback_pending = False
            callback = self._tools_changed_callback
            if callback is None or self._status != "connected":
                continue
            try:
                callback(self)
            except Exception:
                logger.warning("Failed to refresh MCP tools", exc_info=True)


class LazyClientRef:
    """延迟创建并复用 MCP SDK client。"""

    def __init__(
        self,
        server_name: str,
        config: dict[str, Any],
        tools_changed_callback: Callable[[McpClient], None] | None = None,
        max_connect_attempts: int = MAX_LAZY_CONNECT_ATTEMPTS,
        workspace_roots: tuple[Path, ...] = (),
    ) -> None:
        """保存服务器名称和原始运行配置。"""
        if max_connect_attempts < 1:
            raise ValueError("max_connect_attempts must be at least 1")
        self.server_name = server_name
        self.config = config
        self.tools_changed_callback = tools_changed_callback
        self.max_connect_attempts = max_connect_attempts
        self.workspace_roots = tuple(workspace_roots)
        self.client: McpClient | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def get_or_create(self) -> McpClient:
        """返回可用客户端，必要时完成有限次数重连。"""
        with self._lock:
            if self.client is not None and self.client.status == "connected":
                return self.client
            self._stop_client()

            command = [self.config["command"], *self.config.get("args", [])]
            env = self.config.get("env")
            timeout = self.config.get("timeout")
            for _ in range(self.max_connect_attempts):
                client = McpClient(
                    command,
                    env,
                    timeout=timeout,
                    workspace_roots=self.workspace_roots,
                )
                client.set_tools_changed_callback(self.tools_changed_callback)
                try:
                    client.start()
                except RuntimeError as exc:
                    self.last_error = truncate_redact(str(exc))
                    client.stop()
                    continue
                self.client = client
                self.last_error = None
                return client

            raise RuntimeError(
                f"MCP server {self.server_name!r} connection failed after "
                f"{self.max_connect_attempts} attempts: {self.last_error}"
            )

    def stop(self) -> None:
        """停止并清除当前客户端。"""
        with self._lock:
            self._stop_client()

    def update_workspace_roots(self, workspace_roots: tuple[Path, ...]) -> None:
        """更新后续连接与现有连接使用的工作区 roots。"""
        with self._lock:
            self.workspace_roots = tuple(workspace_roots)
            if self.client is not None:
                self.client.update_workspace_roots(self.workspace_roots)

    def _stop_client(self) -> None:
        if self.client is not None:
            self.client.stop()
            self.client = None
