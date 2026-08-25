"""Xcode Web 服务：REST + WebSocket API 与静态前端。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from xcode.coding_agent.app import XcodeApp
from xcode.coding_agent.execution_modes import (
    ExecutionMode,
    parse_execution_mode,
)

from .runner import WebRunHub

STATIC_DIR = Path(__file__).parent / "static"
_SESSION_TRANSCRIPT_LIMIT = 500


AppFactory = Callable[[Path], XcodeApp]


def create_app(
    app: XcodeApp,
    project_root: Path,
    app_factory: AppFactory | None = None,
) -> FastAPI:
    """装配 FastAPI 应用（静态资源 + API + WebSocket）。"""
    server = FastAPI(title="xcode web", docs_url=None, redoc_url=None)
    hub = WebRunHub(app)

    server.state.xcode_app = app
    server.state.hub = hub
    server.state.project_root = project_root

    @server.get("/api/info")
    async def info() -> JSONResponse:
        return JSONResponse(_info_payload(hub.app, server.state.project_root))

    @server.get("/api/model")
    async def model_info() -> JSONResponse:
        return JSONResponse(_model_payload(hub.app))

    @server.post("/api/model")
    async def set_model_endpoint(payload: dict) -> JSONResponse:
        model = str(payload.get("model", "") or "")
        if not model:
            return JSONResponse({"error": "model 不能为空"}, status_code=400)
        try:
            hub.app.set_model(model=model, profile="main")
        except Exception as exc:  # noqa: BLE001 - 返回给前端展示
            return JSONResponse({"error": f"切换模型失败: {exc}"}, status_code=400)
        return JSONResponse(_model_payload(hub.app))

    @server.get("/api/workspaces")
    async def workspaces() -> JSONResponse:
        recent = _load_workspaces()
        current = str(hub.app.session_store.project_root)
        return JSONResponse(
            {"current": current, "recent": _with_current(recent, current)}
        )

    @server.post("/api/workspaces")
    async def switch_workspace_endpoint(payload: dict) -> JSONResponse:
        factory = app_factory
        if factory is None:
            return JSONResponse(
                {"error": "服务未启用工作区切换"}, status_code=400
            )
        raw = str(payload.get("path", "") or "").strip()
        if not raw:
            return JSONResponse({"error": "路径不能为空"}, status_code=400)
        target = Path(raw).expanduser()
        if not target.is_dir():
            return JSONResponse({"error": f"目录不存在: {raw}"}, status_code=400)
        if hub.is_running:
            return JSONResponse(
                {"error": "回合运行中，无法切换工作区"}, status_code=409
            )
        loop = asyncio.get_running_loop()
        try:
            new_app = await loop.run_in_executor(None, lambda: factory(target))
        except Exception as exc:  # noqa: BLE001 - 返回给前端展示
            return JSONResponse(
                {"error": f"工作区装配失败: {exc}"}, status_code=400
            )
        old_app = hub.app
        hub.set_app(new_app)
        old_app.close()
        server.state.project_root = target
        _save_workspace(str(target))
        hub.broadcast(
            {
                "type": "workspace_switched",
                "info": _info_payload(new_app, target),
            }
        )
        return JSONResponse(
            {"workspace": str(target), "info": _info_payload(new_app, target)}
        )

    @server.get("/api/sessions")
    async def sessions() -> JSONResponse:
        store = hub.app.session_store
        try:
            infos = store.list_infos(limit=50)
        except Exception as exc:  # noqa: BLE001 - 会话目录可能损坏
            return JSONResponse({"error": f"无法读取会话索引: {exc}", "sessions": []})
        return JSONResponse(
            {
                "current": store.session_id,
                "sessions": [
                    {
                        "id": item.id,
                        "title": item.title,
                        "summary": item.summary,
                        "updated_at": item.updated_at,
                        "project": item.project_path,
                    }
                    for item in infos
                ],
            }
        )

    @server.post("/api/sessions")
    async def new_session_endpoint() -> JSONResponse:
        new_id = hub.new_session()
        if new_id is None:
            return JSONResponse(
                {"error": "当前回合运行中，请先停止再新建会话。"}, status_code=409
            )
        return JSONResponse(
            {"session_id": new_id}
        )

    @server.get("/api/sessions/{session_id}")
    async def session_transcript(session_id: str) -> JSONResponse:
        store = hub.app.session_store
        try:
            view = store.find_by_id(session_id)
        except Exception:  # noqa: BLE001
            view = None
        if view is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        entries = _read_transcript(view.path)
        return JSONResponse(
            {
                "id": view.id,
                "title": view.title,
                "summary": view.summary,
                "entries": entries,
            }
        )

    @server.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        def sink(payload: dict[str, object]) -> None:
            # 审批回调可能来自工具线程，必须线程安全地投递
            loop.call_soon_threadsafe(queue.put_nowait, payload)

        hub.attach(sink)

        async def forward() -> None:
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)

        sender = asyncio.create_task(forward())
        try:
            while True:
                message = await websocket.receive_json()
                await _handle_message(hub, message, websocket)
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            hub.detach(sink)

    server.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return server


def _model_payload(app: XcodeApp) -> dict[str, object]:
    """当前模型信息 + 可用模型列表。"""
    info: dict[str, object] = dict(app.get_model_info())
    try:
        from xcode.cli.repl import current_model_options

        info["available"] = list(current_model_options(app))
    except Exception:  # noqa: BLE001
        info["available"] = [str(info.get("model", ""))]
    return info


def _info_payload(app: XcodeApp, project_root: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "project": str(project_root),
        "session_id": app.session_store.session_id,
    }
    try:
        payload["model"] = app.get_model_info()
    except Exception:  # noqa: BLE001
        payload["model"] = {}
    try:
        payload["mcp"] = [
            {
                "server": st.get("server"),
                "status": st.get("status"),
                "tools": _count_tools(st.get("tools")),
            }
            for st in app.mcp_status()
        ]
    except Exception:  # noqa: BLE001
        payload["mcp"] = []
    try:
        from xcode.cli.git import git_branch_name

        payload["git_branch"] = git_branch_name(project_root)
    except Exception:  # noqa: BLE001
        payload["git_branch"] = None
    return payload


async def _handle_message(
    hub: WebRunHub,
    message: dict[str, object],
    websocket: WebSocket,
) -> None:
    message_type = message.get("type")
    if message_type == "submit":
        text = str(message.get("text", ""))
        raw_mode = message.get("mode")
        mode = _parse_mode(raw_mode)
        hub.submit(text, mode)
    elif message_type == "cancel":
        hub.cancel()
    elif message_type == "approval":
        resolved = hub.resolve_approval(
            request_id=str(message.get("id", "")),
            decision=str(message.get("decision", "deny")),
            scope=str(message.get("scope", "once")),
            suggestion=str(message.get("suggestion", "")),
        )
        if not resolved:
            await websocket.send_json(
                {"type": "run_error", "message": "审批请求不存在或已失效"}
            )
    elif message_type == "ping":
        await websocket.send_json({"type": "pong"})
    else:
        await websocket.send_json(
            {"type": "run_error", "message": f"未知消息类型: {message_type}"}
        )


def _parse_mode(raw: object) -> ExecutionMode | None:
    return parse_execution_mode(raw)


def _count_tools(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


_WORKSPACES_FILE = Path.home() / ".xcode" / "web_workspaces.json"
_WORKSPACES_LIMIT = 10


def _load_workspaces() -> list[str]:
    try:
        data = json.loads(_WORKSPACES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item) for item in data if isinstance(item, str)]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _save_workspace(path: str) -> None:
    recent = _with_current(_load_workspaces(), path)
    try:
        _WORKSPACES_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WORKSPACES_FILE.write_text(
            json.dumps(recent, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _with_current(recent: list[str], current: str) -> list[str]:
    merged = [current] + [item for item in recent if item != current]
    return merged[:_WORKSPACES_LIMIT]


def _read_transcript(path: Path) -> list[dict[str, object]]:
    """读取单个会话 JSONL 账本的前若干条记录。"""
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(entries) >= _SESSION_TRANSCRIPT_LIMIT:
            break
    return entries
