"""Xcode Web 服务启动器。"""

from __future__ import annotations

from pathlib import Path

from xcode.coding_agent.app import XcodeApp, build_app
from xcode.harness.config import (
    XcodeRuntimeConfig,
    discover_runtime_config,
    resolve_config_path,
)


def run_web_server(
    project_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    config_path: Path | None = None,
    runtime_config: XcodeRuntimeConfig | None = None,
    sessions_dir: Path | None = None,
    open_browser: bool = False,
) -> int:
    """启动 Web 服务；阻塞直到进程退出。"""
    if runtime_config is None:
        runtime_config = discover_runtime_config(project_root, config_path)
    resolved_sessions = (
        sessions_dir
        or resolve_config_path(project_root, runtime_config.paths.sessions_dir)
        or (project_root / ".xcode" / "sessions")
    )
    app = build_app(
        project_root=project_root,
        runtime_config=runtime_config,
        sessions_dir=resolved_sessions,
    )

    def build_for(target: Path) -> XcodeApp:
        """为指定工作区重新装配运行时（工作区切换用）。"""
        rc = discover_runtime_config(target, None)
        sessions = (
            resolve_config_path(target, rc.paths.sessions_dir)
            or (target / ".xcode" / "sessions")
        )
        return build_app(
            project_root=target,
            runtime_config=rc,
            sessions_dir=sessions,
        )

    try:
        from .api import create_app

        fastapi_app = create_app(app, project_root, app_factory=build_for)
        if open_browser:
            import threading
            import webbrowser
            from time import perf_counter

            started = perf_counter()

            def _open() -> None:
                # 给 uvicorn 一点启动时间再打开浏览器
                import time as _time

                while perf_counter() - started < 1.5:
                    _time.sleep(0.1)
                webbrowser.open(f"http://{host}:{port}")

            threading.Thread(target=_open, daemon=True).start()

        import uvicorn

        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        app.close()
