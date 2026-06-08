"""自动下载和管理外部工具（如 fd、ripgrep）。

对齐 Pi-mono 的 ensureTool 逻辑：检测 PATH → 下载预编译二进制 → 缓存到 ~/.xcode/bin/。
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

logger = logging.getLogger("xcode.coding_agent.tools.tools_manager")

_XCODE_BIN_DIR: Path | None = None


def _get_bin_dir() -> Path:
    global _XCODE_BIN_DIR
    if _XCODE_BIN_DIR is None:
        _XCODE_BIN_DIR = Path.home() / ".xcode" / "bin"
        _XCODE_BIN_DIR.mkdir(parents=True, exist_ok=True)
    return _XCODE_BIN_DIR


_OFFLINE_ENV_VAR = "XCODE_OFFLINE"
# 网络超时配置（平衡用户体验与网络稳定性）
_NETWORK_TIMEOUT = 10      # API 请求超时：防止网络卡顿
_DOWNLOAD_TIMEOUT = 120    # 二进制下载超时：适配慢速网络


def _fd_asset(version: str, os_name: str, arch: str) -> str | None:
    if os_name == "darwin":
        return f"fd-v{version}-{arch}-apple-darwin.tar.gz"
    if os_name == "linux":
        return f"fd-v{version}-{arch}-unknown-linux-gnu.tar.gz"
    if os_name == "win32":
        return f"fd-v{version}-{arch}-pc-windows-msvc.zip"
    return None


def _rg_asset(version: str, os_name: str, arch: str) -> str | None:
    if os_name == "darwin":
        return f"ripgrep-{version}-{arch}-apple-darwin.tar.gz"
    if os_name == "linux":
        if arch == "aarch64":
            return f"ripgrep-{version}-aarch64-unknown-linux-gnu.tar.gz"
        return f"ripgrep-{version}-x86_64-unknown-linux-musl.tar.gz"
    if os_name == "win32":
        return f"ripgrep-{version}-{arch}-pc-windows-msvc.zip"
    return None


_TOOLS: dict[str, dict[str, Any]] = {
    "fd": {
        "repo": "sharkdp/fd",
        "binary_name": "fd",
        "system_names": ["fd", "fdfind"],
        "tag_prefix": "v",
        "get_asset": _fd_asset,
    },
    "rg": {
        "repo": "BurntSushi/ripgrep",
        "binary_name": "rg",
        "system_names": ["rg"],
        "tag_prefix": "",
        "get_asset": _rg_asset,
    },
}


def _is_offline() -> bool:
    return os.environ.get(_OFFLINE_ENV_VAR, "").lower() in ("1", "true")


def _normalize_arch(raw: str) -> str:
    m = {
        "AMD64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }
    return m.get(raw, raw)


def _command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def get_tool_path(tool: str) -> str | None:
    """检查工具是否可用（本地缓存优先，其次系统 PATH）。"""
    config = _TOOLS.get(tool)
    if not config:
        return None

    bin_dir = _get_bin_dir()
    binary_path = bin_dir / (
        config["binary_name"] + (".exe" if sys.platform == "win32" else "")
    )
    if binary_path.exists():
        return str(binary_path)

    for name in config["system_names"]:
        found = shutil.which(name)
        if found:
            return found

    return None


def _get_latest_version(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urlopen(url, timeout=_NETWORK_TIMEOUT)
    data = req.read().decode("utf-8")
    import json

    parsed = json.loads(data)
    tag = parsed["tag_name"]
    return tag.lstrip("v")


def _download_file(url: str, dest: Path) -> None:
    req = urlopen(url, timeout=_DOWNLOAD_TIMEOUT)
    with dest.open("wb") as f:
        while True:
            chunk = req.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _find_binary(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        if path.is_file():
            return path
    return None


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    if archive_path.suffix == ".zip" or str(archive_path).endswith(".zip"):
        with zipfile.ZipFile(str(archive_path), "r") as zf:
            zf.extractall(str(extract_dir))
    elif str(archive_path).endswith(".tar.gz"):
        with tarfile.open(str(archive_path), "r:gz") as tf:
            tf.extractall(str(extract_dir))
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def _download_tool(tool: str) -> str:
    config = _TOOLS.get(tool)
    if not config:
        raise ValueError(f"Unknown tool: {tool}")

    os_name = {"win32": "win32", "darwin": "darwin", "linux": "linux"}.get(
        sys.platform, sys.platform
    )
    raw_arch = platform.machine()
    arch = _normalize_arch(raw_arch)

    version = _get_latest_version(config["repo"])
    asset_name = config["get_asset"](version, os_name, arch)
    if not asset_name:
        raise ValueError(f"Unsupported platform: {os_name}/{arch}")

    bin_dir = _get_bin_dir()
    download_url = (
        f"https://github.com/{config['repo']}/releases/download/"
        f"{config['tag_prefix']}{version}/{asset_name}"
    )
    archive_path = bin_dir / asset_name
    binary_ext = ".exe" if sys.platform == "win32" else ""
    binary_path = bin_dir / f"{config['binary_name']}{binary_ext}"

    logger.info("Downloading %s from %s", tool, download_url)
    _download_file(download_url, archive_path)

    with tempfile.TemporaryDirectory(dir=str(bin_dir)) as tmp:
        tmp_dir = Path(tmp)
        _extract_archive(archive_path, tmp_dir)

        extracted = _find_binary(tmp_dir, f"{config['binary_name']}{binary_ext}")
        if not extracted:
            raise ValueError(
                f"Binary not found in archive: {config['binary_name']}{binary_ext}"
            )

        shutil.move(str(extracted), str(binary_path))

    archive_path.unlink(missing_ok=True)

    if sys.platform != "win32":
        binary_path.chmod(
            binary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )

    logger.info("Installed %s to %s", tool, binary_path)
    return str(binary_path)


def ensure_tool(tool: str, silent: bool = False) -> str | None:
    """确保工具可用，必要时自动下载。

    返回工具路径，若不可用则返回 None。
    """
    existing = get_tool_path(tool)
    if existing:
        return existing

    config = _TOOLS.get(tool)
    if not config:
        return None

    if _is_offline():
        if not silent:
            logger.warning(
                "%s not found. Offline mode enabled, skipping download.",
                config["binary_name"],
            )
        return None

    if not silent:
        logger.info("%s not found. Downloading...", config["binary_name"])

    try:
        path = _download_tool(tool)
        return path
    except Exception as exc:
        if not silent:
            logger.warning("Failed to download %s: %s", config["binary_name"], exc)
        return None
