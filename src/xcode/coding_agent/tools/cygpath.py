"""Cygwin/MSYS2 路径转换工具。

当 Xcode 运行在 Git Bash（MSYS2）环境下时，命令中出现的
POSIX 风格路径（/c/Users/name）需要转换为 Windows 风格
（C:/Users/name）才能被原生 Windows 工具识别。

参考 OpenCode: 调用 cygpath -w 做实时转换。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("xcode.coding_agent.tools.cygpath")


_CYGPATH_CACHE: str | None = None
"""缓存的 cygpath 可执行路径，None 表示未找到。"""


def _find_cygpath() -> str | None:
    """查找 cygpath 可执行文件。"""
    global _CYGPATH_CACHE
    if _CYGPATH_CACHE is not None:
        return _CYGPATH_CACHE
    for candidate in ("cygpath", "/usr/bin/cygpath"):
        resolved = shutil.which(candidate)
        if resolved:
            _CYGPATH_CACHE = resolved
            return resolved
    _CYGPATH_CACHE = ""
    return None


def is_cygwin_env() -> bool:
    """判断当前是否运行在 Cygwin/MSYS2 环境下。"""
    if sys.platform != "win32":
        return False
    # MSYS2: $MSYSTEM 环境变量存在
    if os.environ.get("MSYSTEM"):
        return True
    # Cygwin: cygpath 可用
    if _find_cygpath():
        return True
    return False


def to_windows(path: str) -> str:
    """将 Cygwin POSIX 路径转换为 Windows 路径。

    Args:
        path: 可能是 POSIX 风格的路径（/c/Users/name 或 /cygdrive/c/...）。

    Returns:
        Windows 风格路径（C:/Users/name），如果不能转换则返回原路径。
    """
    if not is_cygwin_env():
        return path
    if not path.startswith("/"):
        return path
    cygpath = _find_cygpath()
    if not cygpath:
        return path
    try:
        result = subprocess.run(
            [cygpath, "-w", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            converted = result.stdout.strip()
            if converted:
                return converted.replace("\\", "/")
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("cygpath -w %s failed: %s", path, exc)
    return path


def to_unix(path: str) -> str:
    """将 Windows 路径转换为 Cygwin POSIX 路径。

    Args:
        path: Windows 风格路径（C:/Users/name 或 C:\\Users\\name）。

    Returns:
        POSIX 风格路径（/c/Users/name），如果不能转换则返回原路径。
    """
    if not is_cygwin_env():
        return path
    normalized = Path(path).as_posix()
    # 检查是否为 Windows 驱动路径（如 C:/foo）
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        pass  # 是 Windows 绝对路径
    else:
        return path
    cygpath = _find_cygpath()
    if not cygpath:
        return path
    try:
        result = subprocess.run(
            [cygpath, "-u", normalized],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            converted = result.stdout.strip()
            if converted:
                return converted
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("cygpath -u %s failed: %s", path, exc)
    return path
