"""截断文件的定期清理。

OutputAccumulator 溢出时创建 NamedTemporaryFile(delete=False)，
这些 .log 文件在进程退出后残留。本模块负责发现并清理
超过指定期限的旧文件。
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("xcode.coding_agent.tools.truncation_cleanup")

# 默认清理超过 7 天的文件
DEFAULT_MAX_AGE_DAYS: int = 7


def cleanup_truncation_files(
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    temp_prefix: str = "xcode-bash-",
    dry_run: bool = False,
) -> int:
    """清理超过 max_age_days 的 xcode 截断临时文件。

    Args:
        max_age_days: 文件最长保留天数。
        temp_prefix: 匹配的文件名前缀。
        dry_run: 仅扫描不删除。

    Returns:
        删除的文件数。
    """
    temp_dir = Path(tempfile.gettempdir())
    cutoff = time.time() - (max_age_days * 86400)
    removed = 0

    if not temp_dir.is_dir():
        logger.warning("temp directory not found: %s", temp_dir)
        return 0

    for entry in temp_dir.iterdir():
        if not entry.is_file():
            continue
        if not entry.name.startswith(temp_prefix):
            continue
        if not entry.name.endswith(".log"):
            continue

        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue

        if mtime < cutoff:
            if dry_run:
                logger.info("would remove stale truncation file: %s", entry)
            else:
                try:
                    entry.unlink(missing_ok=True)
                    logger.debug("removed stale truncation file: %s", entry)
                except OSError as exc:
                    logger.warning("failed to remove %s: %s", entry, exc)
                    continue
            removed += 1

    if removed:
        logger.info("cleaned up %d stale truncation file(s)", removed)
    return removed
