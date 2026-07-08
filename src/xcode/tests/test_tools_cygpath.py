"""Cygwin 路径检测纯函数单元测试。"""

from __future__ import annotations

from xcode.coding_agent.tools.cygpath import (
    _find_cygpath,
    is_cygwin_env,
)


class TestFindCygpath:
    def test_cache_initialized(self) -> None:
        # Ensure module-level cache is a state we can observe
        result = _find_cygpath()
        # Should be str, None, or "" (cached miss)
        assert result is None or isinstance(result, str)


class TestIsCygwinEnv:
    def test_returns_bool(self) -> None:
        result = is_cygwin_env()
        assert isinstance(result, bool)
