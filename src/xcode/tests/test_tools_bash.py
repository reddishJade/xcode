"""bash 工具请求解析纯函数单元测试。"""

from __future__ import annotations

import pytest
from xcode.coding_agent.tools.bash import (
    _parse_bash_request,
    _parse_timeout,
    _parse_workdir,
)


class TestParseBashRequest:
    def test_valid(self) -> None:
        request = _parse_bash_request({"command": "echo hi"})
        assert request.command == "echo hi"
        assert request.timeout > 0

    def test_missing_command_raises(self) -> None:
        with pytest.raises(ValueError, match="command"):
            _parse_bash_request({})


class TestParseTimeout:
    def test_default(self) -> None:
        assert _parse_timeout({}) == 30000

    def test_timeout_ms_priority(self) -> None:
        assert _parse_timeout({"timeout_ms": 5000}) == 5000

    def test_seconds_deprecated(self) -> None:
        assert _parse_timeout({"timeout": 60}) == 60000

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _parse_timeout({"timeout_ms": -1})

    def test_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="<= 300000"):
            _parse_timeout({"timeout_ms": 999999})

    def test_non_int_raises(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_timeout({"timeout_ms": "abc"})


class TestParseWorkdir:
    def test_none(self) -> None:
        assert _parse_workdir({}) is None

    def test_valid(self) -> None:
        assert _parse_workdir({"workdir": "src"}) == "src"

    def test_empty_string(self) -> None:
        assert _parse_workdir({"workdir": ""}) is None
