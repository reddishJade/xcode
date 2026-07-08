"""Agent 运行结果纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.result import RunState, _parse_execution_mode


class TestRunState:
    def test_defaults(self) -> None:
        state = RunState(messages=[])
        assert state.current_mode == "act"
        assert state.last_agent == "main"
        assert not state.needs_follow_up

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        original = RunState(
            messages=[{"role": "user", "content": "hi"}],
            current_mode="build",
            last_agent="sub",
            needs_follow_up=True,
        )
        d = original.to_dict()
        restored = RunState.from_dict(d)
        assert restored.current_mode == "build"
        assert restored.last_agent == "sub"
        assert restored.needs_follow_up

    def test_from_dict_non_dict(self) -> None:
        state = RunState.from_dict("not a dict")
        assert state.messages == []

    def test_from_dict_invalid_mode_falls_back(self) -> None:
        state = RunState.from_dict({"current_mode": "unknown"})
        assert state.current_mode == "act"


class TestParseExecutionMode:
    def test_valid(self) -> None:
        assert _parse_execution_mode("plan") == "plan"
        assert _parse_execution_mode("build") == "build"
        assert _parse_execution_mode("act") == "act"

    def test_non_string_returns_none(self) -> None:
        assert _parse_execution_mode(123) is None

    def test_invalid_returns_none(self) -> None:
        assert _parse_execution_mode("invalid") is None
