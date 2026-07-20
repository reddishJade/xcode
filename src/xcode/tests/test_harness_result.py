"""Agent 运行结果纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.result import RunState


class TestRunState:
    def test_defaults(self) -> None:
        state = RunState(messages=[])
        assert state.messages == []

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        original = RunState(
            messages=[{"role": "user", "content": "hi"}],
        )
        d = original.to_dict()
        restored = RunState.from_dict(d)
        assert restored.messages == original.messages

    def test_from_dict_non_dict(self) -> None:
        state = RunState.from_dict("not a dict")
        assert state.messages == []
