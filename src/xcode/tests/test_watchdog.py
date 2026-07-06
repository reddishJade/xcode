"""重复工具调用检测测试。

测试带文件变更感知的重复工具调用抑制。
"""

from xcode.agent.config import AgentLoopConfig, _LoopRunState
from xcode.agent.messages import ToolResultMessage
from xcode.agent.types import ToolCallContent
from xcode.agent.watchdog import (
    is_file_mutation_tool,
    is_file_read_tool,
    should_clear_read_history,
    tool_call_signature,
    tool_calls_signature,
    update_repeated_tool_watchdog,
)
import pytest


class TestToolSignature:
    """测试工具签名生成。"""

    def test_single_call_signature(self):
        """测试单个工具调用签名。"""
        call = ToolCallContent(
            id="call_1",
            name="read_file",
            arguments={"path": "/test/file.txt"},
        )
        sig = tool_call_signature(call)
        assert "read_file" in sig
        assert "/test/file.txt" in sig

    def test_signature_stable_same_args(self):
        """测试相同参数生成相同签名。"""
        call1 = ToolCallContent(
            id="call_1",
            name="test",
            arguments={"a": 1, "b": 2},
        )
        call2 = ToolCallContent(
            id="call_2",
            name="test",
            arguments={"b": 2, "a": 1},  # 参数顺序不同
        )
        assert tool_call_signature(call1) == tool_call_signature(call2)

    def test_batch_signature_order_independent(self):
        """测试批次签名与调用顺序无关。"""
        calls1 = [
            ToolCallContent(id="1", name="tool_a", arguments={}),
            ToolCallContent(id="2", name="tool_b", arguments={}),
        ]
        calls2 = [
            ToolCallContent(id="3", name="tool_b", arguments={}),
            ToolCallContent(id="4", name="tool_a", arguments={}),
        ]
        assert tool_calls_signature(calls1) == tool_calls_signature(calls2)


class TestToolClassification:
    """测试工具分类。"""

    def test_file_mutation_tools(self):
        """测试文件变更工具识别。"""
        assert is_file_mutation_tool("write_file")
        assert is_file_mutation_tool("edit_file")
        assert is_file_mutation_tool("bash")
        assert not (is_file_mutation_tool("read_file"))

    def test_file_read_tools(self):
        """测试只读工具识别。"""
        assert is_file_read_tool("read_file")
        assert is_file_read_tool("grep_search")
        assert is_file_read_tool("glob_files")
        assert not (is_file_read_tool("write_file"))

    def test_should_clear_read_history(self):
        """测试是否应清除只读历史。"""
        read_calls = [
            ToolCallContent(id="1", name="read_file", arguments={}),
        ]
        write_calls = [
            ToolCallContent(id="2", name="write_file", arguments={}),
        ]
        # 只读调用不清除历史
        assert not (should_clear_read_history(read_calls, []))
        # 写入调用清除历史
        assert should_clear_read_history(write_calls, [])


class TestWatchdogInterference:
    """测试重复看门狗与空闲看门狗的协作规则。

    修复：当连续相同工具调用全部失败时，重复看门狗不应触发，
    应留给空闲看门狗处理，避免 "repeated tool call" 掩盖根因。
    """

    def _make_call(
        self, name: str = "write_file", path: str = "/test/file.txt"
    ) -> ToolCallContent:
        return ToolCallContent(id="call_1", name=name, arguments={"path": path})

    def _make_result(self, is_error: bool) -> ToolResultMessage:
        return ToolResultMessage(
            tool_call_id="call_1",
            tool_name="write_file",
            content="error" if is_error else "ok",
            is_error=is_error,
        )

    def _make_config(self) -> AgentLoopConfig:
        return AgentLoopConfig(watchdog_repeated_tool_limit=3)

    # ── 重复但成功：正常触发 ──

    def test_repeated_successful_calls_triggers_watchdog(self):
        """连续调用同一工具且成功时，重复看门狗正常触发。"""
        state = _LoopRunState()
        config = self._make_config()
        call = self._make_call()
        result = self._make_result(is_error=False)

        # 第一次调用→建立签名
        assert update_repeated_tool_watchdog(state, [call], config, [result]) is None
        assert state.repeated_tool_count == 0

        # 第二次重复→计数增加
        assert update_repeated_tool_watchdog(state, [call], config, [result]) is None
        assert state.repeated_tool_count == 1

        # 第三次重复→计数增加
        assert update_repeated_tool_watchdog(state, [call], config, [result]) is None
        assert state.repeated_tool_count == 2

        # 第四次重复→触发看门狗
        reason = update_repeated_tool_watchdog(state, [call], config, [result])
        assert reason is not None
        assert "repeated tool call" in reason

    # ── 重复但全部失败：不触发，留给空闲看门狗 ──

    def test_repeated_all_errors_does_not_trigger_watchdog(self):
        """连续调用同一工具且全部失败时，重复看门狗不触发。"""
        state = _LoopRunState()
        config = self._make_config()
        call = self._make_call()
        error_result = self._make_result(is_error=True)

        # 多次连续失败调用，重复计数始终为 0
        for _ in range(5):
            reason = update_repeated_tool_watchdog(
                state, [call], config, [error_result]
            )
            assert reason is None, f"全部失败不应触发重复看门狗: {reason}"
            assert state.repeated_tool_count == 0

    def test_all_errors_resets_accumulated_count(self):
        """如果之前已有重复计数，全失败调用应重置计数。"""
        state = _LoopRunState()
        config = self._make_config()
        call = self._make_call()
        ok_result = self._make_result(is_error=False)
        error_result = self._make_result(is_error=True)

        # 先两次成功重复→计数到 1
        assert update_repeated_tool_watchdog(state, [call], config, [ok_result]) is None
        assert update_repeated_tool_watchdog(state, [call], config, [ok_result]) is None
        assert state.repeated_tool_count == 1

        # 然后一次失败→重置计数
        assert (
            update_repeated_tool_watchdog(state, [call], config, [error_result]) is None
        )
        assert state.repeated_tool_count == 0
        assert state.last_tool_signature is None

    # ── 参数变化重置计数 ──

    def test_diff_args_resets_count_even_on_error(self):
        """参数变化重置重复计数，即使上次结果失败。"""
        state = _LoopRunState()
        config = self._make_config()
        call_a = self._make_call(path="/a.txt")
        call_b = self._make_call(path="/b.txt")
        ok_result = self._make_result(is_error=False)

        assert (
            update_repeated_tool_watchdog(state, [call_a], config, [ok_result]) is None
        )
        assert (
            update_repeated_tool_watchdog(state, [call_a], config, [ok_result]) is None
        )
        assert state.repeated_tool_count == 1

        # 参数变化→重置
        assert (
            update_repeated_tool_watchdog(state, [call_b], config, [ok_result]) is None
        )
        assert state.repeated_tool_count == 0

    # ── 部分失败（混合结果）→ 重复看门狗仍触发 ──

    def test_mixed_results_triggers_watchdog(self):
        """混合结果（部分成功部分失败）时，重复看门狗正常触发。"""
        state = _LoopRunState()
        config = self._make_config()
        call = self._make_call()
        results = [
            self._make_result(is_error=False),  # 成功
            self._make_result(is_error=True),  # 失败
        ]

        # 三次重复，每次有部分成功
        assert update_repeated_tool_watchdog(state, [call], config, results) is None
        assert state.repeated_tool_count == 0
        assert update_repeated_tool_watchdog(state, [call], config, results) is None
        assert state.repeated_tool_count == 1
        assert update_repeated_tool_watchdog(state, [call], config, results) is None
        assert state.repeated_tool_count == 2
        reason = update_repeated_tool_watchdog(state, [call], config, results)
        assert reason is not None
        assert "repeated tool call" in reason


if __name__ == "__main__":
    pytest.main()
