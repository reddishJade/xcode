"""Token 估算和压缩触发测试。

测试基于真实 token 的压缩触发逻辑。
"""

from xcode.agent.compaction import (
    estimate_tokens,
    extract_prompt_tokens_from_usage,
    get_model_soft_threshold,
    should_compact_token_aware,
)
from xcode.agent.messages import UserMessage
from xcode.agent.types import TextContent
import pytest
class TestEstimateTokens:
    """测试 token 估算。"""

    def test_estimate_simple(self):
        """测试 tiktoken 估算。"""
        assert estimate_tokens("") == 1
        assert estimate_tokens("test") == 1
        assert estimate_tokens("test" * 10) == 10

    def test_extract_prompt_tokens(self):
        """测试从 usage 提取 prompt_tokens。"""
        usage = {"prompt_tokens": 1000, "completion_tokens": 200}
        assert extract_prompt_tokens_from_usage(usage) == 1000

    def test_extract_prompt_tokens_none(self):
        """测试 usage 为 None。"""
        assert extract_prompt_tokens_from_usage(None) is None
        assert extract_prompt_tokens_from_usage({}) is None

class TestModelSoftThreshold:
    """测试模型软阈值。"""

    def test_deepseek_threshold(self):
        """测试 DeepSeek 模型阈值。"""
        assert get_model_soft_threshold("deepseek-v4-flash") == 60000
        assert get_model_soft_threshold("deepseek-v3-pro") == 60000
        assert get_model_soft_threshold("deepseek-v2") == 28000

    def test_chatglm_threshold(self):
        """测试 ChatGLM 模型阈值。"""
        assert get_model_soft_threshold("glm-4.7") == 120000
        assert get_model_soft_threshold("glm-5") == 120000

    def test_mimo_threshold(self):
        """测试 MiMo 模型阈值。"""
        assert get_model_soft_threshold("mimo-v2.5-pro") == 60000

    def test_gpt_threshold(self):
        """测试 GPT 模型阈值。"""
        assert get_model_soft_threshold("gpt-4-turbo") == 120000
        assert get_model_soft_threshold("gpt-3.5-turbo") == 14000

    def test_default_threshold(self):
        """测试未知模型使用默认阈值。"""
        assert get_model_soft_threshold("unknown-model") == 32000
        assert get_model_soft_threshold(None) == 32000

class TestShouldCompactTokenAware:
    """测试 token-aware 压缩触发。"""

    def test_compact_by_real_tokens(self):
        """测试使用真实 token 触发压缩。"""
        messages = [UserMessage(content=[TextContent(text="test")])]
        # 真实 token 超过阈值
        assert should_compact_token_aware(
                messages,
                last_prompt_tokens=35000,
                model_soft_threshold=32000,
            )

    def test_no_compact_real_tokens_below(self):
        """测试真实 token 未超阈值不触发。"""
        messages = [UserMessage(content=[TextContent(text="test")])]
        assert not (should_compact_token_aware(
                messages,
                last_prompt_tokens=20000,
                model_soft_threshold=32000,
            ))

    def test_compact_by_message_count(self):
        """测试消息数阈值触发。"""
        messages = [UserMessage(content=[TextContent(text="test")]) for _ in range(10)]
        assert should_compact_token_aware(
                messages,
                compact_threshold=8,
            )

    def test_compact_by_estimated_tokens(self):
        """测试估算 token 阈值触发。"""
        messages = [
            UserMessage(content=[TextContent(text="x" * 1000)]) for _ in range(50)
        ]
        assert should_compact_token_aware(
                messages,
                compact_token_threshold=5000,
            )

    def test_no_compact_no_threshold(self):
        """测试所有阈值都为 0 不触发。"""
        messages = [UserMessage(content=[TextContent(text="test")]) for _ in range(100)]
        assert not (should_compact_token_aware(
                messages,
                compact_threshold=0,
                compact_token_threshold=0,
            ))

    def test_priority_real_tokens(self):
        """测试真实 token 优先级高于估算。"""
        # 消息数很多，但真实 token 未超标
        messages = [UserMessage(content=[TextContent(text="test")]) for _ in range(20)]
        assert not (should_compact_token_aware(
                messages,
                last_prompt_tokens=20000,
                model_soft_threshold=32000,
                compact_threshold=10,  # 消息数已超标
            ))

if __name__ == "__main__":
    pytest.main()
