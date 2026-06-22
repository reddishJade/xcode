"""缓存优化单元测试。

测试缓存统计口径、工具 schema 规范化和 fingerprint 生成。
"""

from dataclasses import dataclass
from typing import Any

from xcode.ai.cache import CacheUsage, extract_cache_usage
from xcode.ai.providers.codec import (
    canonical_tool_schema,
    canonical_tools,
    tool_catalog_fingerprint,
)
from xcode.ai.types import ToolDefinition
import pytest
@dataclass
class MockUsage:
    """模拟 provider usage 对象。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None
    prompt_tokens_details: Any | None = None
    completion_tokens_details: Any | None = None

@dataclass
class MockResponse:
    """模拟 provider 响应对象。"""

    usage: MockUsage | None = None

@dataclass
class MockPromptTokensDetails:
    """模拟 prompt_tokens_details。"""

    cached_tokens: int = 0

class TestCacheUsage:
    """测试 CacheUsage 统计计算。"""

    def test_cache_usage_properties(self):
        """测试基本属性计算。"""
        usage = CacheUsage(hit_tokens=800, miss_tokens=200)
        assert usage.total_tokens == 1000
        assert usage.hit_rate == 0.8

    def test_cache_usage_zero_tokens(self):
        """测试零 token 情况。"""
        usage = CacheUsage()
        assert usage.total_tokens == 0
        assert usage.hit_rate == 0.0

    def test_cache_usage_perfect_hit(self):
        """测试 100% 命中。"""
        usage = CacheUsage(hit_tokens=1000, miss_tokens=0)
        assert usage.hit_rate == 1.0

    def test_cache_usage_no_hit(self):
        """测试 0% 命中。"""
        usage = CacheUsage(hit_tokens=0, miss_tokens=1000)
        assert usage.hit_rate == 0.0

class TestExtractCacheUsage:
    """测试从 provider 响应提取缓存统计。"""

    def test_extract_deepseek_native_fields(self):
        """测试 DeepSeek 原生字段优先。"""
        usage = MockUsage(
            prompt_tokens=1000,
            prompt_cache_hit_tokens=800,
            prompt_cache_miss_tokens=200,
        )
        response = MockResponse(usage=usage)
        result = extract_cache_usage(response)
        assert result.hit_tokens == 800
        assert result.miss_tokens == 200
        assert result.hit_rate == 0.8

    def test_extract_deepseek_native_hit_only(self):
        """测试只有原生 hit 时从 prompt_tokens 推算 miss。"""
        usage = MockUsage(
            prompt_tokens=1000,
            prompt_cache_hit_tokens=800,
            prompt_cache_miss_tokens=None,
        )
        response = MockResponse(usage=usage)
        result = extract_cache_usage(response)
        assert result.hit_tokens == 800
        assert result.miss_tokens == 200
        assert result.hit_rate == 0.8

    def test_extract_compat_cached_tokens(self):
        """测试兼容字段回退（ChatGLM/MiMo）。"""
        details = MockPromptTokensDetails(cached_tokens=600)
        usage = MockUsage(
            prompt_tokens=1000,
            prompt_tokens_details=details,
        )
        response = MockResponse(usage=usage)
        result = extract_cache_usage(response)
        assert result.hit_tokens == 600
        assert result.miss_tokens == 400
        assert result.hit_rate == 0.6

    def test_extract_no_cache_fields(self):
        """测试无缓存字段时返回空统计。"""
        usage = MockUsage(prompt_tokens=1000)
        response = MockResponse(usage=usage)
        result = extract_cache_usage(response)
        assert result.hit_tokens == 0
        assert result.miss_tokens == 0
        assert result.hit_rate == 0.0

    def test_extract_no_usage(self):
        """测试无 usage 对象时返回空统计。"""
        response = MockResponse(usage=None)
        result = extract_cache_usage(response)
        assert result.hit_tokens == 0
        assert result.miss_tokens == 0

class TestToolSchemaCanonical:
    """测试工具 schema 规范化。"""

    def test_canonical_tool_schema_sorts_keys(self):
        """测试 schema 字典键排序。"""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "z_param": {"type": "string"},
                    "a_param": {"type": "number"},
                },
            },
        )
        result = canonical_tool_schema(tool)
        # 检查顶级键顺序（按字母排序）
        assert list(result.keys()) == ["description", "name", "schema"]
        # 检查 properties 键顺序
        props = result["schema"]["properties"]
        assert list(props.keys()) == ["a_param", "z_param"]
        # 验证 schema 内部也排序
        assert list(result["schema"].keys()) == ["properties", "type"]

    def test_canonical_tool_schema_nested(self):
        """测试嵌套字典递归排序。"""
        tool = ToolDefinition(
            name="nested_tool",
            description="Nested schema",
            parameters={
                "z_top": {
                    "z_nested": "value",
                    "a_nested": "value",
                },
                "a_top": "value",
            },
        )
        result = canonical_tool_schema(tool)
        schema = result["schema"]
        assert list(schema.keys()) == ["a_top", "z_top"]
        assert list(schema["z_top"].keys()) == ["a_nested", "z_nested"]

    def test_canonical_tools_sorts_by_name(self):
        """测试工具列表按 name 排序。"""
        tools = [
            ToolDefinition(name="zebra", description="Z", parameters={}),
            ToolDefinition(name="apple", description="A", parameters={}),
            ToolDefinition(name="middle", description="M", parameters={}),
        ]
        result = canonical_tools(tools)
        assert [t["name"] for t in result] == ["apple", "middle", "zebra"]

class TestToolCatalogFingerprint:
    """测试工具集合指纹生成。"""

    def test_fingerprint_stable_same_tools(self):
        """测试相同工具生成相同指纹。"""
        tools1 = [
            ToolDefinition(
                name="tool_a", description="A", parameters={"type": "object"}
            ),
            ToolDefinition(
                name="tool_b", description="B", parameters={"type": "string"}
            ),
        ]
        tools2 = [
            ToolDefinition(
                name="tool_a", description="A", parameters={"type": "object"}
            ),
            ToolDefinition(
                name="tool_b", description="B", parameters={"type": "string"}
            ),
        ]
        fp1 = tool_catalog_fingerprint(tools1)
        fp2 = tool_catalog_fingerprint(tools2)
        assert fp1 == fp2
        assert len(fp1) == 16  # SHA256 前 16 字符

    def test_fingerprint_stable_different_order(self):
        """测试不同顺序生成相同指纹（排序后稳定）。"""
        tools1 = [
            ToolDefinition(
                name="tool_b", description="B", parameters={"type": "string"}
            ),
            ToolDefinition(
                name="tool_a", description="A", parameters={"type": "object"}
            ),
        ]
        tools2 = [
            ToolDefinition(
                name="tool_a", description="A", parameters={"type": "object"}
            ),
            ToolDefinition(
                name="tool_b", description="B", parameters={"type": "string"}
            ),
        ]
        fp1 = tool_catalog_fingerprint(tools1)
        fp2 = tool_catalog_fingerprint(tools2)
        assert fp1 == fp2

    def test_fingerprint_different_tools(self):
        """测试不同工具生成不同指纹。"""
        tools1 = [
            ToolDefinition(
                name="tool_a", description="A", parameters={"type": "object"}
            ),
        ]
        tools2 = [
            ToolDefinition(
                name="tool_b", description="B", parameters={"type": "string"}
            ),
        ]
        fp1 = tool_catalog_fingerprint(tools1)
        fp2 = tool_catalog_fingerprint(tools2)
        assert fp1 != fp2

    def test_fingerprint_schema_key_order_stable(self):
        """测试 schema 键顺序不影响指纹。"""
        tools1 = [
            ToolDefinition(
                name="tool",
                description="Test",
                parameters={"z_key": "z", "a_key": "a"},
            ),
        ]
        tools2 = [
            ToolDefinition(
                name="tool",
                description="Test",
                parameters={"a_key": "a", "z_key": "z"},
            ),
        ]
        fp1 = tool_catalog_fingerprint(tools1)
        fp2 = tool_catalog_fingerprint(tools2)
        assert fp1 == fp2

    def test_fingerprint_includes_builtin_tool_metadata(self) -> None:
        """测试内建工具元数据变化会改变指纹。"""
        tools1 = [
            ToolDefinition(
                name="shell",
                description="Run shell.",
                parameters={},
                builtin={"type": "shell", "environment": {"type": "local"}},
            ),
        ]
        tools2 = [
            ToolDefinition(
                name="shell",
                description="Run shell.",
                parameters={},
                builtin={"type": "web_search_preview"},
            ),
        ]
        fp1 = tool_catalog_fingerprint(tools1)
        fp2 = tool_catalog_fingerprint(tools2)
        assert fp1 != fp2

if __name__ == "__main__":
    pytest.main()
