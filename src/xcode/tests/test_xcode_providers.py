from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typing import Any
from xcode.ai.events import TextDelta, ToolCallEvent
from xcode.ai.types import ToolDefinition
from dotenv import dotenv_values
from xcode.ai.providers.factory import (
    ProviderRuntime,
    ProviderSettings,
    RateLimitPolicy,
    RetryPolicy,
    build_provider_bundle,
    get_config_value,
    _resolve_api_key,
)
from xcode.ai.providers.openai import OpenAIChatProvider
from xcode.ai.providers.chatglm import ChatGLMProvider
from xcode.harness.config import ModelProfileRuntimeConfig


# ── 测试辅助：创建 mock OpenAI 客户端 ──


def _make_mock_client(chunks: list | None = None) -> MagicMock:
    """创建 mock openai.OpenAI 客户端，捕获请求参数。"""
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks or [])
    return client


# ── 流式 chunk 模拟对象 ──


class FakeStreamChunk:
    def __init__(self, content=None, tool_call=None, reasoning=None) -> None:
        self.choices = [FakeStreamChoice(content, tool_call, reasoning)]
        self.usage = None


class FakeStreamChoice:
    def __init__(self, content, tool_call, reasoning=None) -> None:
        self.delta = FakeStreamDelta(content, tool_call, reasoning)


class FakeStreamDelta:
    def __init__(self, content, tool_call, reasoning=None) -> None:
        self.content = content
        self.tool_calls = [tool_call] if tool_call is not None else []
        self.reasoning_content = reasoning


class FakeStreamToolCall:
    def __init__(self, index, call_id=None, name=None, arguments=None) -> None:
        self.index = index
        self.id = call_id
        self.function = FakeStreamFunction(name, arguments)


class FakeStreamFunction:
    def __init__(self, name, arguments) -> None:
        self.name = name
        self.arguments = arguments


# ── Env / Factory 测试 ──


class XcodeProviderEnvTests(unittest.TestCase):
    def test_env_file_loading_and_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("A=from_file\nQUOTED='value'\n", encoding="utf-8")

            self.assertEqual(dotenv_values(env_file)["A"], "from_file")
            self.assertEqual(get_config_value("QUOTED", (env_file,)), "value")

    def test_provider_factory_reads_env_and_builds_providers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=openai\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                bundle = build_provider_bundle(
                    ProviderSettings(
                        env_files=(env_file,),
                        model_profiles={
                            "main": ModelProfileRuntimeConfig(
                                chat_model="main-model",
                                base_url="https://main.test",
                                transport="openai_chat",
                            ),
                            "subagent": ModelProfileRuntimeConfig(
                                chat_model="small-model",
                                base_url="https://small.test",
                            ),
                        },
                    )
                )

                self.assertIsInstance(bundle.llm, OpenAIChatProvider)
                llm = bundle.llm
                assert isinstance(llm, OpenAIChatProvider)
                subagent = bundle.llms["subagent"]
                assert isinstance(subagent, OpenAIChatProvider)
                judge = bundle.llms["judge"]
                assert isinstance(judge, OpenAIChatProvider)
                self.assertEqual(llm.transport, "openai_chat")
                self.assertEqual(llm.model, "main-model")
                self.assertEqual(subagent.model, "small-model")
                self.assertEqual(judge.model, "main-model")

    def test_api_key_resolution_follows_documented_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "MAIN_API_KEY=profile\nOPENAI_API_KEY=openai\nAPI_KEY=generic\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(_resolve_api_key("", "main", (env_file,)), "profile")

            env_file.write_text(
                "DEEPSEEK_API_KEY=legacy\nOPENAI_API_KEY=openai\nAPI_KEY=generic\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(_resolve_api_key("", "main", (env_file,)), "openai")

            env_file.write_text(
                "DEEPSEEK_API_KEY=legacy\nAPI_KEY=generic\n", encoding="utf-8"
            )

            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(_resolve_api_key("", "main", (env_file,)), "generic")

    def test_provider_factory_builds_chatglm_from_provider_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("ZHIPUAI_API_KEY=glm-key\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                bundle = build_provider_bundle(
                    ProviderSettings(
                        env_files=(env_file,),
                        model_profiles={
                            "main": ModelProfileRuntimeConfig(
                                transport="chatglm_chat",
                                chat_model="glm-4.7",
                                base_url="",
                                reasoning_effort=None,
                                clear_thinking=True,
                                tool_stream=False,
                                response_format={"type": "json_object"},
                            ),
                        },
                    )
                )

                provider = bundle.llm
                self.assertIsInstance(provider, ChatGLMProvider)
                assert isinstance(provider, ChatGLMProvider)
                self.assertEqual(provider.client.api_key, "glm-key")
                self.assertEqual(provider.model, "glm-4.7")
                self.assertTrue(provider.clear_thinking)
                self.assertFalse(provider.tool_stream)
                self.assertEqual(provider.response_format, {"type": "json_object"})

    def test_explicit_profile_api_key_overrides_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "MAIN_API_KEY=profile\nOPENAI_API_KEY=openai\nAPI_KEY=generic\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"MAIN_API_KEY": "env-profile"}, clear=True):
                bundle = build_provider_bundle(
                    ProviderSettings(
                        env_files=(env_file,),
                        model_profiles={
                            "main": ModelProfileRuntimeConfig(
                                chat_model="main-model",
                                base_url="https://main.test",
                                api_key="configured",
                            ),
                        },
                    )
                )

                llm = bundle.llm
                assert isinstance(llm, OpenAIChatProvider)
                self.assertEqual(llm.client.api_key, "configured")

    def test_profile_api_key_overrides_openai_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "SUBAGENT_API_KEY=subagent\nOPENAI_API_KEY=openai\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                bundle = build_provider_bundle(
                    ProviderSettings(
                        env_files=(env_file,),
                        model_profiles={
                            "main": ModelProfileRuntimeConfig(
                                chat_model="main-model",
                                base_url="https://main.test",
                            ),
                            "subagent": ModelProfileRuntimeConfig(
                                chat_model="small-model",
                                base_url="https://small.test",
                            ),
                        },
                    )
                )

                main = bundle.llms["main"]
                assert isinstance(main, OpenAIChatProvider)
                subagent = bundle.llms["subagent"]
                assert isinstance(subagent, OpenAIChatProvider)
                self.assertEqual(main.client.api_key, "openai")
                self.assertEqual(subagent.client.api_key, "subagent")


# ── Runtime 测试 ──


class XcodeProviderRuntimeTests(unittest.TestCase):
    def test_retry_succeeds_after_transient_failure(self) -> None:
        calls = []
        runtime = ProviderRuntime(
            retry=RetryPolicy(max_attempts=3, initial_delay_seconds=0.1),
            sleeper=lambda _seconds: None,
        )

        def flaky() -> str:
            calls.append("call")
            if len(calls) == 1:
                raise RuntimeError("temporary")
            return "ok"

        self.assertEqual(runtime.run(flaky), "ok")
        self.assertEqual(len(calls), 2)

    def test_rate_limit_waits_between_calls(self) -> None:
        current = [10.0]
        sleeps = []
        runtime = ProviderRuntime(
            rate_limit=RateLimitPolicy(min_interval_seconds=1.0),
            now=lambda: current[0],
            sleeper=lambda seconds: sleeps.append(seconds),
        )

        runtime.run(lambda: "first")
        current[0] = 10.2
        runtime.run(lambda: "second")

        self.assertEqual(sleeps, [0.8000000000000007])


# ── OpenAI Chat 测试 ──


class XcodeStructuredProviderTests(unittest.TestCase):
    def test_stream_converts_tool_schema_and_tool_calls(self) -> None:
        client = _make_mock_client(
            [
                FakeStreamChunk(
                    tool_call=FakeStreamToolCall(
                        index=0,
                        call_id="call-1",
                        name="echo",
                        arguments='{"text": "hello"}',
                    )
                ),
            ]
        )
        llm = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.test/v1",
            model="model",
            thinking=True,
            reasoning_effort=None,
            runtime=ProviderRuntime(),
            client=client,
        )
        tool = ToolDefinition(
            name="echo",
            description="Echo input.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

        events = list(
            llm._stream_sync(
                [{"role": "user", "content": "echo"}],
                (tool,),
            )
        )
        tool_call = events[-1]
        self.assertIsInstance(tool_call, ToolCallEvent)
        assert isinstance(tool_call, ToolCallEvent)
        self.assertEqual(tool_call.calls[0].name, "echo")
        self.assertEqual(tool_call.calls[0].input, {"text": "hello"})
        sent_tool = client.chat.completions.create.call_args.kwargs["tools"][0]
        self.assertEqual(sent_tool["function"]["parameters"], tool.parameters)

    def test_stream_converts_tool_results_to_openai_messages(self) -> None:
        client = _make_mock_client([FakeStreamChunk(content="done")])
        llm = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.test/v1",
            model="model",
            thinking=True,
            reasoning_effort=None,
            runtime=ProviderRuntime(),
            client=client,
        )

        events = list(
            llm._stream_sync(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "echo",
                                "input": {"text": "hi"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "hi",
                            }
                        ],
                    },
                ],
                (),
            )
        )

        sent_messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent_messages[0]["role"], "assistant")
        self.assertEqual(sent_messages[1]["role"], "tool")
        self.assertEqual(
            [e for e in events if isinstance(e, TextDelta)][0].chunk, "done"
        )

    def test_stream_yields_text_and_tool_call_deltas(self) -> None:
        client = _make_mock_client(
            [
                FakeStreamChunk(content="he"),
                FakeStreamChunk(content="llo"),
                FakeStreamChunk(
                    tool_call=FakeStreamToolCall(
                        index=0,
                        call_id="call-1",
                        name="echo",
                        arguments='{"text": ',
                    )
                ),
                FakeStreamChunk(
                    tool_call=FakeStreamToolCall(
                        index=0,
                        arguments='"hi"}',
                    )
                ),
            ]
        )
        llm = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.test/v1",
            model="model",
            thinking=True,
            reasoning_effort=None,
            runtime=ProviderRuntime(),
            client=client,
        )

        events = list(llm._stream_sync([{"role": "user", "content": "go"}], ()))

        self.assertIsInstance(events[0], TextDelta)
        assert isinstance(events[0], TextDelta)
        self.assertEqual(events[0].chunk, "he")
        self.assertIsInstance(events[1], TextDelta)
        assert isinstance(events[1], TextDelta)
        self.assertEqual(events[1].chunk, "llo")
        self.assertIsInstance(events[-1], ToolCallEvent)
        assert isinstance(events[-1], ToolCallEvent)
        self.assertEqual(events[-1].calls[0].id, "call-1")
        self.assertEqual(events[-1].calls[0].name, "echo")
        self.assertEqual(events[-1].calls[0].input, {"text": "hi"})
        self.assertTrue(client.chat.completions.create.call_args.kwargs["stream"])


# ── ChatGLM 测试 ──


def _make_glm_provider(client: Any = None, **overrides: Any) -> ChatGLMProvider:
    kwargs: dict[str, Any] = dict(
        api_key="test-key",
        model="glm-4-flash",
        thinking=True,
        clear_thinking=False,
        tool_stream=True,
    )
    if client is not None:
        kwargs["client"] = client
    kwargs.update(overrides)
    return ChatGLMProvider(**kwargs)


class XcodeChatGLMProviderTests(unittest.TestCase):
    """ChatGLM provider 边界测试：thinking 清理、tool_stream、参数组合。"""

    def test_thinking_disabled_sets_extra_body(self) -> None:
        """thinking=False 时 extra_body 为 disabled。"""
        client = _make_mock_client()
        provider = _make_glm_provider(client=client, thinking=False)
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        extra = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        self.assertEqual(extra.get("thinking", {}).get("type"), "disabled")

    def test_uses_openai_compatible_model_and_credentials(self) -> None:
        """ChatGLM 参数正确传递到 OpenAI 客户端。"""
        client = _make_mock_client()
        provider = _make_glm_provider(
            client=client,
            api_key="glm-key",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )

        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "glm-4-flash")

    def test_thinking_enabled_clear_false(self) -> None:
        """clear_thinking=False 时 extra_body 包含 clear_thinking=false。"""
        client = _make_mock_client()
        provider = _make_glm_provider(client=client, clear_thinking=False)
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        extra = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        thinking = extra.get("thinking", {})
        self.assertEqual(thinking.get("type"), "enabled")
        self.assertIs(thinking.get("clear_thinking"), False)

    def test_thinking_enabled_clear_true(self) -> None:
        """clear_thinking=True 时 extra_body 包含 clear_thinking=true。"""
        client = _make_mock_client()
        provider = _make_glm_provider(client=client, clear_thinking=True)
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        extra = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        thinking = extra.get("thinking", {})
        self.assertEqual(thinking.get("type"), "enabled")
        self.assertIs(thinking.get("clear_thinking"), True)

    def test_turn_level_thinking_override_is_per_request(self) -> None:
        """单次请求可覆盖 thinking，不改变 provider 默认值。"""
        client = _make_mock_client()
        provider = _make_glm_provider(client=client, thinking=True)

        list(
            provider._stream_sync(
                [{"role": "user", "content": "quick"}],
                (),
                thinking=False,
            )
        )
        first = client.chat.completions.create.call_args.kwargs["extra_body"][
            "thinking"
        ]
        self.assertEqual(first["type"], "disabled")
        self.assertTrue(provider.thinking)

        list(provider._stream_sync([{"role": "user", "content": "hard"}], ()))
        second = client.chat.completions.create.call_args.kwargs["extra_body"][
            "thinking"
        ]
        self.assertEqual(second["type"], "enabled")

    def test_structured_output_response_format_is_sent(self) -> None:
        """结构化输出透传 response_format。"""
        client = _make_mock_client()
        provider = _make_glm_provider(
            client=client, response_format={"type": "json_object"}
        )
        list(provider._stream_sync([{"role": "user", "content": "json"}], ()))

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_tool_stream_disabled(self) -> None:
        """tool_stream=False 时不传 tool_stream 参数。"""
        client = _make_mock_client()
        provider = _make_glm_provider(client=client, tool_stream=False)
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        extra = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        self.assertNotIn("tool_stream", extra)

    def test_tool_stream_enabled_for_supported_model(self) -> None:
        """支持的模型开启 tool_stream 时传 tool_stream=true。"""
        client = _make_mock_client()
        provider = _make_glm_provider(client=client, model="glm-4.7", tool_stream=True)
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        extra = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        self.assertIs(extra.get("tool_stream"), True)

    def test_tool_stream_omitted_for_unsupported_model(self) -> None:
        """不支持的模型不传 tool_stream 参数。"""
        client = _make_mock_client()
        provider = _make_glm_provider(
            client=client, model="glm-4-flash", tool_stream=True
        )
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        extra = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        self.assertNotIn("tool_stream", extra)

    def test_clean_reasoning_no_tool_loop(self) -> None:
        """clear_thinking=True 时清除所有历史 reasoning_content。"""
        provider = _make_glm_provider(clear_thinking=True)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "think1"},
            {"role": "user", "content": "q2"},
        ]
        cleaned = provider._clean_reasoning_content(messages)
        for msg in cleaned:
            self.assertNotIn("reasoning_content", msg)

    def test_clean_reasoning_retains_all_when_clear_false(self) -> None:
        """clear_thinking=False 时保留所有 reasoning_content。"""
        provider = _make_glm_provider(clear_thinking=False)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "think1"},
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": "a2",
                "reasoning_content": "think2",
                "tool_calls": [],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
        cleaned = provider._clean_reasoning_content(messages)
        self.assertEqual(cleaned[1]["reasoning_content"], "think1")
        self.assertEqual(cleaned[3]["reasoning_content"], "think2")

    def test_clean_reasoning_clear_true_removes_tool_loop_reasoning(self) -> None:
        """clear_thinking=True 时工具循环也清除 reasoning_content。"""
        provider = _make_glm_provider(clear_thinking=True)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "think1"},
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "think2",
                "tool_calls": [
                    {"id": "t1", "function": {"name": "echo", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
        cleaned = provider._clean_reasoning_content(messages)
        self.assertNotIn("reasoning_content", cleaned[1])
        self.assertNotIn("reasoning_content", cleaned[3])

    def test_thinking_true_streams_reasoning(self) -> None:
        """thinking=True 时流包含 reasoning delta。"""
        from xcode.ai.events import ReasoningDelta

        client = _make_mock_client(
            [
                FakeStreamChunk(content="hello", reasoning="thinking..."),
                FakeStreamChunk(content=" world"),
            ]
        )
        provider = _make_glm_provider(client=client)
        events = list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        reasoning_events = [e for e in events if isinstance(e, ReasoningDelta)]
        self.assertEqual(len(reasoning_events), 1)
        self.assertEqual(reasoning_events[0].chunk, "thinking...")

    def test_combo_tool_stream_plus_clear_thinking(self) -> None:
        """tool_stream=True + clear_thinking=True 组合参数正确传递。"""
        client = _make_mock_client()
        provider = _make_glm_provider(
            client=client,
            model="glm-4.7",
            tool_stream=True,
            clear_thinking=True,
        )
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        extra = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        self.assertIs(extra["tool_stream"], True)
        self.assertIs(extra["thinking"]["clear_thinking"], True)

    def test_record_usage_cached_and_reasoning_tokens(self) -> None:
        """usage 统计记录 cached_tokens 和 reasoning_tokens。"""
        from collections import namedtuple

        FakeUsage = namedtuple(
            "FakeUsage",
            [
                "prompt_tokens",
                "completion_tokens",
                "prompt_tokens_details",
                "completion_tokens_details",
            ],
        )
        FakePromptDetails = namedtuple("FakePromptDetails", ["cached_tokens"])
        FakeCompletionDetails = namedtuple(
            "FakeCompletionDetails", ["reasoning_tokens"]
        )

        class FakeUsageResponse:
            usage = FakeUsage(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_tokens_details=FakePromptDetails(cached_tokens=20),
                completion_tokens_details=FakeCompletionDetails(reasoning_tokens=10),
            )

        provider = _make_glm_provider()
        provider._record_usage(FakeUsageResponse(), sent_messages=2)
        self.assertEqual(provider.metrics["prompt_tokens"], 100)
        self.assertEqual(provider.metrics["completion_tokens"], 50)
        self.assertEqual(provider.metrics["total_tokens"], 150)
        self.assertEqual(provider.metrics["cached_tokens"], 20)
        self.assertEqual(provider.metrics["cache_hit_rate"], 0.2)
        self.assertEqual(provider.metrics["reasoning_tokens"], 10)
        self.assertEqual(provider.metrics["sent_messages"], 2)

    def test_record_usage_none_details_degradation(self) -> None:
        """prompt_tokens_details / completion_tokens_details 为 None 时降级为 0。"""
        from collections import namedtuple

        FakeUsage = namedtuple(
            "FakeUsage",
            [
                "prompt_tokens",
                "completion_tokens",
                "prompt_tokens_details",
                "completion_tokens_details",
            ],
        )

        class FakeUsageNoDetails:
            usage = FakeUsage(
                prompt_tokens=50,
                completion_tokens=30,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            )

        provider = _make_glm_provider()
        provider._record_usage(FakeUsageNoDetails(), sent_messages=1)
        self.assertEqual(provider.metrics["cached_tokens"], 0)
        self.assertEqual(provider.metrics["reasoning_tokens"], 0)

    def test_transport_is_chatglm(self) -> None:
        provider = _make_glm_provider()
        self.assertEqual(provider.transport, "chatglm_chat")


if __name__ == "__main__":
    unittest.main()
