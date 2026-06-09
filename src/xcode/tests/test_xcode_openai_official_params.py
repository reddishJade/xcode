from __future__ import annotations

import asyncio
from dataclasses import dataclass
import unittest
from typing import Any, cast

from xcode.agent.messages import UserMessage, convert_to_llm
from xcode.agent.types import FileContent, ImageContent, TextContent
from xcode.ai.cache import tool_catalog_fingerprint
from xcode.ai.events import FinalMessage, ReasoningDelta, TextDelta, ToolCallEvent
from xcode.ai.providers.codec import to_responses_input, to_responses_tool
from xcode.ai.providers.factory import _build_llm_profile
from xcode.ai.providers.openai import OpenAIChatProvider, OpenAIResponsesProvider
from xcode.ai.providers.runtime import ProviderRuntime
from xcode.ai.providers.stream_codec import responses_stream_to_events
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.agent_runtime.prompting import SYSTEM_PROMPT_DYNAMIC_BOUNDARY


class XcodeOpenAIOfficialParamsTests(unittest.TestCase):
    """OpenAI 官方 API 参数边界测试。"""

    def test_chat_provider_does_not_send_provider_specific_thinking_body(self) -> None:
        """OpenAI Chat 不发送兼容 provider 专属 extra_body.thinking。"""
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            thinking=True,
            reasoning_effort="high",
            client=client,
        )

        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))

        kwargs = client.chat.completions.kwargs
        self.assertNotIn("extra_body", kwargs)
        self.assertEqual(kwargs["reasoning_effort"], "high")

    def test_chat_provider_maps_disabled_thinking_to_none_effort(self) -> None:
        """thinking 关闭时使用官方 reasoning_effort=none。"""
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            thinking=False,
            reasoning_effort=None,
            client=client,
        )

        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))

        kwargs = client.chat.completions.kwargs
        self.assertNotIn("extra_body", kwargs)
        self.assertEqual(kwargs["reasoning_effort"], "none")

    def test_responses_provider_applies_stream_options(self) -> None:
        """Responses 公共入口透传请求级选项。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )
            options = StreamOptions(
                api_key="override-key",
                headers={"x-extra": "value"},
                session_id="session-1",
                metadata={"task": "unit"},
                max_tokens=123,
                temperature=0.2,
                timeout_ms=2500,
            )

            events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}], [], options=options
                )
            ]

            kwargs = client.responses.kwargs
            self.assertEqual(events, [])
            self.assertEqual(client.override_api_key, "override-key")
            self.assertEqual(kwargs["extra_headers"]["x-session-id"], "session-1")
            self.assertEqual(kwargs["extra_headers"]["x-extra"], "value")
            self.assertEqual(kwargs["metadata"], {"task": "unit"})
            self.assertEqual(kwargs["max_output_tokens"], 123)
            self.assertEqual(kwargs["temperature"], 0.2)
            self.assertEqual(kwargs["timeout"], 2.5)

        asyncio.run(run_test())

    def test_chat_provider_sends_configured_response_format(self) -> None:
        """OpenAI Chat 使用 Chat Completions 的 response_format 字段。"""
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            response_format={"type": "json_object"},
            client=client,
        )

        list(provider._stream_sync([{"role": "user", "content": "json"}], ()))

        self.assertEqual(
            client.chat.completions.kwargs["response_format"],
            {"type": "json_object"},
        )

    def test_chat_provider_warns_for_builtin_tools(self) -> None:
        """Chat Completions 遇到 Responses 内建工具时记录 warning。"""
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            client=client,
        )
        tool = ToolDefinition(
            name="shell",
            description="Run shell commands.",
            schema={},
            builtin={"type": "shell", "environment": {"type": "local"}},
        )

        with self.assertLogs("xcode.ai.providers.openai", level="WARNING") as logs:
            list(provider._stream_sync([{"role": "user", "content": "hi"}], (tool,)))

        self.assertIn(
            "OpenAI Chat Completions does not support builtin tool",
            logs.output[0],
        )

    def test_chat_provider_keeps_system_prompt_in_messages(self) -> None:
        """Chat Completions 保留完整 system 消息。"""
        client = FakeOpenAIClient()
        provider = OpenAIChatProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            client=client,
        )
        system_prompt = (
            "Stable instructions."
            f"\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\n"
            "<environment>dynamic</environment>"
        )

        list(
            provider._stream_sync(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "hi"},
                ],
                (),
            )
        )

        messages = client.chat.completions.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], system_prompt)
        self.assertNotIn("instructions", client.chat.completions.kwargs)

    def test_responses_provider_maps_response_format_to_text_config(self) -> None:
        """Responses 使用 text.format 承载结构化输出配置。"""
        client = FakeOpenAIClient()
        provider = OpenAIResponsesProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    "strict": True,
                },
            },
            client=client,
        )

        list(provider._stream_sync([{"role": "user", "content": "json"}], ()))

        text_config = client.responses.kwargs["text"]
        self.assertEqual(text_config["format"]["type"], "json_schema")
        self.assertEqual(text_config["format"]["name"], "answer")
        self.assertTrue(text_config["format"]["strict"])

    def test_responses_provider_moves_stable_prompt_to_instructions(self) -> None:
        """Responses 将稳定 prompt 前缀移入 instructions 参数。"""
        client = FakeOpenAIClient()
        provider = OpenAIResponsesProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            client=client,
        )
        system_prompt = (
            "Stable instructions."
            f"\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\n"
            "<environment>dynamic</environment>"
        )

        list(
            provider._stream_sync(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "hi"},
                ],
                (),
            )
        )

        kwargs = client.responses.kwargs
        self.assertEqual(kwargs["instructions"], "Stable instructions.")
        self.assertEqual(kwargs["input"][0]["role"], "developer")
        self.assertEqual(
            kwargs["input"][0]["content"],
            [
                {
                    "type": "input_text",
                    "text": "<environment>dynamic</environment>",
                }
            ],
        )
        self.assertNotIn("Stable instructions.", str(kwargs["input"]))

    def test_responses_previous_response_id_keeps_current_instructions(self) -> None:
        """previous_response_id 后仍发送当前请求 instructions。"""
        client = FakeOpenAIClient(
            response_outputs=[
                [
                    FakeResponsesStreamEvent(
                        "response.completed",
                        response=FakeResponsesResponse(response_id="r1"),
                    )
                ],
                [
                    FakeResponsesStreamEvent(
                        "response.completed",
                        response=FakeResponsesResponse(response_id="r2"),
                    )
                ],
            ]
        )
        provider = OpenAIResponsesProvider(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-5.4",
            client=client,
        )
        system_prompt = (
            "Stable instructions."
            f"\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\n"
            "<environment>dynamic</environment>"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "one"},
        ]

        list(provider._stream_sync(messages, ()))
        messages.extend(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "two"},
            ]
        )
        list(provider._stream_sync(messages, ()))

        second_call = client.responses.calls[1]
        self.assertEqual(second_call["previous_response_id"], "r1")
        self.assertEqual(second_call["instructions"], "Stable instructions.")
        self.assertEqual(
            second_call["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "two"}],
                }
            ],
        )

    def test_responses_reset_conversation_state_clears_server_cursor(self) -> None:
        """清理会话状态后不再发送 previous_response_id。"""

        async def run_test() -> None:
            client = FakeOpenAIClient(
                response_outputs=[
                    [
                        FakeResponsesStreamEvent(
                            "response.completed",
                            response=FakeResponsesResponse(response_id="r1"),
                        )
                    ],
                    [],
                ]
            )
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )

            _first = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "first"}], []
                )
            ]
            provider.reset_conversation_state()
            _second = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "second"}], []
                )
            ]

            second_call = client.responses.calls[1]
            self.assertNotIn("previous_response_id", second_call)
            self.assertEqual(second_call["input"][0]["content"][0]["text"], "second")
            self.assertIsNone(provider.metrics["previous_response_id"])

        asyncio.run(run_test())

    def test_factory_preserves_openai_response_format(self) -> None:
        """factory 构建 OpenAI provider 时保留结构化输出配置。"""
        provider = _build_llm_profile(
            MockProfile(transport="openai_responses"),
            profile_name="main",
            env_files=(),
            runtime=ProviderRuntime(),
        )

        self.assertIsInstance(provider, OpenAIResponsesProvider)
        assert isinstance(provider, OpenAIResponsesProvider)
        self.assertEqual(provider.response_format, {"type": "json_object"})

    def test_responses_provider_applies_extended_response_options(self) -> None:
        """Responses 透传官方请求级控制参数。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                response_format={"type": "json_object"},
                client=client,
            )
            options = StreamOptions(
                background=True,
                include=["reasoning.encrypted_content"],
                instructions="Answer briefly.",
                max_tool_calls=2,
                parallel_tool_calls=False,
                prompt_cache_retention="24h",
                safety_identifier="user-1",
                service_tier="flex",
                store=False,
                tool_choice="auto",
                top_logprobs=3,
                top_p=0.8,
                truncation="auto",
                user="end-user",
                verbosity="low",
                response_extra_params={"custom_beta": "value", "store": True},
            )

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}], [], options=options
                )
            ]

            kwargs = client.responses.kwargs
            self.assertIs(kwargs["background"], True)
            self.assertEqual(kwargs["include"], ["reasoning.encrypted_content"])
            self.assertEqual(kwargs["instructions"], "Answer briefly.")
            self.assertEqual(kwargs["max_tool_calls"], 2)
            self.assertIs(kwargs["parallel_tool_calls"], False)
            self.assertEqual(kwargs["prompt_cache_retention"], "24h")
            self.assertEqual(kwargs["safety_identifier"], "user-1")
            self.assertEqual(kwargs["service_tier"], "flex")
            self.assertIs(kwargs["store"], False)
            self.assertEqual(kwargs["tool_choice"], "auto")
            self.assertEqual(kwargs["top_logprobs"], 3)
            self.assertEqual(kwargs["top_p"], 0.8)
            self.assertEqual(kwargs["truncation"], "auto")
            self.assertEqual(kwargs["user"], "end-user")
            self.assertEqual(kwargs["text"]["format"], {"type": "json_object"})
            self.assertEqual(kwargs["text"]["verbosity"], "low")
            self.assertEqual(kwargs["custom_beta"], "value")

        asyncio.run(run_test())

    def test_responses_provider_counts_input_tokens(self) -> None:
        """Responses 调用官方 input_tokens 计数接口并记录 metrics。"""

        async def run_test() -> None:
            client = FakeOpenAIClient(input_token_count=42)
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                response_format={"type": "json_object"},
                client=client,
            )
            tool = ToolDefinition(
                name="lookup",
                description="Lookup records.",
                schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
            options = StreamOptions(
                instructions="Use files carefully.",
                max_tokens=200,
                parallel_tool_calls=False,
                truncation="auto",
                timeout_ms=1500,
            )

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}],
                    [tool],
                    options=options,
                )
            ]

            kwargs = client.responses.input_tokens.kwargs
            self.assertEqual(provider.metrics["input_tokens"], 42)
            self.assertEqual(kwargs["model"], "gpt-5.4")
            self.assertEqual(kwargs["instructions"], "Use files carefully.")
            self.assertIs(kwargs["parallel_tool_calls"], False)
            self.assertEqual(kwargs["truncation"], "auto")
            self.assertEqual(kwargs["timeout"], 1.5)
            self.assertEqual(kwargs["text"]["format"], {"type": "json_object"})
            self.assertEqual(kwargs["tools"][0]["name"], "lookup")
            self.assertNotIn("stream", kwargs)
            self.assertNotIn("max_output_tokens", kwargs)

        asyncio.run(run_test())

    def test_responses_provider_retrieves_completed_background_response(self) -> None:
        """Responses retrieve 轮询完成响应并转为最终消息。"""

        async def run_test() -> None:
            client = FakeOpenAIClient(
                retrieve_responses=[
                    FakeResponsesResponse(
                        response_id="resp_bg",
                        output_text="done",
                        status="completed",
                    )
                ]
            )
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )
            options = StreamOptions(
                headers={"x-extra": "value"},
                session_id="session-1",
                timeout_ms=2500,
            )

            events = [
                event
                async for event in provider.retrieve_response(
                    "resp_bg",
                    include=["reasoning.encrypted_content"],
                    options=options,
                )
            ]

            kwargs = client.responses.retrieve_kwargs
            self.assertEqual(client.responses.retrieve_response_id, "resp_bg")
            self.assertIs(kwargs["stream"], False)
            self.assertEqual(kwargs["include"], ["reasoning.encrypted_content"])
            self.assertEqual(kwargs["extra_headers"]["x-session-id"], "session-1")
            self.assertEqual(kwargs["extra_headers"]["x-extra"], "value")
            self.assertEqual(kwargs["timeout"], 2.5)
            self.assertIsInstance(events[-1], FinalMessage)
            final = cast(FinalMessage, events[-1])
            self.assertEqual(final.content, "done")
            self.assertEqual(final.stop_reason, "end_turn")
            self.assertEqual(provider.previous_response_id, "resp_bg")
            self.assertEqual(provider.metrics["background_response_id"], "resp_bg")
            self.assertEqual(
                provider.metrics["background_response_status"], "completed"
            )

        asyncio.run(run_test())

    def test_responses_provider_polls_pending_background_response(self) -> None:
        """Responses retrieve 轮询未完成响应时只记录状态。"""

        async def run_test() -> None:
            client = FakeOpenAIClient(
                retrieve_responses=[
                    FakeResponsesResponse(
                        response_id="resp_bg",
                        output_text="",
                        status="in_progress",
                    )
                ]
            )
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )

            events = [
                event
                async for event in provider.retrieve_response(
                    "resp_bg",
                    options=StreamOptions(include=["file_search_call.results"]),
                )
            ]

            self.assertEqual(events, [])
            self.assertIs(client.responses.retrieve_kwargs["stream"], False)
            self.assertEqual(
                client.responses.retrieve_kwargs["include"],
                ["file_search_call.results"],
            )
            self.assertIsNone(provider.previous_response_id)
            self.assertEqual(
                provider.metrics["background_response_status"], "in_progress"
            )

        asyncio.run(run_test())

    def test_responses_provider_resumes_background_response_stream(self) -> None:
        """Responses retrieve 可从指定事件序号恢复流式读取。"""

        async def run_test() -> None:
            client = FakeOpenAIClient(
                retrieve_outputs=[
                    [
                        FakeResponsesStreamEvent(
                            "response.output_text.delta",
                            delta="done",
                        ),
                        FakeResponsesStreamEvent(
                            "response.completed",
                            response=FakeResponsesResponse(
                                response_id="resp_bg",
                                output_text="done",
                                status="completed",
                            ),
                        ),
                    ]
                ]
            )
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )

            events = [
                event
                async for event in provider.retrieve_response(
                    "resp_bg",
                    stream=True,
                    starting_after=12,
                    include_obfuscation=False,
                )
            ]

            kwargs = client.responses.retrieve_kwargs
            self.assertIs(kwargs["stream"], True)
            self.assertEqual(kwargs["starting_after"], 12)
            self.assertIs(kwargs["include_obfuscation"], False)
            self.assertIsInstance(events[0], TextDelta)
            text = cast(TextDelta, events[0])
            self.assertEqual(text.chunk, "done")
            self.assertIsInstance(events[-1], FinalMessage)
            self.assertEqual(provider.previous_response_id, "resp_bg")

        asyncio.run(run_test())

    def test_responses_provider_maps_server_compact_threshold(self) -> None:
        """Responses 将服务端压缩阈值映射到 context_management。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}],
                    [],
                    options=StreamOptions(server_compact_threshold=12000),
                )
            ]

            self.assertEqual(
                client.responses.kwargs["context_management"],
                [{"type": "compaction", "compact_threshold": 12000}],
            )
            self.assertNotIn(
                "context_management",
                client.responses.input_tokens.kwargs,
            )

        asyncio.run(run_test())

    def test_responses_provider_prefers_explicit_context_management(self) -> None:
        """显式 context_management 优先于服务端压缩阈值快捷配置。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )
            explicit = [{"type": "compaction", "compact_threshold": 8000}]

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}],
                    [],
                    options=StreamOptions(
                        context_management=explicit,
                        server_compact_threshold=12000,
                    ),
                )
            ]

            self.assertEqual(client.responses.kwargs["context_management"], explicit)

        asyncio.run(run_test())

    def test_responses_provider_maps_cache_retention(self) -> None:
        """Responses 将通用缓存保留策略映射到官方 prompt cache 参数。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}],
                    [],
                    options=StreamOptions(cache_retention="long"),
                )
            ]

            self.assertEqual(client.responses.kwargs["prompt_cache_retention"], "24h")

        asyncio.run(run_test())

    def test_responses_provider_prefers_prompt_cache_retention(self) -> None:
        """显式官方 prompt cache 参数优先于通用缓存保留策略。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}],
                    [],
                    options=StreamOptions(
                        cache_retention="long",
                        prompt_cache_retention="in_memory",
                    ),
                )
            ]

            self.assertEqual(
                client.responses.kwargs["prompt_cache_retention"],
                "in_memory",
            )

        asyncio.run(run_test())

    def test_responses_provider_adds_tool_fingerprint_prompt_cache_key(self) -> None:
        """Responses prompt_cache_key 包含当前工具目录指纹。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )
            tools = [
                ToolDefinition(
                    name="lookup",
                    description="Lookup records.",
                    schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                )
            ]

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}],
                    tools,
                )
            ]

            fingerprint = tool_catalog_fingerprint(tools)
            self.assertEqual(
                client.responses.kwargs["prompt_cache_key"],
                f"tools:{fingerprint}",
            )

        asyncio.run(run_test())

    def test_responses_provider_combines_base_prompt_cache_key(self) -> None:
        """配置的 prompt_cache_key 会追加工具目录指纹。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                prompt_cache_key="xcode-main",
                client=client,
            )
            tools = [
                ToolDefinition(
                    name="write_file",
                    description="Write a file.",
                    schema={"type": "object"},
                )
            ]

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}],
                    tools,
                )
            ]

            fingerprint = tool_catalog_fingerprint(tools)
            self.assertEqual(
                client.responses.kwargs["prompt_cache_key"],
                f"xcode-main:tools:{fingerprint}",
            )

        asyncio.run(run_test())

    def test_responses_provider_omits_empty_prompt_cache_key(self) -> None:
        """没有基础 key 和工具目录时不发送 prompt_cache_key。"""

        async def run_test() -> None:
            client = FakeOpenAIClient()
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                client=client,
            )

            _events = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "hi"}],
                    [],
                )
            ]

            self.assertNotIn("prompt_cache_key", client.responses.kwargs)

        asyncio.run(run_test())

    def test_responses_store_false_round_trips_encrypted_reasoning(self) -> None:
        """store=false 时回灌 encrypted reasoning item。"""

        async def run_test() -> None:
            client = FakeOpenAIClient(
                response_outputs=[
                    [
                        FakeResponsesStreamEvent(
                            "response.completed",
                            response=FakeResponsesResponse(
                                response_id="r1",
                                output=[
                                    {
                                        "type": "reasoning",
                                        "encrypted_content": "ciphertext",
                                    }
                                ],
                            ),
                        )
                    ],
                    [],
                ]
            )
            provider = OpenAIResponsesProvider(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-5.4",
                thinking=True,
                client=client,
            )
            options = StreamOptions(store=False)

            _first = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "first"}], [], options=options
                )
            ]
            _second = [
                event
                async for event in provider.stream(
                    [{"role": "user", "content": "second"}], [], options=options
                )
            ]

            first_call, second_call = client.responses.calls
            self.assertEqual(first_call["include"], ["reasoning.encrypted_content"])
            self.assertNotIn("previous_response_id", second_call)
            self.assertEqual(second_call["input"][0]["type"], "reasoning")
            self.assertEqual(second_call["input"][0]["encrypted_content"], "ciphertext")

        asyncio.run(run_test())

    def test_responses_stream_accepts_reasoning_text_delta(self) -> None:
        """Responses reasoning 文本增量事件会转为 ReasoningDelta。"""
        events = list(
            responses_stream_to_events(
                cast(
                    Any,
                    [
                        FakeResponsesStreamEvent(
                            "response.reasoning_text.delta", delta="why"
                        )
                    ],
                )
            )
        )

        self.assertIsInstance(events[0], ReasoningDelta)
        assert isinstance(events[0], ReasoningDelta)
        self.assertEqual(events[0].chunk, "why")

    def test_responses_stream_decodes_shell_call(self) -> None:
        """Responses shell_call item 会转为内部工具调用事件。"""
        events = list(
            responses_stream_to_events(
                cast(
                    Any,
                    [
                        FakeResponsesStreamEvent(
                            "response.output_item.done",
                            item=FakeResponsesOutputItem(
                                item_type="shell_call",
                                call_id="call_1",
                                action=FakeShellAction(
                                    commands=["python --version"],
                                    timeout_ms=120000,
                                    max_output_length=4096,
                                ),
                            ),
                            output_index=0,
                        )
                    ],
                )
            )
        )

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], ToolCallEvent)
        tool_event = cast(ToolCallEvent, events[0])
        self.assertEqual(tool_event.calls[0].id, "call_1")
        self.assertEqual(tool_event.calls[0].name, "shell")
        self.assertEqual(
            tool_event.calls[0].input,
            {
                "commands": ["python --version"],
                "timeout_ms": 120000,
                "max_output_length": 4096,
            },
        )

    def test_responses_stream_failed_yields_error_final(self) -> None:
        """Responses failed 事件会转为 error stop reason。"""
        events = list(
            responses_stream_to_events(
                cast(
                    Any,
                    [
                        FakeResponsesStreamEvent(
                            "response.output_text.delta",
                            delta="partial",
                        ),
                        FakeResponsesStreamEvent(
                            "response.failed",
                            response=FakeResponsesResponse(
                                response_id="r_failed",
                                output_text="failed text",
                                status="failed",
                            ),
                        ),
                    ],
                )
            )
        )

        self.assertIsInstance(events[-1], FinalMessage)
        final = cast(FinalMessage, events[-1])
        self.assertEqual(final.content, "failed text")
        self.assertEqual(final.stop_reason, "error")

    def test_responses_stream_incomplete_yields_max_tokens_final(self) -> None:
        """Responses incomplete 事件会转为 max_tokens stop reason。"""
        events = list(
            responses_stream_to_events(
                cast(Any, [FakeResponsesStreamEvent("response.incomplete")])
            )
        )

        self.assertIsInstance(events[-1], FinalMessage)
        final = cast(FinalMessage, events[-1])
        self.assertEqual(final.stop_reason, "max_tokens")

    def test_responses_builtin_tool_is_passed_through(self) -> None:
        """Responses 内建工具定义不包成 function tool。"""
        tool = ToolDefinition(
            name="web_search",
            description="Search the web.",
            schema={},
            builtin={"type": "web_search_preview"},
        )

        encoded = to_responses_tool(
            tool.name,
            tool.description,
            tool.schema,
            tool.builtin,
        )

        self.assertEqual(encoded, {"type": "web_search_preview"})

    def test_responses_builtin_tool_rejects_missing_type(self) -> None:
        """builtin 工具缺少 type 字段时抛出 ValueError。"""
        with self.assertRaises(ValueError):
            to_responses_tool("bad", "desc", None, builtin={})

    def test_responses_builtin_tool_rejects_empty_type(self) -> None:
        """builtin 工具 type 为空字符串时抛出 ValueError。"""
        with self.assertRaises(ValueError):
            to_responses_tool("bad", "desc", None, builtin={"type": ""})

    def test_responses_input_supports_image_and_file_blocks(self) -> None:
        """Responses input 支持图像和文件内容块。"""
        converted = to_responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        ImageContent(source={"url": "https://example.test/a.png"}),
                        {
                            "type": "input_file",
                            "source": {"file_id": "file_123"},
                        },
                    ],
                }
            ]
        )

        content = converted[0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "inspect"})
        self.assertEqual(
            content[1],
            {"type": "input_image", "image_url": "https://example.test/a.png"},
        )
        self.assertEqual(content[2], {"type": "input_file", "file_id": "file_123"})

    def test_agent_file_content_reaches_responses_input(self) -> None:
        """Agent 文件内容块会转换为 Responses input_file。"""
        messages = convert_to_llm(
            [
                UserMessage(
                    content=[
                        TextContent(text="inspect"),
                        FileContent(file_id="file_123"),
                        FileContent(
                            filename="notes.txt",
                            file_data="data:text/plain;base64,SGVsbG8=",
                        ),
                    ]
                )
            ]
        )

        converted = to_responses_input(messages)

        content = converted[0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "inspect"})
        self.assertEqual(content[1], {"type": "input_file", "file_id": "file_123"})
        self.assertEqual(
            content[2],
            {
                "type": "input_file",
                "filename": "notes.txt",
                "file_data": "data:text/plain;base64,SGVsbG8=",
            },
        )

    def test_responses_input_supports_shell_call_output_blocks(self) -> None:
        """Responses input 支持官方 shell_call_output 项。"""
        converted = to_responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "shell_call_output",
                            "call_id": "call_1",
                            "max_output_length": 4096,
                            "output": [
                                {
                                    "stdout": "Python 3.11\n",
                                    "stderr": "",
                                    "outcome": {"type": "exit", "exit_code": 0},
                                }
                            ],
                        }
                    ],
                }
            ]
        )

        self.assertEqual(
            converted,
            [
                {
                    "type": "shell_call_output",
                    "call_id": "call_1",
                    "max_output_length": 4096,
                    "output": [
                        {
                            "stdout": "Python 3.11\n",
                            "stderr": "",
                            "outcome": {"type": "exit", "exit_code": 0},
                        }
                    ],
                }
            ],
        )

    def test_responses_input_maps_system_role_to_developer(self) -> None:
        """Responses input 将系统指令映射为 developer 角色。"""
        converted = to_responses_input(
            [
                {"role": "system", "content": "Follow project instructions."},
                {"role": "user", "content": "hello"},
            ]
        )

        self.assertEqual(converted[0]["role"], "developer")
        self.assertEqual(converted[1]["role"], "user")


@dataclass(frozen=True)
class MockProfile:
    """模拟 provider profile 配置。"""

    transport: str
    chat_model: str = "gpt-5.4"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "test-key"
    thinking: bool = True
    reasoning_effort: str | None = "high"
    clear_thinking: bool = False
    tool_stream: bool = True
    response_format: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """为测试 profile 设置默认结构化输出配置。"""
        if self.response_format is not None:
            return
        object.__setattr__(self, "response_format", {"type": "json_object"})


class FakeOpenAIClient:
    """记录 Chat Completions 请求参数的测试客户端。"""

    def __init__(
        self,
        response_outputs: list[list[Any]] | None = None,
        input_token_count: int = 0,
        retrieve_responses: list[Any] | None = None,
        retrieve_outputs: list[list[Any]] | None = None,
    ) -> None:
        self.chat = FakeChat()
        self.responses = FakeResponses(
            response_outputs or [],
            input_token_count,
            retrieve_responses or [],
            retrieve_outputs or [],
        )
        self.override_api_key: str | None = None

    def with_options(self, *, api_key: str) -> FakeOpenAIClient:
        """记录请求级 API key 并返回同一个测试客户端。"""
        self.override_api_key = api_key
        return self


class FakeChat:
    """模拟 OpenAI chat 命名空间。"""

    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeCompletions:
    """记录 create 调用参数。"""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        """返回空流并保存请求。"""
        self.kwargs = kwargs
        return iter([])


class FakeResponses:
    """记录 Responses create 调用参数。"""

    def __init__(
        self,
        outputs: list[list[Any]],
        input_token_count: int,
        retrieve_responses: list[Any],
        retrieve_outputs: list[list[Any]],
    ) -> None:
        self.kwargs: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.outputs = outputs
        self.input_tokens = FakeInputTokens(input_token_count)
        self.retrieve_response_id: str | None = None
        self.retrieve_kwargs: dict[str, Any] = {}
        self.retrieve_calls: list[dict[str, Any]] = []
        self.retrieve_responses = retrieve_responses
        self.retrieve_outputs = retrieve_outputs

    def create(self, **kwargs: Any) -> Any:
        """返回空流并保存请求。"""
        self.kwargs = kwargs
        self.calls.append(kwargs)
        if self.outputs:
            return iter(self.outputs.pop(0))
        return iter([])

    def retrieve(self, response_id: str, **kwargs: Any) -> Any:
        """返回后台响应或恢复流并保存请求。"""
        self.retrieve_response_id = response_id
        self.retrieve_kwargs = kwargs
        self.retrieve_calls.append({"response_id": response_id, **kwargs})
        if kwargs.get("stream"):
            if self.retrieve_outputs:
                return iter(self.retrieve_outputs.pop(0))
            return iter([])
        if self.retrieve_responses:
            return self.retrieve_responses.pop(0)
        return FakeResponsesResponse(response_id=response_id)


class FakeInputTokens:
    """记录 Responses input_tokens.count 调用参数。"""

    def __init__(self, input_token_count: int) -> None:
        self.kwargs: dict[str, Any] = {}
        self.input_token_count = input_token_count

    def count(self, **kwargs: Any) -> FakeInputTokenCountResponse:
        """返回固定输入 token 数并保存请求。"""
        self.kwargs = kwargs
        return FakeInputTokenCountResponse(self.input_token_count)


@dataclass(frozen=True)
class FakeInputTokenCountResponse:
    """模拟 Responses input token 计数响应。"""

    input_tokens: int


class FakeResponsesStreamEvent:
    """模拟 Responses 流式事件。"""

    def __init__(
        self,
        event_type: str,
        response: FakeResponsesResponse | None = None,
        delta: str | None = None,
        item: FakeResponsesOutputItem | None = None,
        output_index: int = 0,
    ) -> None:
        self.type = event_type
        self.response = response
        self.delta = delta
        self.item = item
        self.output_index = output_index


class FakeResponsesOutputItem:
    """模拟 Responses 输出 item。"""

    def __init__(
        self,
        item_type: str,
        call_id: str | None = None,
        action: FakeShellAction | None = None,
    ) -> None:
        self.type = item_type
        self.call_id = call_id
        self.id = None
        self.name = None
        self.arguments = None
        self.action = action


class FakeShellAction:
    """模拟 Responses shell action。"""

    def __init__(
        self,
        commands: list[str],
        timeout_ms: int,
        max_output_length: int,
    ) -> None:
        self.commands = commands
        self.timeout_ms = timeout_ms
        self.max_output_length = max_output_length


class FakeResponsesResponse:
    """模拟 Responses 完成响应。"""

    def __init__(
        self,
        response_id: str,
        output: list[dict[str, Any]] | None = None,
        output_text: str = "",
        status: str = "completed",
        usage: Any | None = None,
    ) -> None:
        self.id = response_id
        self.output_text = output_text
        self.output = output or []
        self.status = status
        self.usage = usage


if __name__ == "__main__":
    unittest.main()
