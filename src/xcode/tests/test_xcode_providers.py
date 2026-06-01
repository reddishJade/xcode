from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typing import Any
from xcode.agent.types import tool_definition_from_spec
from xcode.harness.agent_runtime.events import TextDelta, ToolCallReady, FinalMessage
from xcode.ai.providers.factory import (
    ProviderRuntime,
    ProviderSettings,
    RateLimitPolicy,
    RetryPolicy,
    build_provider_bundle,
    get_config_value,
    load_env_file,
    _resolve_api_key,
)
from xcode.ai.providers.openai import OpenAIChatProvider, OpenAIResponsesProvider
from xcode.harness.config import ModelProfileRuntimeConfig
from xcode.harness.skills import ToolSpec


class XcodeProviderEnvTests(unittest.TestCase):
    def test_env_file_loading_and_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("A=from_file\nQUOTED='value'\n", encoding="utf-8")

            self.assertEqual(load_env_file(env_file)["A"], "from_file")
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
                                transport="responses_stateful",
                            ),
                            "subagent": ModelProfileRuntimeConfig(
                                chat_model="small-model",
                                base_url="https://small.test",
                            ),
                        },
                    )
                )

                self.assertIsInstance(bundle.llm, OpenAIResponsesProvider)
                self.assertEqual(bundle.llm.transport, "responses_stateful")
                self.assertEqual(bundle.llm.model, "main-model")
                self.assertEqual(bundle.llms["subagent"].model, "small-model")
                self.assertEqual(bundle.llms["judge"].model, "main-model")

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

                self.assertEqual(bundle.llm.client.api_key, "configured")

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

                self.assertEqual(bundle.llms["main"].client.api_key, "openai")
                self.assertEqual(bundle.llms["subagent"].client.api_key, "subagent")


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


class XcodeStructuredProviderTests(unittest.TestCase):
    def test_stream_converts_tool_schema_and_tool_calls(self) -> None:
        llm = OpenAIChatProvider.__new__(OpenAIChatProvider)
        llm.model = "model"
        llm.thinking = True
        llm.reasoning_effort = None
        llm.client = FakeOpenAIClient(
            stream_chunks=[
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
        llm.runtime = ProviderRuntime()
        llm.transport = "chat_completions"
        tool = ToolSpec(
            name="echo",
            description="Echo input.",
            input_hint='JSON: {"text": "..."}',
            handler=lambda data: data["text"],
            schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

        events = list(
            llm._stream_sync(
                [{"role": "user", "content": "echo"}],
                (tool_definition_from_spec(tool),),
            )
        )
        tool_call = events[-1]
        self.assertIsInstance(tool_call, ToolCallReady)
        assert isinstance(tool_call, ToolCallReady)
        self.assertEqual(tool_call.calls[0].name, "echo")
        self.assertEqual(tool_call.calls[0].input, {"text": "hello"})
        sent_tool = llm.client.chat.completions.kwargs["tools"][0]
        self.assertEqual(sent_tool["function"]["parameters"], tool.schema)

    def test_stream_converts_tool_results_to_openai_messages(self) -> None:
        llm = OpenAIChatProvider.__new__(OpenAIChatProvider)
        llm.model = "model"
        llm.thinking = True
        llm.reasoning_effort = None
        llm.client = FakeOpenAIClient(
            stream_chunks=[FakeStreamChunk(content="done")],
        )
        llm.runtime = ProviderRuntime()
        llm.transport = "chat_completions"

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

        sent_messages = llm.client.chat.completions.kwargs["messages"]
        self.assertEqual(sent_messages[0]["role"], "assistant")
        self.assertEqual(sent_messages[1]["role"], "tool")
        self.assertEqual(
            [e for e in events if isinstance(e, TextDelta)][0].chunk, "done"
        )

    def test_stream_yields_text_and_tool_call_deltas(self) -> None:
        llm = OpenAIChatProvider.__new__(OpenAIChatProvider)
        llm.model = "model"
        llm.thinking = True
        llm.reasoning_effort = None
        llm.client = FakeOpenAIClient(
            stream_chunks=[
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
        llm.runtime = ProviderRuntime()
        llm.transport = "chat_completions"

        events = list(llm._stream_sync([{"role": "user", "content": "go"}], ()))

        self.assertIsInstance(events[0], TextDelta)
        assert isinstance(events[0], TextDelta)
        self.assertEqual(events[0].chunk, "he")
        self.assertIsInstance(events[1], TextDelta)
        assert isinstance(events[1], TextDelta)
        self.assertEqual(events[1].chunk, "llo")
        self.assertIsInstance(events[-1], ToolCallReady)
        assert isinstance(events[-1], ToolCallReady)
        self.assertEqual(events[-1].calls[0].id, "call-1")
        self.assertEqual(events[-1].calls[0].name, "echo")
        self.assertEqual(events[-1].calls[0].input, {"text": "hi"})
        self.assertTrue(llm.client.chat.completions.kwargs["stream"])

    def test_responses_stream_with_previous_response_id(self) -> None:
        llm = OpenAIResponsesProvider.__new__(OpenAIResponsesProvider)
        llm.model = "model"
        llm.thinking = True
        llm.client = FakeOpenAIClient(
            response_outputs=[
                [
                    FakeResponsesStreamEvent("response.output_text.delta", delta="he"),
                    FakeResponsesStreamEvent("response.output_text.delta", delta="llo"),
                    FakeResponsesStreamEvent(
                        "response.completed", FakeResponsesResponse("r1")
                    ),
                ],
                [
                    FakeResponsesStreamEvent("response.output_text.delta", delta="ok"),
                    FakeResponsesStreamEvent(
                        "response.completed", FakeResponsesResponse("r2")
                    ),
                ],
            ]
        )
        llm.runtime = ProviderRuntime()
        llm.transport = "responses_stateful"
        llm.prompt_cache_key = None
        llm.previous_response_id = None
        llm.reasoning_effort = None
        llm.metrics = {}
        llm._last_sent_message_index = 0
        llm._pending_sent_message_index = 0

        messages = [{"role": "user", "content": "one"}]
        first = list(llm._stream_sync(messages, ()))
        messages.append({"role": "user", "content": "two"})
        second = list(llm._stream_sync(messages, ()))

        self.assertEqual(
            [event.chunk for event in first if isinstance(event, TextDelta)],
            ["he", "llo"],
        )
        assert isinstance(first[-1], FinalMessage)
        self.assertEqual(first[-1].content, "ok-r1")
        self.assertEqual(
            [event.chunk for event in second if isinstance(event, TextDelta)], ["ok"]
        )
        assert isinstance(second[-1], FinalMessage)
        self.assertEqual(second[-1].content, "ok-r2")
        calls = llm.client.responses.calls
        self.assertNotIn("previous_response_id", calls[0])
        self.assertEqual(calls[1]["previous_response_id"], "r1")

    def test_responses_stream_yields_text_delta_immediately(self) -> None:
        """验证 streaming 行为：TextDelta 在流迭代过程中立即产出，
        而不是等底层 iterator 全部耗尽后才产出。"""
        llm = OpenAIResponsesProvider.__new__(OpenAIResponsesProvider)
        llm.model = "model"
        llm.thinking = True
        llm.reasoning_effort = None
        llm.client = FakeOpenAIClient(
            response_outputs=[
                [
                    FakeResponsesStreamEvent(
                        "response.output_text.delta", delta="he"
                    ),
                    FakeResponsesStreamEvent(
                        "response.output_text.delta", delta="llo"
                    ),
                    FakeResponsesStreamEvent(
                        "response.completed", FakeResponsesResponse("r1")
                    ),
                ],
            ]
        )
        llm.runtime = ProviderRuntime()
        llm.transport = "responses_stateful"
        llm.prompt_cache_key = None
        llm.previous_response_id = None
        llm.metrics = {}
        llm._last_sent_message_index = 0
        llm._pending_sent_message_index = 0

        events = llm._stream_sync([{"role": "user", "content": "hi"}], ())

        # 第一个事件应该是 TextDelta，立即产出（completed 事件尚未到达）
        first = next(events)
        self.assertIsInstance(first, TextDelta)
        self.assertEqual(first.chunk, "he")

        # 第二个事件也是 TextDelta，在 completed 之前产出
        second = next(events)
        self.assertIsInstance(second, TextDelta)
        self.assertEqual(second.chunk, "llo")

        # 第三个事件应该是 FinalMessage（completed 已处理）
        third = next(events)
        self.assertIsInstance(third, FinalMessage)

        # 迭代器已耗尽
        with self.assertRaises(StopIteration):
            next(events)

    def test_responses_tool_loop_does_not_resend_function_call(self) -> None:
        """第二轮 messages 包含 assistant tool_calls 和 tool result 时，
        实际发给 responses.create() 的 input 只有 function_call_output，
        没有重复的 function_call。"""
        llm = OpenAIResponsesProvider.__new__(OpenAIResponsesProvider)
        llm.model = "model"
        llm.thinking = True
        llm.reasoning_effort = None
        llm.client = FakeOpenAIClient(
            response_outputs=[
                # 第一轮：返回 function_call 的流
                [
                    FakeResponsesStreamEvent(
                        "response.completed", FakeResponsesResponse("r1")
                    ),
                ],
                # 第二轮：返回文本的流
                [
                    FakeResponsesStreamEvent(
                        "response.output_text.delta", delta="done"
                    ),
                    FakeResponsesStreamEvent(
                        "response.completed", FakeResponsesResponse("r2")
                    ),
                ],
            ]
        )
        llm.runtime = ProviderRuntime()
        llm.transport = "responses_stateful"
        llm.prompt_cache_key = None
        llm.previous_response_id = None
        llm.metrics = {}
        llm._last_sent_message_index = 0
        llm._pending_sent_message_index = 0

        # 第一轮：用户消息
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "call tool"},
        ]
        list(llm._stream_sync(messages, ()))
        self.assertEqual(len(llm.client.responses.calls), 1)
        first_call = llm.client.responses.calls[0]
        self.assertNotIn("previous_response_id", first_call)

        # 第二轮：包含 assistant tool_calls + tool result
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }
            ],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": "call1",
            "content": "tool output",
        })

        list(llm._stream_sync(messages, ()))
        self.assertEqual(len(llm.client.responses.calls), 2)
        second_call = llm.client.responses.calls[1]
        self.assertEqual(second_call["previous_response_id"], "r1")

        # input 中只能有 function_call_output，不能有 function_call
        input_items = second_call["input"]
        self.assertTrue(all(
            item.get("type") != "function_call"
            for item in input_items
        ))
        function_call_outputs = [
            item for item in input_items
            if item.get("type") == "function_call_output"
        ]
        self.assertEqual(len(function_call_outputs), 1)
        self.assertEqual(function_call_outputs[0]["call_id"], "call1")
        self.assertEqual(function_call_outputs[0]["output"], "tool output")


class FakeOpenAIClient:
    def __init__(
        self, content=None, tool_calls=None, stream_chunks=None, response_outputs=None
    ) -> None:
        self.chat = FakeChat(content, tool_calls, stream_chunks)
        self.responses = FakeResponses(response_outputs or [])


class FakeResponses:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class FakeResponsesResponse:
    def __init__(self, response_id) -> None:
        self.id = response_id
        self.output_text = f"ok-{response_id}"
        self.output: list[Any] = []
        self.usage = None


class FakeResponsesStreamEvent:
    def __init__(self, type_: str, response=None, delta=None) -> None:
        self.type = type_
        self.response = response
        self.delta = delta


class FakeChat:
    def __init__(self, content, tool_calls, stream_chunks) -> None:
        self.completions = FakeCompletions(content, tool_calls, stream_chunks)


class FakeCompletions:
    def __init__(self, content, tool_calls, stream_chunks) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.stream_chunks = stream_chunks
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return iter(self.stream_chunks or [])
        return FakeResponse(self.content, self.tool_calls)


class FakeResponse:
    def __init__(self, content, tool_calls) -> None:
        self.choices = [FakeChoice(content, tool_calls)]


class FakeChoice:
    def __init__(self, content, tool_calls) -> None:
        self.message = FakeMessage(content, tool_calls)


class FakeMessage:
    def __init__(self, content, tool_calls) -> None:
        self.content = content
        self.tool_calls = tool_calls if tool_calls is not None else [FakeToolCall()]


class FakeToolCall:
    id = "call-1"

    class function:
        name = "echo"
        arguments = '{"text": "hello"}'


class FakeStreamChunk:
    def __init__(self, content=None, tool_call=None) -> None:
        self.choices = [FakeStreamChoice(content, tool_call)]
        self.usage = None


class FakeStreamChoice:
    def __init__(self, content, tool_call) -> None:
        self.delta = FakeStreamDelta(content, tool_call)


class FakeStreamDelta:
    def __init__(self, content, tool_call) -> None:
        self.content = content
        self.tool_calls = [tool_call] if tool_call is not None else []
        self.reasoning_content = None


class FakeStreamToolCall:
    def __init__(self, index, call_id=None, name=None, arguments=None) -> None:
        self.index = index
        self.id = call_id
        self.function = FakeStreamFunction(name, arguments)


class FakeStreamFunction:
    def __init__(self, name, arguments) -> None:
        self.name = name
        self.arguments = arguments


class XcodeChatGLMProviderTests(unittest.TestCase):
    """ChatGLM provider 边界测试：thinking 清理、tool_stream、参数组合。"""

    def _make_provider(self, **overrides) -> ChatGLMProvider:
        from xcode.ai.providers.chatglm import ChatGLMProvider

        kwargs = dict(
            api_key="test-key",
            model="glm-4-flash",
            thinking=True,
            clear_thinking=False,
            tool_stream=True,
            client=FakeGLMClient(),
        )
        kwargs.update(overrides)
        return ChatGLMProvider(**kwargs)

    def test_thinking_disabled_sets_extra_body(self) -> None:
        """thinking=False 时 extra_body 为 disabled。"""
        provider = self._make_provider(thinking=False, client=FakeGLMClient())
        # 消费生成器以触发 _stream_sync 内部代码
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        kwargs = provider.client.chat.completions.kwargs
        extra = kwargs.get("extra_body", {})
        self.assertEqual(extra.get("thinking", {}).get("type"), "disabled")

    def test_thinking_enabled_clear_false(self) -> None:
        """clear_thinking=False 时 extra_body 包含 clear_thinking=false。"""
        provider = self._make_provider(clear_thinking=False, client=FakeGLMClient())
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        kwargs = provider.client.chat.completions.kwargs
        extra = kwargs.get("extra_body", {})
        thinking = extra.get("thinking", {})
        self.assertEqual(thinking.get("type"), "enabled")
        self.assertIs(thinking.get("clear_thinking"), False)

    def test_thinking_enabled_clear_true(self) -> None:
        """clear_thinking=True 时 extra_body 包含 clear_thinking=true。"""
        provider = self._make_provider(clear_thinking=True, client=FakeGLMClient())
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        kwargs = provider.client.chat.completions.kwargs
        extra = kwargs.get("extra_body", {})
        thinking = extra.get("thinking", {})
        self.assertEqual(thinking.get("type"), "enabled")
        self.assertIs(thinking.get("clear_thinking"), True)

    def test_tool_stream_disabled(self) -> None:
        """tool_stream=False 时不传 tool_stream 参数。"""
        provider = self._make_provider(tool_stream=False, client=FakeGLMClient())
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        kwargs = provider.client.chat.completions.kwargs
        self.assertNotIn("tool_stream", kwargs)

    def test_tool_stream_enabled(self) -> None:
        """tool_stream=True 时传 tool_stream=true。"""
        provider = self._make_provider(tool_stream=True, client=FakeGLMClient())
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        kwargs = provider.client.chat.completions.kwargs
        self.assertIs(kwargs.get("tool_stream"), True)

    def test_clean_reasoning_no_tool_loop(self) -> None:
        """非工具循环：清除所有历史 reasoning_content。"""
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "think1"},
            {"role": "user", "content": "q2"},
        ]
        cleaned = provider._clean_reasoning_content(messages)
        for msg in cleaned:
            self.assertNotIn("reasoning_content", msg)

    def test_clean_reasoning_in_tool_loop_retains_current(self) -> None:
        """工具循环中保留当前轮次的 reasoning_content。"""
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "think1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant",
             "content": "a2",
             "reasoning_content": "think2",
             "tool_calls": []},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
        cleaned = provider._clean_reasoning_content(messages)
        # 倒数第二条（当前轮 assistant）应保留 reasoning_content
        for i, msg in enumerate(cleaned):
            if i == len(cleaned) - 2:  # 当前轮次 assistant
                self.assertIn("reasoning_content", msg,
                              f"msg[{i}] should retain reasoning_content")
                self.assertEqual(msg["reasoning_content"], "think2")
            elif msg.get("role") == "assistant":
                self.assertNotIn("reasoning_content", msg,
                                 f"msg[{i}] should not have reasoning_content")

    def test_clean_reasoning_pre_tool_loop_cleared(self) -> None:
        """工具循环中，当前轮之前的所有 assistant reasoning_content 都被清除。"""
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "think1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant",
             "content": "",
             "reasoning_content": "think2",
             "tool_calls": [{"id": "t1", "function": {"name": "echo", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
        cleaned = provider._clean_reasoning_content(messages)
        # 第一个 assistant (index 1) 的 reasoning 应被清除
        self.assertNotIn("reasoning_content", cleaned[1])
        # 第二个 assistant (index 3, 当前轮) 保留
        self.assertIn("reasoning_content", cleaned[3])

    def test_thinking_true_streams_reasoning(self) -> None:
        """thinking=True 时流包含 reasoning delta。"""
        from xcode.harness.agent_runtime.events import ReasoningDelta

        client = FakeGLMClient(stream_chunks=[
            FakeGLMChunk(content="hello", reasoning="thinking..."),
            FakeGLMChunk(content=" world"),
        ])
        provider = self._make_provider(client=client)
        events = list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        reasoning_events = [e for e in events if isinstance(e, ReasoningDelta)]
        self.assertEqual(len(reasoning_events), 1)
        self.assertEqual(reasoning_events[0].chunk, "thinking...")

    def test_combo_tool_stream_plus_clear_thinking(self) -> None:
        """tool_stream=True + clear_thinking=True 组合参数正确传递。"""
        provider = self._make_provider(
            tool_stream=True, clear_thinking=True, client=FakeGLMClient()
        )
        list(provider._stream_sync([{"role": "user", "content": "hi"}], ()))
        kwargs = provider.client.chat.completions.kwargs
        self.assertIs(kwargs["tool_stream"], True)
        self.assertIs(kwargs["extra_body"]["thinking"]["clear_thinking"], True)

    def test_record_usage_cached_and_reasoning_tokens(self) -> None:
        """usage 统计记录 cached_tokens 和 reasoning_tokens。"""
        from collections import namedtuple

        FakeUsage = namedtuple("FakeUsage", ["prompt_tokens", "completion_tokens",
                                              "prompt_tokens_details", "completion_tokens_details"])
        FakePromptDetails = namedtuple("FakePromptDetails", ["cached_tokens"])
        FakeCompletionDetails = namedtuple("FakeCompletionDetails", ["reasoning_tokens"])

        class FakeResponse:
            usage = FakeUsage(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_tokens_details=FakePromptDetails(cached_tokens=20),
                completion_tokens_details=FakeCompletionDetails(reasoning_tokens=10),
            )

        provider = self._make_provider()
        provider._record_usage(FakeResponse(), sent_messages=2)
        self.assertEqual(provider.metrics["cached_tokens"], 20)
        self.assertEqual(provider.metrics["reasoning_tokens"], 10)
        self.assertEqual(provider.metrics["sent_messages"], 2)

    def test_record_usage_none_details_degradation(self) -> None:
        """prompt_tokens_details / completion_tokens_details 为 None 时降级为 0。"""
        from collections import namedtuple

        FakeUsage = namedtuple("FakeUsage", ["prompt_tokens", "completion_tokens",
                                              "prompt_tokens_details", "completion_tokens_details"])

        class FakeResponseNoDetails:
            usage = FakeUsage(
                prompt_tokens=50,
                completion_tokens=30,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            )

        provider = self._make_provider()
        provider._record_usage(FakeResponseNoDetails(), sent_messages=1)
        self.assertEqual(provider.metrics["cached_tokens"], 0)
        self.assertEqual(provider.metrics["reasoning_tokens"], 0)

    def test_transport_is_chatglm(self) -> None:
        provider = self._make_provider()
        self.assertEqual(provider.transport, "chatglm")


class FakeGLMClient:
    """模拟 OpenAI Chat Completion 客户端，用于 ChatGLM provider 测试。"""

    def __init__(self, stream_chunks=None) -> None:
        self.chat = FakeGLMChat(stream_chunks or [])


class FakeGLMChat:
    def __init__(self, stream_chunks) -> None:
        self.completions = FakeGLMCompletions(stream_chunks)


class FakeGLMCompletions:
    def __init__(self, stream_chunks) -> None:
        self.stream_chunks = stream_chunks
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.stream_chunks or [FakeGLMChunk(content="ok")])


class FakeGLMChunk:
    def __init__(self, content="", reasoning=None) -> None:
        self.choices = [FakeGLMChoice(content, reasoning)]
        self.usage = None


class FakeGLMChoice:
    def __init__(self, content, reasoning) -> None:
        self.delta = FakeGLMDelta(content, reasoning)


class FakeGLMDelta:
    def __init__(self, content, reasoning) -> None:
        self.content = content
        self.reasoning_content = reasoning
        self.tool_calls = []


if __name__ == "__main__":
    unittest.main()
