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
        llm.metrics = {}

        first = list(llm._stream_sync([{"role": "user", "content": "one"}], ()))
        second = list(llm._stream_sync([{"role": "user", "content": "two"}], ()))

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


if __name__ == "__main__":
    unittest.main()
