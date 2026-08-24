"""Provider 流式收集、取消与故障分类测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from xcode.agent._provider import _collect_provider_events
from xcode.agent.agent_loop import run_agent_loop
from xcode.agent.config import AgentContext, AgentLoopConfig
from xcode.agent.messages import UserMessage
from xcode.agent.results import TerminationReason
from xcode.ai.events import (
    FinalMessage,
    ProviderEvent,
    ProviderFailure,
    TextDelta,
)
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.harness.agent_runtime.cancellation import CancellationToken
from xcode.harness.agent_runtime.result import _build_structured_result


class _EndlessProvider:
    """无限产出文本增量，并记录中断中止调用。"""

    def __init__(self) -> None:
        self.abort_calls = 0

    def abort_active_stream(self) -> None:
        self.abort_calls += 1

    async def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[object],
        options: object | None = None,
        **kwargs: object,
    ) -> object:
        while True:
            yield TextDelta(chunk="tok")
            await asyncio.sleep(0)


class _RaisingProvider:
    """流在首个事件前即抛出连接异常，模拟打断关闭连接的场景。"""

    def __init__(self) -> None:
        self.abort_calls = 0

    def abort_active_stream(self) -> None:
        self.abort_calls += 1

    async def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[object],
        options: object | None = None,
        **kwargs: object,
    ) -> object:
        if False:  # pragma: no cover
            yield TextDelta(chunk="never")
        raise RuntimeError("connection reset by abort")


async def test_collect_returns_none_when_cancelled_mid_stream() -> None:
    """流式生成中途取消：返回 None 并中止在途请求。"""
    token = CancellationToken()
    provider = _EndlessProvider()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.01)
        token.cancel("interrupted by user")

    canceller = asyncio.create_task(cancel_soon())
    events = await _collect_provider_events(
        provider, [], [], None, lambda _event: None, token
    )
    await canceller

    assert events is None
    assert provider.abort_calls >= 1


async def test_abort_exception_during_cancelled_stream_is_not_error() -> None:
    """打断关闭连接导致的流异常：已取消时按取消处理而非报错。"""
    token = CancellationToken()
    token.cancel("interrupted by user")
    provider = _RaisingProvider()

    events = await _collect_provider_events(
        provider, [], [], None, lambda _event: None, token
    )

    assert events is None


class _ShortProvider:
    async def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[object],
        options: object | None = None,
        **kwargs: object,
    ) -> object:
        yield TextDelta(chunk="hello")


async def test_collect_returns_events_when_not_cancelled() -> None:
    """未取消时正常收集全部事件。"""
    events = await _collect_provider_events(
        _ShortProvider(), [], [], None, lambda _event: None, None
    )
    assert events == [TextDelta(chunk="hello")]


async def test_agent_loop_terminates_when_interrupted_mid_stream() -> None:
    """端到端：模型流式生成期间被打断，循环以 CANCELLED 退出并中止流。"""
    provider = _EndlessProvider()
    token = CancellationToken()
    config = AgentLoopConfig(provider=provider)
    context = AgentContext(
        system_prompt="",
        messages=[UserMessage(content="hi")],
    )

    async def cancel_soon() -> None:
        await asyncio.sleep(0.05)
        token.cancel("interrupted by user")

    canceller = asyncio.create_task(cancel_soon())
    result = await run_agent_loop(
        [UserMessage(content="hi")],
        context,
        config,
        emit=lambda _event: None,
        signal=token,
    )
    await canceller

    assert result.termination_reason is TerminationReason.CANCELLED
    assert provider.abort_calls >= 1


class _ServiceUnavailable(RuntimeError):
    status_code = 503


class _FailingProvider:
    @property
    def model(self) -> str:
        return "failing-model"

    async def stream(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **_kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        del messages, tools, options
        raise _ServiceUnavailable("temporarily unavailable")
        yield FinalMessage(content="", stop_reason="end_turn")


async def test_provider_exception_is_a_structured_event() -> None:
    events = await _collect_provider_events(
        _FailingProvider(),
        [],
        [],
        None,
        lambda _event: None,
    )

    assert events == [
        ProviderFailure(
            message="temporarily unavailable",
            exception_type="_ServiceUnavailable",
            status_code=503,
        )
    ]


async def test_provider_failure_reaches_loop_and_harness_results() -> None:
    result = await run_agent_loop(
        [UserMessage(content="continue")],
        AgentContext(),
        AgentLoopConfig(
            provider=_FailingProvider(),
            max_step_retries=0,
        ),
        lambda _event: None,
    )

    assert result.termination_reason is TerminationReason.PROVIDER_ERROR
    assert result.error_detail == "Provider error: temporarily unavailable"
    assert result.provider_failure is not None
    assert result.provider_failure.exception_type == "_ServiceUnavailable"
    assert result.provider_failure.status_code == 503

    harness_result = _build_structured_result(result)
    assert harness_result.provider_failure == result.provider_failure
