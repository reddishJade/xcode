from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Iterator, Union, cast
from xcode.agent.types import ToolDefinition
from xcode.harness.agent_runtime.provider import ModelProvider
from xcode.harness.agent_runtime.events import ProviderEvent


class FakeProvider(ModelProvider):
    """A strict FakeProvider that only accepts explicit ProviderEvent lists or factories."""

    _iterator: Iterator[list[ProviderEvent]] | None

    def __init__(
        self,
        events: Union[
            list[ProviderEvent],
            list[list[ProviderEvent]],
            Callable[[list[Any], list[Any]], list[ProviderEvent]],
        ],
    ) -> None:
        self.events = events
        self._iterator = None
        if isinstance(events, list):
            if events and isinstance(events[0], list):
                self._iterator = iter(cast(list[list[ProviderEvent]], events))
            else:
                self._iterator = iter([cast(list[ProviderEvent], events)])

    async def stream(
        self,
        messages: list[Any],
        tools: list[ToolDefinition],
    ) -> AsyncIterator[ProviderEvent]:
        if callable(self.events):
            res_list = self.events(messages, tools)
        elif self._iterator is not None:
            try:
                res_list = next(self._iterator)
            except StopIteration:
                res_list = []
        else:
            raise TypeError(
                "events must be a list of ProviderEvents or a callable factory"
            )

        for event in res_list:
            yield event
