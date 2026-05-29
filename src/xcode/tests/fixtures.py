from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Union

from xcode.harness.agent_runtime.provider import ModelProvider
from xcode.harness.agent_runtime.events import ProviderEvent
from xcode.harness.skills import ToolSpec


class FakeProvider(ModelProvider):
    """A strict FakeProvider that only accepts explicit ProviderEvent lists or factories."""

    def __init__(
        self,
        events: Union[
            list[ProviderEvent],
            list[list[ProviderEvent]],
            Callable[[list[dict], list[ToolSpec]], list[ProviderEvent]],
        ],
    ) -> None:
        self.events = events
        if isinstance(events, list):
            if events and isinstance(events[0], list):
                self._iterator = iter(events)
            else:
                self._iterator = iter([events])
        else:
            self._iterator = None

    async def stream(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
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
