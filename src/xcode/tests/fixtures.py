from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xcode.ai.events import ProviderEvent
from xcode.ai.providers.faux import FauxProvider


class FakeProvider(FauxProvider):
    """Lightweight alias for FauxProvider.

    Supports ProviderEvent lists, list-of-lists, or callable factories.
    Delegates to the consolidated FauxProvider implementation.
    """

    def __init__(
        self,
        events: list[ProviderEvent]
        | list[list[ProviderEvent]]
        | Callable[[list[Any], list[Any]], list[ProviderEvent]],
    ) -> None:
        super().__init__(response_spec=events)
