"""Provider 运行时：重试、限速、API 错误分类。

不依赖任何特定 provider 库，通过 Callable 注入客户端调用。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep
from typing import TypeVar

import tenacity

T = TypeVar("T")

API_ERROR_MESSAGES: dict[int, str] = {
    400: "Bad request: check JSON format, required parameters, model name, and multimodal file validity",
    401: "Invalid or expired API key, check your configuration",
    402: "Insufficient API balance, please top up and retry",
    403: "Access denied, create a new API key and ensure input safety",
    404: "Resource not found, verify the model/endpoint supports this capability",
    421: "Content blocked, avoid unsafe or sensitive input",
    429: "Too many requests, please retry later",
    500: "Server temporarily unavailable, please retry later",
    502: "Server temporarily unavailable (gateway error), please retry later",
    503: "Service temporarily unavailable (maintenance), please retry later",
}


def classify_api_error(exc: BaseException) -> str:
    """将异常分类为人类可读的错误消息。"""
    status_code = _try_extract_status_code(exc)
    if status_code is not None:
        msg = API_ERROR_MESSAGES.get(status_code)
        if msg:
            return f"{msg} (HTTP {status_code}): {exc}"
        return f"API returned abnormal status (HTTP {status_code}): {exc}"
    return f"Request failed: {exc}"


def _try_extract_status_code(exc: BaseException) -> int | None:
    """尝试从异常中提取 HTTP 状态码。"""
    for attr in ("status_code", "status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val <= 599:
            return val
    return None


def _is_transient_error(exc: BaseException) -> bool:
    """判断是否为可重试的临时性错误。

    基于 HTTP 状态码和异常特征，不依赖任何特定库的类型。
    """
    status_code = _try_extract_status_code(exc)
    if status_code is not None and status_code in (429, 500, 502, 503, 529):
        return True

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    msg = str(exc).lower()
    transient_keywords = [
        "timeout",
        "connection reset",
        "connection refused",
        "429",
        "500",
        "502",
        "503",
        "529",
        "temporary",
        "rate limit",
        "too many requests",
    ]
    for kw in transient_keywords:
        if kw in msg:
            return True
    return False


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 0.2
    backoff: float = 2.0
    max_delay_seconds: float = 2.0


@dataclass(frozen=True)
class RateLimitPolicy:
    min_interval_seconds: float = 0.0


class ProviderRuntime:
    """处理重试和本地限速的 provider 运行时。

    通过 run() 方法接受任意可调用对象，不绑定任何特定客户端库。
    """

    def __init__(
        self,
        retry: RetryPolicy | None = None,
        rate_limit: RateLimitPolicy | None = None,
        now: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self.retry = retry or RetryPolicy()
        self.rate_limit = rate_limit or RateLimitPolicy()
        self.now = now
        self.sleeper = sleeper
        self._last_call_at: float | None = None

    def run(self, operation: Callable[[], T]) -> T:
        retry_policy = tenacity.retry_if_exception(_is_transient_error)

        retrier = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(self.retry.max_attempts),
            wait=tenacity.wait_random_exponential(
                multiplier=self.retry.initial_delay_seconds,
                max=self.retry.max_delay_seconds,
            ),
            retry=retry_policy,
            reraise=True,
        )

        try:

            def wrapped_operation():
                self._wait_for_rate_limit()
                return operation()

            return retrier(wrapped_operation)
        except Exception as last_error:
            msg = classify_api_error(last_error)
            raise RuntimeError(msg) from last_error

    def _wait_for_rate_limit(self) -> None:
        interval = self.rate_limit.min_interval_seconds
        if interval <= 0:
            self._last_call_at = self.now()
            return
        current = self.now()
        if self._last_call_at is not None:
            wait_for = interval - (current - self._last_call_at)
            if wait_for > 0:
                self.sleeper(wait_for)
                current = self.now()
        self._last_call_at = current
