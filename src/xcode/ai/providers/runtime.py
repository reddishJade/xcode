from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, sleep
from collections.abc import Callable
from typing import TypeVar

import tenacity

"""Provider 运行时：重试、限速、API 错误处理。"""

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
    from openai import APIStatusError

    if isinstance(exc, APIStatusError):
        code = exc.status_code
        msg = API_ERROR_MESSAGES.get(code)
        if msg:
            return f"{msg} (HTTP {code}): {exc.message}"
        return f"API returned abnormal status (HTTP {code}): {exc.message}"
    return f"Request failed: {exc}"


def is_transient_provider_error(exc: BaseException) -> bool:
    from openai import APIStatusError, APITimeoutError, APIConnectionError

    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 529)

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    msg = str(exc).lower()
    # 临时性错误关键词（基于 HTTP 标准和经验值）
    # - timeout: 网络超时或服务端处理超时
    # - connection reset/refused: 网络连接问题
    # - 429: Too Many Requests（速率限制）
    # - 500/502/503: 服务端临时故障
    # - 529: Cloudflare 限流（非标准状态码，部分 CDN 使用）
    # - temporary: 服务端明确标识的临时错误
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
    """处理重试和本地限速的 provider 运行时。"""

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
        retry_policy = tenacity.retry_if_exception(is_transient_provider_error)

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
