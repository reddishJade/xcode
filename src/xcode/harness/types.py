from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

"""Harness 层类型系统：错误体系 + Result[T, E] 模式。"""

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T, E]):
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True)
class Err(Generic[T, E]):
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:
        raise (
            self.error
            if isinstance(self.error, Exception)
            else RuntimeError(str(self.error))
        )


Result = Ok[T, E] | Err[T, E]


def ok[T](value: T) -> Ok[T, Any]:
    return Ok(value)


def err[E](error: E) -> Err[Any, E]:
    return Err(error)


class AgentHarnessError(Exception):
    """Harness 层基础错误。"""

    def __init__(
        self, message: str, code: str = "UNKNOWN", cause: Exception | None = None
    ) -> None:
        self.code = code
        self.cause = cause
        super().__init__(message)


class SessionError(AgentHarnessError):
    """会话错误。"""

    pass


class CompactionError(AgentHarnessError):
    """压缩错误。"""

    pass


class ProviderError(AgentHarnessError):
    """Provider 调用错误。"""

    pass


class ToolExecutionError(AgentHarnessError):
    """工具执行错误。"""

    pass
