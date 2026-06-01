from __future__ import annotations


"""Harness 层类型系统：错误体系。"""


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
