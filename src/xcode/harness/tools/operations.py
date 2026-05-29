from __future__ import annotations

from typing import Protocol


class BashOperations(Protocol):
    """Bash 执行操作的抽象接口。"""

    async def exec(
        self,
        command: str,
        cwd: str,
        timeout: float | None = None,
    ) -> str: ...


class ReadOperations(Protocol):
    """文件读取操作的抽象接口。"""

    async def read_file(self, path: str) -> str: ...

    def exists(self, path: str) -> bool: ...

    def size(self, path: str) -> int: ...


class WriteOperations(Protocol):
    """文件写入操作的抽象接口。"""

    async def write_file(self, path: str, content: str) -> None: ...

    async def make_dir(self, path: str) -> None: ...


class EditOperations(Protocol):
    """文件编辑操作的抽象接口。"""

    async def read_file(self, path: str) -> tuple[str, str]: ...

    async def write_file(self, path: str, content: str, encoding: str) -> None: ...


class GlobOperations(Protocol):
    """文件搜索操作的抽象接口。"""

    def glob(self, pattern: str, base: str, max_results: int) -> list[str]: ...


class GrepOperations(Protocol):
    """文本搜索操作的抽象接口。"""

    def grep(
        self, pattern: str, base: str, glob: str | None, max_results: int
    ) -> str: ...


class LsOperations(Protocol):
    """目录列表操作的抽象接口。"""

    def list_dir(self, path: str, limit: int) -> str: ...
