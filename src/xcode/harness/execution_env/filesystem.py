from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol


class FileSystem(Protocol):
    def exists(self, path: Path) -> bool: ...
    def is_file(self, path: Path) -> bool: ...
    def is_dir(self, path: Path) -> bool: ...
    def size(self, path: Path) -> int: ...
    def read_bytes(self, path: Path) -> bytes: ...
    def write_bytes(self, path: Path, data: bytes) -> None: ...
    def mkdir(self, path: Path) -> None: ...
    def remove_file(self, path: Path) -> None: ...
    def iter_lines(self, path: Path) -> Iterator[str]: ...
    def read_dir_entries(self, path: Path) -> list[tuple[str, bool]]: ...
    def read_head(self, path: Path, n: int) -> bytes: ...


class LocalFileSystem:
    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_file(self, path: Path) -> bool:
        return path.is_file()

    def is_dir(self, path: Path) -> bool:
        return path.is_dir()

    def size(self, path: Path) -> int:
        return path.stat().st_size

    def read_bytes(self, path: Path) -> bytes:
        return path.read_bytes()

    def write_bytes(self, path: Path, data: bytes) -> None:
        path.write_bytes(data)

    def mkdir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def remove_file(self, path: Path) -> None:
        path.unlink()

    def iter_lines(self, path: Path) -> Iterator[str]:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                yield line.rstrip("\r\n")

    def read_dir_entries(self, path: Path) -> list[tuple[str, bool]]:
        entries = [(entry.name, entry.is_dir()) for entry in path.iterdir()]
        entries.sort(key=lambda x: x[0].casefold())
        return entries

    def read_head(self, path: Path, n: int) -> bytes:
        with open(path, "rb") as f:
            return f.read(n)
