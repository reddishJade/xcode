"""图片检测与读取。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Protocol

import filetype
from PIL import Image

from xcode.harness.skills import ToolOutput


class _ImageFileOperations(Protocol):
    def read_bytes(self, path: Path) -> bytes: ...


def _detect_image(path: Path, operations: _ImageFileOperations) -> str | None:
    try:
        buf = operations.read_bytes(path)
    except Exception:
        return None
    return filetype.guess_mime(buf)


def _read_image(
    path: Path, display_path: str, mime: str, operations: _ImageFileOperations
) -> str:
    data = operations.read_bytes(path)
    img = Image.open(BytesIO(data))
    orig_w, orig_h = img.width, img.height
    max_dim = 2000
    if orig_w > max_dim or orig_h > max_dim:
        ratio = min(max_dim / orig_w, max_dim / orig_h)
        new_size = (int(orig_w * ratio), int(orig_h * ratio))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        save_format = img.format or "PNG"
        resized.save(buf, format=save_format)
        new_w, new_h = resized.width, resized.height
        data = buf.getvalue()
        _img_mime: dict[str, str] = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime = _img_mime.get(save_format.lower(), mime)
    else:
        new_w, new_h = orig_w, orig_h
    b64 = base64.b64encode(data).decode("ascii")
    hint = (
        f" (resized from {orig_w}x{orig_h} to {new_w}x{new_h})"
        if orig_w > max_dim or orig_h > max_dim
        else ""
    )
    return ToolOutput(
        f"Read image file [{mime}]{hint}\nImage data is available in metadata.",
        metadata={"image": {"mime": mime, "data": b64}},
    )
