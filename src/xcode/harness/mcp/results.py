"""MCP 工具结果校验与宿主内容映射。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import cast
from urllib.parse import urlparse

import jsonschema
from jsonschema.validators import validator_for

from xcode.agent.types import FileContent, ImageContent
from xcode.harness.skills import (
    AGENT_CONTENT_BLOCKS_METADATA_KEY,
    ToolOutput,
)
from xcode.harness.session import JsonValue

from .client import redact_mcp_text

MCP_RESULT_METADATA_KEY = "mcp_result"


def convert_mcp_tool_result(
    response: Mapping[str, object],
    output_schema: object,
) -> ToolOutput:
    """校验 MCP 调用结果并映射为宿主工具输出。"""
    content = response.get("content", [])
    rendered_parts: list[str] = []
    host_blocks: list[ImageContent | FileContent] = []
    protocol_errors: list[str] = []
    is_error = response.get("isError", False)
    if not isinstance(is_error, bool):
        protocol_errors.append("MCP tool result isError must be a boolean")
        is_error = True

    if not isinstance(content, list):
        protocol_errors.append("MCP tool result content must be a list")
        raw_content: list[object] = [content]
    else:
        raw_content = content

    for index, block in enumerate(raw_content):
        rendered, host_block, error = _convert_content_block(index, block)
        if rendered:
            rendered_parts.append(rendered)
        if host_block is not None:
            host_blocks.append(host_block)
        if error is not None:
            protocol_errors.append(error)

    structured_content, structured_error = _structured_content(response)
    if structured_error is not None:
        protocol_errors.append(structured_error)

    validation_status, validation_error = _validate_structured_content(
        structured_content,
        output_schema,
        is_error,
    )
    if validation_error is not None:
        protocol_errors.append(validation_error)

    if structured_content is not None:
        serialized = json.dumps(
            structured_content,
            ensure_ascii=False,
            sort_keys=True,
        )
        if not any(serialized == part.strip() for part in rendered_parts):
            rendered_parts.append(f"[MCP structuredContent]\n{serialized}")

    if protocol_errors:
        rendered_parts.extend(
            f"[MCP result error] {error}" for error in protocol_errors
        )

    details = {
        "content": _redacted_json_value(raw_content),
        "structuredContent": _redacted_json_value(structured_content),
        "outputSchema": _redacted_json_value(output_schema),
        "validation": {
            "status": validation_status,
            "error": (
                redact_mcp_text(validation_error)
                if validation_error is not None
                else None
            ),
        },
        "isError": is_error,
        "protocolErrors": _redacted_json_value(protocol_errors),
    }
    metadata: dict[str, object] = {MCP_RESULT_METADATA_KEY: details}
    if host_blocks:
        metadata[AGENT_CONTENT_BLOCKS_METADATA_KEY] = host_blocks

    text = "\n".join(part for part in rendered_parts if part)
    return ToolOutput(
        redact_mcp_text(text),
        metadata=metadata,
        is_error=is_error or bool(protocol_errors),
    )


def _structured_content(
    response: Mapping[str, object],
) -> tuple[dict[str, JsonValue] | None, str | None]:
    """提取协议要求为对象的 structuredContent。"""
    value = response.get("structuredContent")
    if value is None:
        return None, None
    if not isinstance(value, dict):
        return None, "MCP structuredContent must be a JSON object"
    return cast(dict[str, JsonValue], dict(value)), None


def _validate_structured_content(
    structured_content: dict[str, JsonValue] | None,
    output_schema: object,
    is_error: bool,
) -> tuple[str, str | None]:
    """按工具 outputSchema 校验结构化结果。"""
    if output_schema is None:
        return "not_declared", None
    if not isinstance(output_schema, dict):
        return "invalid_schema", "MCP outputSchema must be a JSON object"
    if structured_content is None:
        if is_error:
            return "not_applicable", None
        return (
            "invalid",
            "MCP tool declares outputSchema but returned no structuredContent",
        )
    try:
        validator_class = validator_for(output_schema)
        validator_class.check_schema(output_schema)
        validator_class(output_schema).validate(structured_content)
    except jsonschema.SchemaError as exc:
        return "invalid_schema", f"MCP outputSchema is invalid: {exc.message}"
    except jsonschema.ValidationError as exc:
        return (
            "invalid",
            f"MCP structuredContent violates outputSchema at "
            f"{exc.json_path}: {exc.message}",
        )
    return "valid", None


def _convert_content_block(
    index: int,
    value: object,
) -> tuple[str, ImageContent | FileContent | None, str | None]:
    """转换单个 MCP content block，并保留可诊断错误。"""
    if not isinstance(value, Mapping):
        return _unsupported_block(index, value, "content block must be an object")

    block = {str(key): item for key, item in value.items()}
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        if isinstance(text, str):
            return text, None, None
        return _unsupported_block(index, block, "text block has no string text")
    if block_type == "image":
        return _image_block(index, block)
    if block_type == "audio":
        return _audio_block(index, block)
    if block_type == "resource_link":
        return _resource_link_block(index, block)
    if block_type == "resource":
        return _embedded_resource_block(index, block)
    return _unsupported_block(
        index,
        block,
        f"unsupported content type {block_type!r}",
    )


def _image_block(
    index: int,
    block: dict[str, object],
) -> tuple[str, ImageContent | None, str | None]:
    """将 MCP image 映射为宿主 ImageContent。"""
    data = block.get("data")
    mime_type = block.get("mimeType")
    if not isinstance(data, str) or not isinstance(mime_type, str):
        rendered, _, error = _unsupported_block(
            index,
            block,
            "image block requires string data and mimeType",
        )
        return rendered, None, error
    source = {
        "type": "base64",
        "media_type": mime_type,
        "data": data,
        "annotations": _redacted_json_value(block.get("annotations")),
    }
    return (
        f"[MCP image content: {mime_type}; data preserved in structured result]",
        ImageContent(source=source),
        None,
    )


def _audio_block(
    index: int,
    block: dict[str, object],
) -> tuple[str, FileContent | None, str | None]:
    """将 MCP audio 映射为通用宿主文件内容。"""
    data = block.get("data")
    mime_type = block.get("mimeType")
    if not isinstance(data, str) or not isinstance(mime_type, str):
        rendered, _, error = _unsupported_block(
            index,
            block,
            "audio block requires string data and mimeType",
        )
        return rendered, None, error
    source = {
        "type": "base64",
        "media_type": mime_type,
        "annotations": _redacted_json_value(block.get("annotations")),
    }
    return (
        f"[MCP audio content: {mime_type}; data preserved as file content]",
        FileContent(
            source=source,
            filename="mcp-audio",
            file_data=data,
        ),
        None,
    )


def _resource_link_block(
    index: int,
    block: dict[str, object],
) -> tuple[str, FileContent | None, str | None]:
    """将 MCP resource_link 映射为 URI 文件引用。"""
    uri = block.get("uri")
    name = block.get("name")
    if not isinstance(uri, str) or not isinstance(name, str):
        rendered, _, error = _unsupported_block(
            index,
            block,
            "resource_link block requires string uri and name",
        )
        return rendered, None, error
    mime_type = block.get("mimeType")
    redacted_uri = redact_mcp_text(uri)
    redacted_name = redact_mcp_text(name)
    source = {
        "type": "url",
        "url": redacted_uri,
        "media_type": mime_type if isinstance(mime_type, str) else None,
        "annotations": _redacted_json_value(block.get("annotations")),
        "description": _redacted_json_value(block.get("description")),
    }
    return (
        f"[MCP resource link: {redacted_name} ({redacted_uri})]",
        FileContent(
            source=source,
            file_id=redacted_uri,
            filename=redacted_name,
        ),
        None,
    )


def _embedded_resource_block(
    index: int,
    block: dict[str, object],
) -> tuple[str, FileContent | None, str | None]:
    """将 MCP embedded resource 映射为宿主文件内容。"""
    resource = block.get("resource")
    if not isinstance(resource, Mapping):
        rendered, _, error = _unsupported_block(
            index,
            block,
            "resource block requires an object resource",
        )
        return rendered, None, error
    normalized = {str(key): item for key, item in resource.items()}
    uri = normalized.get("uri")
    if not isinstance(uri, str):
        rendered, _, error = _unsupported_block(
            index,
            block,
            "embedded resource requires a string uri",
        )
        return rendered, None, error

    text = normalized.get("text")
    blob = normalized.get("blob")
    if isinstance(text, str):
        data = redact_mcp_text(text)
        source_type = "text"
    elif isinstance(blob, str):
        data = blob
        source_type = "base64"
    else:
        rendered, _, error = _unsupported_block(
            index,
            block,
            "embedded resource requires string text or blob",
        )
        return rendered, None, error

    mime_type = normalized.get("mimeType")
    redacted_uri = redact_mcp_text(uri)
    source = {
        "type": source_type,
        "uri": redacted_uri,
        "media_type": mime_type if isinstance(mime_type, str) else None,
        "annotations": _redacted_json_value(normalized.get("annotations")),
    }
    return (
        f"[MCP embedded resource: {redacted_uri}]",
        FileContent(
            source=source,
            file_id=redacted_uri,
            filename=_filename_from_uri(redacted_uri),
            file_data=data,
        ),
        None,
    )


def _unsupported_block(
    index: int,
    block: object,
    reason: str,
) -> tuple[str, None, str]:
    """生成包含完整脱敏原始块的 unsupported 诊断。"""
    diagnostic = {
        "index": index,
        "reason": reason,
        "block": _redacted_json_value(block),
    }
    rendered = json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)
    return f"[unsupported MCP content block] {rendered}", None, reason


def _filename_from_uri(uri: str) -> str:
    """从资源 URI 提取稳定文件名。"""
    path = urlparse(uri).path
    name = PurePosixPath(path).name
    return name or uri


def _redacted_json_value(value: object) -> object:
    """递归脱敏可进入宿主详情模型的 JSON 值。"""
    if isinstance(value, str):
        return redact_mcp_text(value)
    if value is None or isinstance(value, int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_redacted_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _redacted_json_value(item) for key, item in value.items()}
    return redact_mcp_text(str(value))
