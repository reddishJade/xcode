"""Anthropic Messages API provider（尚未实现）。

注册在 PROVIDER_REGISTRY 中供后续实现，构造时立即报错。
"""

NOT_IMPLEMENTED_MSG = (
    "AnthropicProvider is not implemented. "
    "Select a different provider in your configuration."
)


class AnthropicProvider:
    """Anthropic Messages API 占位。选择此 provider 会立即抛出 RuntimeError。"""

    def __init__(self, api_key: str, model: str) -> None:
        raise RuntimeError(NOT_IMPLEMENTED_MSG)
