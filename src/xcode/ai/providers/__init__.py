"""Provider 适配器注册。"""

from .chatglm import ChatGLMProvider
from .deepseek import DeepSeekProvider
from .factory import ProviderBundle, ProviderSettings, build_provider_bundle
from .faux import (
    FauxProvider,
    FauxResponse,
    faux_final,
    faux_text,
    faux_thinking,
    faux_tool_call,
    faux_usage,
    register_faux_provider,
)
from .mimo import MiMoProvider
from .openai import OpenAIChatProvider
from ._registry import PROVIDER_REGISTRY

__all__ = [
    "ChatGLMProvider",
    "DeepSeekProvider",
    "FauxProvider",
    "FauxResponse",
    "MiMoProvider",
    "OpenAIChatProvider",
    "ProviderBundle",
    "ProviderSettings",
    "PROVIDER_REGISTRY",
    "build_provider_bundle",
    "faux_final",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "faux_usage",
    "register_faux_provider",
]
