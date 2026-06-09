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
from .openai import OpenAIChatProvider, OpenAIResponsesProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "openai_chat": OpenAIChatProvider,
    "openai_responses": OpenAIResponsesProvider,
    "chatglm_chat": ChatGLMProvider,
    "deepseek_chat": DeepSeekProvider,
    "mimo_chat": MiMoProvider,
}

__all__ = [
    "ChatGLMProvider",
    "DeepSeekProvider",
    "FauxProvider",
    "FauxResponse",
    "MiMoProvider",
    "OpenAIChatProvider",
    "OpenAIResponsesProvider",
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
