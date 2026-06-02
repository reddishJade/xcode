"""Provider 适配器注册。"""

from .anthropic import AnthropicProvider
from .chatglm import ChatGLMProvider
from .deepseek import DeepSeekProvider
from .factory import ProviderBundle, ProviderSettings, build_provider_bundle
from .faux import FauxProvider
from .mimo import MiMoProvider
from .openai import OpenAIChatProvider, OpenAIResponsesProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "openai_chat": OpenAIChatProvider,
    "openai_responses": OpenAIResponsesProvider,
    "anthropic_messages": AnthropicProvider,
    "chatglm_chat": ChatGLMProvider,
    "deepseek_chat": DeepSeekProvider,
    "mimo_chat": MiMoProvider,
}

__all__ = [
    "AnthropicProvider",
    "ChatGLMProvider",
    "DeepSeekProvider",
    "FauxProvider",
    "MiMoProvider",
    "OpenAIChatProvider",
    "OpenAIResponsesProvider",
    "ProviderBundle",
    "ProviderSettings",
    "PROVIDER_REGISTRY",
    "build_provider_bundle",
]
