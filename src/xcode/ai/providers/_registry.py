"""Provider 注册表。

独立模块以避免 factory.py 与 providers/__init__.py 之间的循环依赖。
"""

from .chatglm import ChatGLMProvider
from .deepseek import DeepSeekProvider
from .faux import FauxProvider
from .mimo import MiMoProvider
from .openai import OpenAIChatProvider

__all__ = [
    "ChatGLMProvider",
    "DeepSeekProvider",
    "FauxProvider",
    "MiMoProvider",
    "OpenAIChatProvider",
    "PROVIDER_REGISTRY",
]

PROVIDER_REGISTRY: dict[str, type] = {
    "openai_chat": OpenAIChatProvider,
    "chatglm_chat": ChatGLMProvider,
    "deepseek_chat": DeepSeekProvider,
    "mimo_chat": MiMoProvider,
    "faux_chat": FauxProvider,
}
