# Xcode AI — LLM 通信层

AI 层是 Agent 与 LLM 之间的翻译层，解决一个核心问题：如何用统一的协议表达不同模型的对话、推理和工具调用。

它定义了一套流事件协议（TextDelta、ReasoningDelta、ToolCallEvent、FinalMessage），将各模型的 stream delta 归一化为同一事件流，让上层 Agent 无需关心背后的模型是 OpenAI、Anthropic 还是 DeepSeek。`ProviderEvent` 联合类型是所有 provider 的输出契约。

`providers/` 下每个文件对应一个平台适配，共享 `openai_compat.py` 基类避免重复。`faux.py` 提供测试用假 provider，使上层测试不依赖真实 API。这一层不关心 Agent 循环如何组织，只负责与 LLM 的通信和编解码。
