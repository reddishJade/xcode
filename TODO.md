# Xcode 已知问题与待办

## LLM-as-judge eval 未实际生效

`src/xcode/evals/graders.py:run_llm_judge()` 从 `app.agent.provider` 获取 judge provider，但 `ModelProvider` 协议仅有 `stream()` 方法，而 `run_llm_judge` 只检查 `ask()` 和 `run()` 接口，导致始终返回空 tuple。内置 eval 套件的 `llm_judge_criteria` 因此不会被评分。

需要：改造 `run_llm_judge` 使用 `ModelProvider.stream()`，或单独注入一个 judge provider。

## tool_catalog.py 新增 builder 无代码强制

`src/xcode/cli/tool_catalog.py` 的 docstring 要求 `build_*_tools()` 必须注册入 `_builders()`，但无类型系统或测试约束。新增 builder 后工具目录会遗漏。

需要：为 `_builders` 列表添加注册机制或编译期校验。

## daemon 生命周期未接入 build_app

`assembly.py:load_opt_in_services()` 构造 `HeartbeatDaemon` 但未调用 `start()`，`XcodeApp.close()` 也未调用 `stop()`。调用方需要手动管理 daemon 启动/停止。

需要：`build_app()` 在 daemon 构造后调用 `start()`，`close()` 注册 `stop()` 为 closer。

## PROVIDER_REGISTRY 不完整

`src/xcode/ai/providers/_registry.py` 注册了 5 种 transport：`openai_chat`、`chatglm_chat`、`deepseek_chat`、`mimo_chat`、`faux_chat`。但 `ProviderTransport` 类型包含 `anthropic_messages`，没有对应的注册实现。新 transport 只需在 `_registry.py` 添加条目，缺少文档说明。

需要：补充文档注释说明注册流程，或移除未实现的 transport 类型。

## ProviderMetricsMixin 调用链不一

`metrics.py:ProviderMetricsMixin` 定义了 `_record_usage` 让子类覆写，但 `OpenAIChatProvider` 直接在 `_intercept_stream` 闭包内调用 `self._record_usage`，而其他 provider（如 `DeepSeekProvider`）继承后可能有自己的一套拦截逻辑。指标记录分散在各子类中。

需要：统一 provider 的 usage 记录入口，确保 metrics 在所有 provider 下一致。

## memory consolidate 质量门宽松

`experimental/memory.py:consolidate()` 只检查 `##` 标题和必需字段关键字存在即接受，不做 `_content_quality_check` 的长度和重复校验。低质量压缩摘要可能通过 `consolidate` 写入 MEMORY.md，绕过 `add_memory_block` 的质量门和冲突合并。

需要：`consolidate()` 复用 `add_memory_block` 的校验路径。
