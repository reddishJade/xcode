# Xcode Harness — 应用运行时

Harness 是 Xcode 最外层的应用装配与运行时层，把 Agent 循环、LLM provider、工具系统、安全策略和观测基础设施组合为可运行的应用。

它负责三件事：读取配置并装配组件（`app.py` / `assembly.py`）、定义工具协议和执行边界（`tools/` / `skills.py` / `adapters/`）、提供会话管理、审计日志、权限判定等运行时基础设施（`session.py` / `observability/` / `agent_runtime/`）。

`StructuredAgent` 位于 `agent_runtime/` 中，是 harness 对 agent loop 的结构化封装——增加了上下文压缩、重复调用检测、subagent 管理等生产环境需要的功能。这一层依赖 agent 和 ai 层，但不被它们反向依赖。
