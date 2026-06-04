# Xcode Agent — 思考循环

Agent 是 Xcode 最内层的循环抽象，只关心一件事：模型如何推理、决策、观察、重复。

它定义了一套中性的消息类型（SystemMessage、UserMessage、AssistantMessage、ToolResultMessage）和事件协议（BeforeToolCall、AfterToolCall、TurnStart、TurnEnd），让 loop 的实现与具体工具、LLM 接口、运行时配置解耦。Provider 和工具执行通过合约注入，agent 层本身不依赖 harness 或 cli 的任何类型。

这一层的设计原则是：loop 逻辑一旦稳定就不应该因外部变化而修改。新增工具、切换模型、改变安全策略都不应该触及这里的代码。
