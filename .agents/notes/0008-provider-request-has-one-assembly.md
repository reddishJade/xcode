# Provider request has one assembly
Status: Implemented
Date: 2026-08-18

## Context

模型请求曾由 `_provider.call_provider()` 临时拼接 `request_prefix`、session
messages、context collectors、context assembler、`transform_context`、message
converter 和 tool schemas。`transform_context` 是一个无来源约束的末端改写口，
而 `before_provider_request` 只看到两个可变 list；新增模型输入可以绕开上下文
来源记录，也无法证明审计内容与 provider 实际收到的 envelope 是同一份结果。

## Decision

所有普通 agent provider 调用必须先生成一个 `RequestAssembly`。固定组装顺序为：

1. 合并 scoped request prefix 与当前 session surface；
2. 收集结构化 context blocks；
3. 由 `ContextAssembler` 决定注入和丢弃；
4. 应用唯一的确定性 `RequestHygiene`；
5. 编码 wire messages；
6. 从当前 agent tools 生成 tool definitions；
7. 固化 options、step 与 context provenance trace。

provider stream 和 `before_provider_request` 审计 hook 消费同一个 assembly。
`provider_request` 事件记录精确 messages、tools、provider、options、请求指纹，
以及每个动态 context block 的 source、target、included 决议、token 数和内容指纹。

`AgentLoopConfig.transform_context`、`convert_to_llm`、`context_collectors` 和
`context_assembler` 四个并行入口被直接删除，由唯一 `request_assembler` 取代；
不提供旧字段、适配器或双路径。

## Consequences

- provider 调用前不再存在通用的隐式消息改写入口。
- 请求卫生只改变本次 assembly，不修改 durable session surface。
- 动态上下文和 tool schema 与实际 wire envelope 一起被审计和指纹化。
- 自定义 agent 若需要不同组装规则，必须提供完整 `RequestAssembler`，而不是在
  发送前局部 patch messages。
- compaction summarizer 和 Goal judge 是明确的独立模型调用，不伪装成普通 agent
  request assembly。

## Validation

- assembly 契约测试覆盖 prefix/surface/context 顺序、included/dropped provenance、
  tool schema、请求卫生的非破坏性以及旧字段缺失。
- 真实 `build_app()` 测试比对 provider 收到的 envelope 与持久化
  `provider_request`。
- Ruff、Pyright 和完整测试套件验证所有调用点已一次迁移。

## Supersedes

None
