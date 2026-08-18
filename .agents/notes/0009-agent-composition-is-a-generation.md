# Agent composition is a generation
Status: Implemented
Date: 2026-08-18

## Context

`AgentHarness` 曾分别持有可变的 provider、registry、`AgentConfig`、runtime context、
request hygiene 和 gate。turn 开始时再临时构造 `TurnSnapshot`，只能复制其中四项；
`/model` 会直接赋值 provider 或修改 fallback wrapper，权限设置则旁路写入
`_gate._permission_policy`。因此同一 run 可以从不同时间点读取到互不匹配的能力与
策略，也无法从持久化请求判断当时运行的是哪套完整装配。

## Decision

provider、fallback provider、工具 registry、agent config、静态 gate 策略、
`RequestAssembler` 和 runtime context provider 必须先组成冻结的
`AgentComposition`，并取得唯一 `generation_id`，之后才能构造 harness。

会话 inbox、取消、压缩、hook、audit、session grant 和 session identity 留在
`AgentRuntimeConfig`；它们是有生命周期的服务，不属于产品 generation。工具
schema、collector registry、gate mappings 和 `AgentConfig` 在发布时复制或冻结。

run 获取所有权和 composition 替换共享同一同步边界。一个 run 捕获一次
composition 与有效 provider，所有 step 使用该 generation。模型或静态权限变更
只能创建并原子发布新 generation，active run 期间拒绝替换。删除 provider setter、
fallback provider 的原地 `replace_primary()`、`TurnSnapshot` 和私有 gate 字段写入；
不提供旧构造签名或适配层。

每次 `provider_request` 必须记录 `composition_id`，并将其纳入请求指纹。

## Consequences

- 产品能力和策略只能在明确的发布边界改变，不会在一个 run 中逐项漂移。
- runtime services 与产品配置的所有权分离，session 切换不需要伪造新产品装配。
- fallback wrapper 属于某一 generation；主模型变更会创建新的 wrapper 和干净的
  容灾状态。
- 动态增删工具、collector 或配置字段必须发布新 composition，不能修改已暴露
  集合。
- 当前只允许替换 main composition；subagent 的独立 composition 与 continuation
  由后续 session-backed subagent 决策实现。

## Validation

- composition 测试覆盖 schema/ruleset 发布后隔离、冻结 AgentConfig、collector
  registry 快照和 provider replacement generation。
- 真实 `build_app()` 测试断言 provider request 的 `composition_id` 等于 run 使用的
  generation。
- Ruff、Pyright 与完整测试套件验证旧 harness 构造和原地修改路径已全部删除。

## Supersedes

None
