# Subagents are session-backed identities
Status: Implemented
Date: 2026-08-18

## Context

旧 `subagent` 工具为每项任务直接构造一个临时 `Agent` 并调用 `prompt()`。父 session
只记录 started/completed 等摘要事件；child 的用户输入、模型请求、工具结果和回答
没有独立日志。进程退出后 run ID 无法恢复，也不存在 continuation。所谓 multi-agent
实质上仍是复制 prompt 后并发调用模型。

## Decision

每次委派先由 `TreeSessionRepo.spawn_child()` 创建带 `parent_id` 的独立 session，
再写入严格的 `subagent/descriptor`。descriptor 包含 child/parent session ID、
`one_shot` 或 `continuable` 模式、description、persona、provider model 和初始
composition ID。

`SubagentSessionManager` 是 child materialization 的唯一所有者。它为 child 构造
独立 inbox、surface、correlation、provider-request recorder、tool gate 和
`AgentComposition`。父 transcript 不会被复制；child 只接收显式任务和产品定义的
persona/tool filter。

并行任务只创建 one-shot child。continuable child 保留 durable session ID；
`subagent_continue` 把后续消息写入同一 child inbox。若当前进程没有 live
activation，manager 从 child session surface 冷恢复后继续。`subagent_list` 通过
session index 和 descriptor 枚举 direct children，不启动 child。

subagent model profile 由 manager 单独持有；运行时切换只影响之后创建或冷恢复的
activation，不修改已经发布的 child composition。

旧的裸 `Agent.prompt()` 路径、生命周期 sink 注入和无 mode 的 subagent schema
直接删除，不提供兼容调用。

## Consequences

- child 的输入、请求、工具与回答可以独立 replay、审计和继续。
- durable child identity 与单次 run ID 分离；一个 continuable session 可以产生多次
  turn/run。
- one-shot 完成后不保留 live activation，但 session 及 lineage 继续存在。
- 当前 creation 语义只有 spawn；不暗示继承父 conversation history。
- activation ownership、权限收窄、interrupt 与 teardown 顺序由下一项运行时决策
  在同一 session identity 之上实现。

## Validation

- 真实落盘测试验证 child index parent、descriptor、inbox lifecycle、provider
  request 和 final event 全部位于 child session。
- cold continuation 测试使用新 manager 恢复同一 child ID，并证明第二次 provider
  request 包含第一轮 child surface。
- 依赖边界测试保证通用 session manager 不反向依赖 coding product。
- Ruff、Pyright 与完整测试套件验证删除后的单一路径。

## Supersedes

0005-subagent-run-lineage.md
