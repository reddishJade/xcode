# Session Inbox owns pending input
Status: Implemented
Date: 2026-08-18

## Context

用户输入曾同时存在于 `Agent` 的 steer 队列、`SessionRunController` 的
follow-up 队列和 CLI/TUI 的 `pending_inject` 字段。进程退出或 session 切换会
丢失尚未消费的输入，而且 transcript 只能记录宿主认为已经开始的普通 turn，
无法准确说明某条输入何时进入模型。

## Decision

每个 session 只拥有一个 `SessionInbox`。输入先以 `inbox/inserted` 事件写入
当前 branch，再在模型边界以 `inbox/claimed` 事件消费；显式丢弃使用
`inbox/discarded`。`next_step` 和 `next_turn` 是仅有的调度 lane，运行控制器只
管理 active run identity、claim 边界和取消状态。

`inbox/claimed` 携带完整的 typed message、展示文本、来源、lane 和 run ID，
并且是 session surface 中用户及运行时注入消息的唯一事实来源。Agent、CLI 和
TUI 不再保存 pending 消息正文。宿主只根据 inbox 是否需要唤醒来启动运行。

旧的 `user` transcript entry、`collect` busy mode、`pending_inject`、
`take_follow_up` 和 Agent 私有 steer 队列被直接删除，不提供双写或读取兼容层。

## Consequences

- 输入在进入内存调度前已经可恢复，claim 时点也可审计。
- follow-up 会唤醒独立的新 run；宿主不再搬运消息正文。
- session 恢复会重建未 claim 输入，迟到的 steer 不会静默丢失。
- 命令以 `command` event 记录，不再伪装成模型可见用户消息。
- 新的 runtime composition 必须显式提供与 recorder 共用 store 的
  `SessionInbox`。

## Validation

- run controller 测试覆盖 active steer、late steer、follow-up、interrupt、重建
  恢复与重叠 run 拒绝。
- session surface、history、recorder、CLI goal 调度和完整测试套件覆盖新 schema。
- Ruff 与 Pyright 验证跨层调用面已经完成直接迁移。

## Supersedes

None
