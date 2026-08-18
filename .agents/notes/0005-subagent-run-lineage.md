# Subagent run lineage is durable
Status: Implemented
Date: 2026-08-18

## Context

子代理进度原先只存在于父工具的临时更新和最终文本中，无法可靠回答哪个任务
启动、失败、取消，或某个父工具调用创建了哪些 child run。

## Decision

每次委派创建 batch ID，每项任务创建 run ID。运行状态以 `subagent_run` 事件
追加到父 session，包含任务索引、类型、状态、摘要或错误；父 tool result 的
subagent render intent 保存 batch ID 和 run IDs。

## Consequences

UI、诊断和后续调度可以基于稳定 ID，而非解析进度字符串。子代理继续继承父
权限门控，但任务上下文必须显式传入。

## Validation

测试覆盖 started/completed/failed、父 session 归属和父 tool result 到 child run
的关联，真实 app 组合负责注入唯一 lifecycle sink。

## Supersedes

None
