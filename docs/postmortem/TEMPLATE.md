# 事故标题

Date: YYYY-MM-DD
Status: Draft
Severity: S1 | S2 | S3

## Summary

用三至五句话说明发生了什么、影响谁、持续多久以及最终如何恢复。

## Impact

- 用户可见影响：
- 数据或安全影响：
- 受影响版本/平台：
- 开始与结束时间：

## Detection

说明最初由用户、监控、测试还是开发者发现，以及检测延迟。

## Timeline

- HH:MM — 事件
- HH:MM — 事件

## Root cause

描述技术根因和促成条件。区分直接触发、潜在设计缺陷与组织流程问题。

## Violated invariant

指出 `docs/architecture.md` 中被破坏的不变量；若不存在，先补充架构不变量。

## Why tests did not catch it

说明验证了哪个错误表面，以及缺少哪条真实 loader、process、session 或 UI 路径。

## Resolution

说明止血、修复、数据恢复和发布过程。

## Corrective actions

| Action | Owner | Status | Validation |
|---|---|---|---|
| 添加 assembled regression test | TBD | Open | 测试路径 |
| 修复根因 | TBD | Open | 测试或观测证据 |

## Lessons

记录可推广到其他模块的工程判断，以及需要创建或更新的 Agent Note。
