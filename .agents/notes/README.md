# Agent Notes

Agent Notes 保存代码本身无法完整表达的工程判断，让后续开发者和 agent 能够
继承决策、约束、拒绝方案与替代关系。

## 何时创建

- 改变跨层不变量、公共 schema 或产品组合方式；
- 明确放弃一个看似合理的方向；
- 一次迁移需要解释为何直接删除旧调用面；
- 事故或性能数据促成长期架构决定。

普通实现细节、临时 TODO 和提交摘要不需要 Note。

## 状态

- `Proposed`：正在讨论，不能作为实现约束；
- `Accepted`：已批准但尚未全部实现；
- `Implemented`：代码与测试已落实；
- `Rejected`：评估后不采用；
- `Superseded`：被另一 Note 明确替代；
- `Archived`：历史背景仍有价值，但不再约束当前实现。

## 文件格式

文件名为 `NNNN-short-title.md`，正文必须包含：

```text
# 标题
Status: Implemented
Date: YYYY-MM-DD

## Context
## Decision
## Consequences
## Validation
## Supersedes
```

`Supersedes` 没有内容时写 `None`。状态变化应直接修改原 Note；替代旧决策时，
新旧两份 Note 都要写明关系。Note 不用于维持旧实现兼容层。
