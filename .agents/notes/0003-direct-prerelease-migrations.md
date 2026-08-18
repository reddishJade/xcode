# Direct prerelease migrations
Status: Implemented
Date: 2026-08-18

## Context

Xcode 尚未承诺稳定公共 API。在此阶段保留旧签名、别名解析、双写或旧 schema
会让临时设计永久进入主路径，并使测试同时验证多个产品。

## Decision

基础设计调整采用一次性迁移：同一提交修改定义、所有调用方、测试和文档，
随后删除旧实现。除非已经存在明确的外部稳定性承诺，不增加 compatibility
wrapper、legacy parser、dual write 或 deprecated parameter。

## Consequences

提交可能是 breaking change，但最终代码只有一个权威调用面。需要重新评估时，
创建新的 Agent Note，而不是在实现中预埋未使用的兼容层。

## Validation

session codec、replay、ExecutionEnv、Bash 参数和 edit_file 参数迁移均删除旧路径；
仓库调用方和完整测试套件在对应提交中同步通过。

## Supersedes

None
