# Local execution only
Status: Implemented
Date: 2026-08-18

## Context

`ExecutionEnv` 同时包装 filesystem、shell 和 run，看似为远程或容器 provider
预留 seam，但 Xcode 产品没有这些目标。未使用的执行世界抽象扩大了 API 和
组合状态空间。

## Decision

Xcode 只支持本地执行。Bash 直接消费 `Shell`，文件工具直接消费 `FileSystem`；
生产实现固定为本地 subprocess 和本地 filesystem。不建设容器、远程 shell、
远程 filesystem 或远程 workspace provider。

## Consequences

本地窄协议仍可用于确定性测试，但不承诺环境可迁移性。需要 OS 隔离时由运行
Xcode 的外部环境提供。

## Validation

`ExecutionEnv` 和 `SubprocessExecutionEnv` 已删除；Bash 测试验证直接注入本地
Shell，真实组合测试验证默认产品仍可组装。

## Supersedes

None
