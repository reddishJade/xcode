# Session ledger is authoritative
Status: Implemented
Date: 2026-08-18

## Context

编程式入口、REPL 和 TUI 曾分别参与 session 记录与恢复，模型请求内容也无法
从 transcript 精确核对。局部模块正确不能保证 resume、fork 和真实宿主一致。

## Decision

session transcript 是 append-only 事实账本，`SessionRecorder` 统一拥有写入，
`session/replay.py` 统一拥有重建。实际 provider 请求必须保存 messages、tools、
provider 参数和请求指纹。compaction 只能追加 epoch，不能重写历史。

## Consequences

CLI/TUI 只选择和展示 session。任何新模型可见内容都需要对应的持久化事件。
内存 history、Goal 和 contextual state 均视为日志投影。

## Validation

`test_app_composition.py` 验证真实 build、落盘、restart、resume 和第二轮请求；
session codec、recorder、surface projector 测试验证逐层契约。

## Supersedes

None
