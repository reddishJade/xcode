# Session surface is a log projection
Status: Implemented
Date: 2026-08-18

## Context

Compaction 曾直接替换运行中的 `messages` 列表，并把另一份 Markdown checkpoint
当作 restart 的恢复来源。内存历史、transcript 与 checkpoint 可能形成三个不同
版本，自动 compaction 后的下一轮也不能保证继续使用刚生成的压缩结果。

## Decision

Current surface 只由 session branch 投影。每个 compaction event 保存完整的类型化
replacement、被替换的 branch prefix entry IDs、单调 generation 和 SHA-256 指纹。
原始事件继续保留；replay 按日志顺序应用 replacement 和其后的原文 tail。

## Consequences

文件 checkpoint 和相关恢复 API 被删除。`AgentLoopResult` 必须同时返回本轮新增
消息和完整 surface，运行时以后者更新下一轮历史。临时 request prefix 不进入
session surface。旧 compaction schema 不被解析。

## Validation

`test_session_surface.py` 覆盖类型化消息 round-trip、tool pairing、replacement
投影、指纹、branch prefix 和 restart；全量测试验证普通与压缩回合。

## Supersedes

None
