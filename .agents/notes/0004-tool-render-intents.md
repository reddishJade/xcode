# Tool render intents are protocol data
Status: Implemented
Date: 2026-08-18

## Context

终端宿主若依据工具名和输出文本猜测 diff、terminal 或文件位置，会把产品语义
复制到多个 UI，并在工具改名或输出调整时失效。

## Decision

工具在结果产生处附加严格的 `ToolRenderIntent`。intent 随 agent message、runtime
event、session codec 和 replay 传播，CLI/TUI 使用共享投影。当前类型为 terminal、
diff、location 和 subagent。

## Consequences

新增呈现类型是协议变更，必须覆盖完整传播链。原始文本继续供模型使用，intent
只表达用户交互语义。

## Validation

工具、event translator、session codec、共享终端投影均有聚焦测试，完整套件验证
现有 UI 行为未回退。

## Supersedes

None
