# TODO

## Context cost display

`/context` 命令已可以利用 registry 中 `Model.cost` 显示费用估算。

待后续扩展：
- 跟踪实际 output token 数（从 `AssistantMessage.usage` 提取）使 cost 更准确
- OpenAI long-context 分段计价（>272K input 触发 2x/1.5x）
