# Xcode 已知问题与待办

基准复核日期：2026-06-18。每项“现状”均已按当前源码重新核对；已修复或已不
成立的描述不保留在本文件中。

优先级定义：

- P0：安全边界或可能导致错误授权。
- P1：核心能力不可用、状态不一致或协议行为错误。
- P2：资源稳定性、正式能力完善和可观测性。
- P3：维护性、质量治理和低风险一致性。

同一优先级内按依赖顺序排列。Skill 和 MCP 是核心能力；Memory 是正式但可选的
能力。现有 Python Plugin 系统不作为产品能力保留。

## 明确不进入近期范围

- 通过 JSON/YAML 配置直接定义任意可执行 Tool；外部工具扩展统一使用 MCP。
- 模糊匹配后直接执行 slash command；模糊能力只用于候选提示，执行必须精确。
- 配置文件加载进程内 Python Hook callback。
- Skill marketplace。
- Harness 侧向量语义匹配。
- `enabled.txt` 专用激活格式。
- Skill 独立脚本执行 runtime。
- MCP Streamable HTTP、OAuth、resources、prompts、sampling、roots 和
  elicitation；出现明确 server 用例后再进入 TODO。
- MCP 旧 HTTP+SSE 和私有 WebSocket transport。
- 通用工具资源标签、依赖图和读写锁调度器。
- Subagent 递归和 child-to-child 通信。
- 完整全系统错误 taxonomy。
- 仅为架构完整性引入 OpenTelemetry。
