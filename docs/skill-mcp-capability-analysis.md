# Skill 与 MCP 核心能力分析

审查日期：2026-06-18。

本报告用于确定 Xcode 对 Agent Skills 和 MCP 的基础兼容范围。它不要求全量
实现两个生态的所有可选能力，也不直接生成 `TODO.md` 条目。

参考标准：

- [Agent Skills specification](https://agentskills.io/specification)。
- [Agent Skills client implementation guide](https://agentskills.io/client-implementation/adding-skills-support)。
- [MCP specification 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18)。
- [MCP lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)。
- [MCP transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)。
- [MCP tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)。

## 结论

### Skill

Xcode 已完成 discovery、catalog 和 dedicated activation tool 的主体结构，但
还不能称为完整的 Agent Skills activation 闭环。

最值得实现的不是 harness 关键词 `auto_trigger` 或向量语义匹配，而是：

1. 在 skill catalog 旁明确告诉模型：任务匹配 description 时必须先调用
   `load_skill`。
2. 让 activation tool 的 schema 只接受真实 skill name。
3. 返回 skill directory 和完整资源目录元数据，使相对路径可解析。
4. 跟踪 session 内 activation，避免重复注入，并在 compaction 中保护已激活
   skill 内容。
5. 增加用户显式激活入口。

Agent Skills 官方实现指南明确说明，多数客户端的 model-driven activation
依赖模型读取 name/description 后自行判断，不要求 harness 进行关键词或向量
匹配。因此“auto trigger 值得做”，但应定义为完善 model-driven activation
闭环，而不是先实现另一套匹配引擎。

### MCP

Xcode 已是可工作的 stdio tools client，但协议实现仍偏向单一 smoke-test
路径。对于本地 coding agent，stdio + tools 足以作为正式基础能力；不需要先
实现 resources、prompts、OAuth 和 Streamable HTTP。

正式化前最值得补齐：

1. 协议版本协商和 server capability 校验。
2. `tools/list` pagination。
3. 区分 response、notification 和 server request；至少正确处理 ping。
4. request timeout 后发送 cancellation notification。
5. 更符合规范的 graceful shutdown。
6. 正确处理 `structuredContent`，并明确非文本 content 的宿主映射策略。
7. 为 persistent client 增加状态、重连和可诊断错误。

## 产品分类建议

| 能力 | 目标分类 | 默认启用建议 |
| --- | --- | --- |
| Skill | Core | 是 |
| MCP | Core | 是；无配置时为空 |
| Memory | Formal optional | 否，先保持显式配置 |
| Plugins | Removed | 不适用 |

MCP 和 Memory 应从 `experimental/` 迁出。该迁移表示维护承诺和模块边界变化，
不表示所有可选协议功能都必须实现。原有进程内 Python Plugins 已删除，不作为
正式扩展机制保留。

---

## Skill 现状核对

### 原矩阵修正

| Feature | 修正后状态 | 说明 |
| --- | --- | --- |
| Discovery paths | 已实现，当前是 4 级 | project `.xcode`、project `.agents`、user `.xcode`、user `.agents`；project 路径仅在 trusted workspace 配置启用时加入 |
| `SKILL.md` frontmatter | 部分实现 | 读取 name、description、hidden；未按规范校验 name 格式、长度、目录名一致性，也未保留 license、compatibility、metadata、allowed-tools |
| Skill catalog | 已实现 | `SkillIndexCollector` 注入 name 和 description |
| Dedicated activation | 已实现 | `load_skill` 返回正文及 references 元数据 |
| Reference loading | 已实现 | 可显式加载单个 `references/` 文件 |
| XML/binary/truncation safety | 已实现 | activation output 与 catalog 均执行 XML 转义；非法控制字符会替换，catalog description 限制为 1024 字符 |
| Conformance tests | 已实现 | 覆盖 discovery、references、binary、symlink、truncation 和 scripts 不自动执行 |
| `scripts/` execution | 未实现，但不是基础兼容缺陷 | 标准允许脚本存在，不要求客户端提供专用脚本执行器 |
| `assets/` discovery | 未实现 | activation 结果没有列出 assets |
| Workspace trust config | 已实现 | `SkillsRuntimeConfig.trust_project_skills` 默认 false，未信任项目不披露项目级 skill |
| Model-driven activation | 部分实现 | 模型看得到 catalog 和 `load_skill`，但 catalog 缺少明确 activation behavioral instruction |
| Harness semantic matching | 未实现，且非必需 | 官方指南倾向模型自行判断 |
| Marketplace | 未实现，且非规范基础要求 | 属于分发产品，不是 runtime compatibility |
| `enabled.txt` | 未实现，且非 Agent Skills 规范要求 | 可由 Xcode 自定义配置替代 |

### 已经符合的基础范式

- 用户级 discovery，以及显式信任后的项目级 discovery。
- `.agents/skills/` 跨客户端目录兼容。
- `SKILL.md` YAML frontmatter + Markdown body。
- name/description catalog 的 progressive disclosure tier 1。
- dedicated `load_skill` activation 的 tier 2。
- references 按需加载的 tier 3。
- 确定性名称冲突规则。
- malformed skill 跳过和日志。
- reference 文件的路径、symlink、binary 和大小治理。
- catalog XML 转义、非法控制字符替换和 description 长度限制。
- 项目级 skill 的显式 trusted-workspace 边界。

### 值得优先实现

#### S1 · Model-driven activation 闭环

在 `<available-skills>` 前后增加简短且稳定的行为指令：

```text
当用户任务明确匹配某个 skill 的 description 时，在执行任务前调用
load_skill(name) 加载完整指令。
```

这就是符合 Agent Skills 范式的 auto trigger。无需先做 harness 关键词匹配。

同时：

- `load_skill.name` schema 使用 discovered names enum。
- 没有 skill 时不注册 `load_skill`，也不注入空 catalog。
- activation 记录可用于 audit 和 eval。

#### S1 · Skill root 与资源清单

activation output 应包含 skill root，并列出 `scripts/`、`references/` 和
`assets/` 的相对路径元数据，但不主动读取或执行。

理由：

- skill 正文中的相对路径目前无法从 `load_skill` 输出可靠解析。
- 现有 `load_skill(reference=...)` 只能访问 references，无法让模型发现
  templates、assets 或 scripts。
- 无需专用 script executor；脚本可以通过现有 read/bash 工具并经过普通权限
  系统执行。

#### S1 · Activation 生命周期

- session 内记录已激活 skill。
- 重复 activation 返回简短状态或复用已有 context，避免重复正文。
- compaction 识别 skill wrapper，确保激活指令不会被静默裁剪。
- session resume 后恢复 activated skill 状态。

#### S2 · 用户显式激活

支持 `$skill-name`、`/skill skill-name` 或等价语法，并提供补全。它不是模型
auto trigger 的替代，而是用户可控入口。

#### S2 · 规范字段和诊断

- 校验 name 字符集、长度、连续 hyphen 和目录名一致性。
- 保留 compatibility、license 和 metadata。
- compatibility 应在 activation 时提供给模型。
- 对 cosmetic 问题 warn + load，对缺失 description 或无法解析 YAML 才 skip。

### 暂不值得实现

- Harness 侧向量语义匹配。
- Skill marketplace。
- `enabled.txt` 专用格式。
- 独立的脚本执行 runtime。
- 默认预批准 `allowed-tools`；该字段本身仍是实验性，并且不能绕过 Xcode
  PermissionEngine。

---

## MCP 现状核对

### 原矩阵修正

| Feature | 修正后状态 | 说明 |
| --- | --- | --- |
| Canonical config | 已实现 | `.local/mcp_config.json` |
| stdio transport | 已实现 | 子进程 + newline-delimited JSON |
| Content-Length | 兼容读取 | 当前规范 stdio 使用 newline delimiter；Content-Length 只是旧格式兼容 |
| initialize + initialized | 部分实现 | 有握手，但固定发送 2024-11-05，未验证响应版本和 capabilities |
| `tools/list` | 部分实现 | 不处理 pagination cursor |
| `tools/call` | 已实现 | 支持 text 和 `isError` |
| Tool naming | 已实现 | `mcp__<server>__<tool>` 和 collision detection |
| Permission integration | 已实现 | MCP action、HITL、grant scope 限制 |
| Deferred loading | 已实现 | schema cache、bootstrap 和 search tool |
| Subagent exclusion | 已实现 | child registry 排除 MCP |
| Real stdio conformance test | 已实现 | 覆盖握手、list、call、error 和 shutdown |
| Streamable HTTP | 未实现 | 对本地基础能力不是阻塞项 |
| OAuth | 未实现 | 仅在 remote transport 进入范围后需要 |
| Resources/prompts | 未实现 | MCP server 的可选 feature，不是 tools client 的最低门槛 |
| List changed | 未实现 | persistent tool catalog 场景值得支持 |
| Ping/server requests | 未实现 | 基础生命周期健壮性缺口 |
| Cancellation/progress | 未实现 | timeout cancellation 值得先做；progress 可后置 |
| Non-text results | 未实现 | 当前只返回 placeholder |
| Structured content/output schema | 未实现 | 对现代 tools server 有实际兼容价值 |

### 基础正式能力的建议边界

Xcode MCP v1 不需要覆盖整个协议。正式支持范围可以明确为：

```text
Transport:
  stdio

Lifecycle:
  initialize/version negotiation/capability negotiation/initialized/shutdown

Server feature:
  tools/list with pagination
  tools/call
  tools list_changed when advertised

Utilities:
  ping
  request timeout + cancellation
  structured errors and logging

Content:
  text
  structuredContent
  explicit unsupported handling for image/audio/resource content
```

不在 v1 承诺范围：

- Streamable HTTP。
- OAuth。
- resources。
- prompts。
- sampling、roots 和 elicitation。
- resource subscription。
- audio/image 原生渲染。

这些能力是 MCP 的合法组成部分，但不是 Xcode 作为本地 tool client 的基础正式
支持门槛。

### 值得优先实现

#### M0 · 正确的版本和 capability negotiation

- 发送客户端实际支持的最新版本。
- 检查 server 返回版本；不支持时断开并报告。
- 保存 server capabilities，只调用已协商 feature。
- 保存 serverInfo 和 instructions 供诊断。

当前固定版本且忽略响应内容，容易形成“握手成功但实际协议不兼容”的假成功。

#### M0 · JSON-RPC 双向消息分类

当前 read loop 将任何带 `id` 的消息放入 pending response。需要区分：

- response：有 `result` 或 `error`。
- server request：有 `method` 和 `id`。
- notification：有 `method`、无 `id`。

至少处理 ping，并对未支持 server request 返回标准 method-not-found，而不是把它
误认为 client request 的响应。

#### M1 · Tool discovery 完整性

- 实现 `tools/list` cursor pagination。
- 支持 `notifications/tools/list_changed` 后刷新 cache/registry。
- cache 记录 protocol version 和 server identity，避免只按 command/env hash
  复用过期 schema。

#### M1 · Timeout、取消和关闭

- request timeout 后发送 `notifications/cancelled`。
- graceful shutdown：先关闭 stdin，等待进程，随后 TERM/KILL。
- persistent client 失败后提供有限重连和明确状态。
- 对所有异常保留脱敏后的 server、method、request id 和状态信息。

#### M1 · 现代 tool result

- 支持 `structuredContent`。
- 保留 `outputSchema` 并在可行时验证。
- 对 image/audio/resource link/embedded resource 返回结构化宿主内容或明确、
  可诊断的 unsupported result，不能只保留第一个 placeholder。

#### M2 · 正式模块和配置管理

- 从 `experimental/` 迁出。
- `mcp` 加入核心 group；无配置时为空。
- 提供最小状态命令：server、transport、protocol version、status、tool count、
  last error。
- 保留 MCP 工具的 HITL 和 exact-target grant 约束。

### 有需求后实现

- Streamable HTTP 和 OAuth：需要真实 remote server 用例后一起实现。
- resources：当 Xcode 需要从 MCP 拉取只读上下文时实现，可映射为
  ContextBlock。
- prompts：与 Xcode system prompt 和 skill 的边界需要先设计，优先级低于
  tools/resources。
- progress notification：长调用 UX 需要时实现。
- roots/sampling/elicitation：只有明确 server 依赖时实现。

### 暂不值得实现

- 已被 Streamable HTTP 取代的旧 HTTP+SSE 作为新主路径。
- WebSocket 私有 transport。
- 在没有 remote transport 的情况下单独实现 OAuth。
- 为追求“全协议覆盖”一次性实现所有 client/server features。

---

## 建议验收标准

### Skill

- 官方最小示例 skill 可发现、显示、由模型主动加载并执行其指令。
- 用户显式激活同一 skill 可用。
- skill 相对 references/scripts/assets 路径可解析。
- 重复 activation 不重复注入全文。
- compaction 和 session resume 后 skill 指令仍有效。
- 不可信 description 不能破坏 catalog 结构。

### MCP

- 官方 stdio tools server 示例可连接。
- 协议版本不兼容时明确失败。
- 多页 `tools/list` 完整发现。
- tool list changed 后 catalog 可刷新。
- ping、timeout cancellation 和 graceful shutdown 可验证。
- text、structuredContent、error result 行为明确。
- 未支持 feature 会报告 capability limitation，而不是静默失败。

## 不进入 TODO 的原因

Skill/MCP 的上述项目包含产品范围选择和正式支持承诺。应先审查本报告，确认
v1 支持边界，再将获批项目拆为独立 TODO 和实现提交。
