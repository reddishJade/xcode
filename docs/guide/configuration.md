# 配置系统与设置浏览器

Xcode 提供了多层级的配置发现栈以及交互式配置浏览器，方便在全局、项目、本地与会话级别灵活管理运行参数。

---

## 1. 配置分层覆盖栈

Xcode 按照以下优先级自底向上发现并合并配置（高优先级覆盖低优先级）：

```
1. 全局配置    ~/.xcode/settings.json        (最低优先级)
      │
      ▼
2. 项目配置    xcode.config.json
      │
      ▼
3. 本地覆盖    .xcode/settings.json
      │
      ▼
4. 环境变量    XCODE_APPROVAL_POLICY, API Keys 等 (最高优先级)
```

* **项目共享配置 (`xcode.config.json`)**：适合提交到 Git 仓库，供全团队共享相同的执行模式规则、Hooks 与 MCP 服务器配置；
* **本地私有覆盖 (`.xcode/settings.json`)**：默认被 `.gitignore` 忽略，适合开发者本地临时调整模型 Profile 或路径；
* **显式指定配置**：启动时使用 `--config /path/to/custom.json` 覆盖配置文件路径。

---

## 2. 交互式设置浏览器 (xcode config / /config)

Xcode 提供了专为运行时调整优化的交互式配置浏览器：

```bash
# 启动 CLI 配置浏览器
xcode config

# 或直接跳转至特定配置项
xcode config approval
```

在 TUI / REPL 会话中，输入 `/config` 即可打开相同的配置面板：

```
┌──────────────────────────────────────────────────────────┐
│ Xcode Settings Browser                                   │
├──────────────────────────────────────────────────────────┤
│ > Default Mode           build                           │
│   Approval Policy        agent decides                   │
│   Non-Workspace Access   on                              │
│   Shell                  auto                            │
├──────────────────────────────────────────────────────────┤
│ Description:                                             │
│ 'agent decides': Automatically review actions with the   │
│ independent reviewer profile.                            │
└──────────────────────────────────────────────────────────┘
```

* **上下键选择**：底部实时展示当前设置项的作用说明与安全权衡；
* **回车进入修改**：快速选择枚举值，当前生效值标注 `(current)`；
* **原子校验落盘**：修改后通过 `XcodeRuntimeConfig` 进行强类型校验，非法值拒绝写入。

---

## 3. 完整配置项速查

详尽的配置文件字段定义与默认值参考，请参阅 [CONFIG.md](../CONFIG.md)。

| 配置模块 | 主要控制字段 | 详细说明 |
|---|---|---|
| `provider` | `model_profiles`, `transport`, `thinking` | 大模型提供商、思考链与多 Profile 配置 |
| `execution_modes` | `default_mode`, `rules` | Plan / Build / Act 模式与自定义匹配规则 |
| `security` | `sandbox`, `approval_policy`, `restricted_dirs` | Bubblewrap 沙箱、审批策略与禁访路径 |
| `tools` | `shell`, `subagent_extra_tools` | 默认 Shell 类型与子代理额外工具开放 |
| `mcp` | `.xcode/mcp_config.json` | Model Context Protocol 服务器配置 |
| `hooks` | `entries` | 外部命令事件 Hook 列表 |
| `paths` | `sessions_dir`, `skills_dir` | 会话与技能扫描目录 |
| `observability` | `audit_path` | JSONL 结构化审计日志输出路径 |

---

← **上一篇**：[权限、安全与 Linux 沙箱 (security.md)](security.md) | **下一篇**：[MCP 服务与工具扩展 (mcp.md)](mcp.md) →

