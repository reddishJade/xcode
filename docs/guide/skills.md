# Skills 技能系统

Xcode 支持轻量级、模块化的 **Skills 技能系统**。Skill 允许开发者为 Agent 封装特定领域的操作规程、代码生成模板与最佳实践。

---

## 1. Skill 目录结构

每个 Skill 是一个独立的文件夹，包含一个 `SKILL.md` 核心定义文件以及可选的脚本、资源目录：

```text
my-custom-skill/
├── SKILL.md          # 包含 YAML Frontmatter 元数据与指令正文
├── scripts/          # 可选辅助脚本
└── references/       # 可选参考手册与代码片段
```

### `SKILL.md` 文件格式示例

```markdown
---
name: refactor-helper
description: 专用于 Python 代码重构的辅助技能，遵循纯函数优先与类型注解规范
---

# 重构指引

当用户要求重构代码时，必须遵循以下步骤：
1. 先运行测试建立 Baseline；
2. 提取公共逻辑，减少重复代码；
3. 为所有公共函数补充类型注解；
4. 运行 `pytest` 确认行为一致。
```

---

## 2. Skill 发现与覆盖顺序

Xcode 按照以下优先级扫描并发现可用 Skill（先发现优先，同名覆盖）：

1. 显式路径：`paths.skills_dir`；
2. 项目私有目录：`.xcode/skills/`；
3. 项目 Agent 目录：`.agents/skills/`；
4. 用户全局目录：`~/.xcode/skills/`；
5. 用户全局 Agent 目录：`~/.agents/skills/`。

> 默认情况下，Xcode 仅加载用户级受信任技能。若需加载项目内的技能，可在配置中启用 `"skills": {"trust_project_skills": true}`。

---

## 3. 渐进式披露与按需加载

为了避免向模型 Context 一次性灌入过多技能正文导致 Token 浪费，Xcode 采用**渐进式披露机制**：

1. **初始注册**：系统启动时，仅向模型提供已发现 Skill 的简短名称与 `description` 摘要；
2. **按需激活**：
   * **自动加载**：当模型判断任务需要某项技能时，调用内置的 `load_skill` 工具读取技能全文及脚本清单；
   * **显式调用**：用户可在 REPL/TUI 中通过行首 `$` 符号或 `/skill` 显式激活：
     ```text
     $refactor-helper 优化 src/xcode/ai/ 目录下的类继承体系
     /skill refactor-helper
     ```
3. **压缩持久化**：已激活的 Skill 状态会在上下文压缩（Compaction）时被标记保留，不会因滚动 Checkpoint 丢失。

---

← **上一篇**：[MCP 服务与工具扩展 (mcp.md)](mcp.md) | **下一篇**：[Subagents 子代理架构 (subagents.md)](subagents.md) →

