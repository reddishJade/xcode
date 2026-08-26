# Skills 技能系统

Skill 是由 `SKILL.md` 描述的可加载工作规程。Xcode 先建立技能目录索引，再按任务需要加载正文和引用资源。

## 1. 目录结构

```text
skill-name/
├── SKILL.md
├── references/
├── scripts/
└── assets/
```

`SKILL.md` 以 YAML frontmatter 开始：

```markdown
---
name: code-review
description: Review focused code changes and report concrete findings.
compatibility: Python 3.12+
allowed-tools: read_file, grep_search
---

# Review procedure

Read the relevant diff, inspect surrounding code, and report evidence.
```

必须提供非空 `description`。`name`、目录名、`disable-model-invocation`、`license`、`compatibility`、`allowed-tools` 和字符串 metadata 会被规范化。

## 2. 发现顺序

搜索目录按优先级排列：

1. 显式 `paths.skills_dir`。
2. 项目 `.xcode/skills/`，需要 `trust_project_skills=true`。
3. 项目 `.agents/skills/`，需要 `trust_project_skills=true`。
4. 用户 `~/.xcode/skills/`。
5. 用户 `~/.agents/skills/`。

同名技能 first-wins。`disable-model-invocation: true` 的技能可以被索引记录，但不会出现在可激活目录。

## 3. 渐进加载

启动时 `SkillIndexCollector` 只向模型提供名称和 description：

```xml
<available-skills>
  <skill>
    <name>code-review</name>
    <description>Review focused code changes.</description>
  </skill>
</available-skills>
```

任务明确匹配时，模型调用：

```json
{"name": "code-review"}
```

`load_skill` 随后返回完整正文、兼容性、advisory allowed-tools、references、scripts 和 assets 元数据。技能正文仅在激活时进入上下文。

## 4. 显式激活

REPL 和 TUI 支持：

```text
$code-review 检查当前修改
/skill code-review
```

显式激活会使用同一个 `load_skill` 工具执行路径，生成 tool use/tool result 语义事件并写入 session。重复激活返回 `already-active` 状态。

## 5. References 与资源

`references/` 在发现阶段扫描元数据，隐藏文件、符号链接、二进制文件和不可读取文件标记为 skipped。单个 reference 读取上限为 50 KB；模型可以使用：

```json
{"name": "code-review", "reference": "checklist.md"}
```

`scripts/` 和 `assets/` 记录相对路径与大小，正文加载过程只披露资源元数据。

## 6. 会话、压缩与安全

技能激活状态通过 `<skill-activation-state>` 标记写入工具结果。session restore 从标记恢复激活集合；compaction 保护激活 tool use 与完整 result 的配对。

技能的 `allowed-tools` 属于 advisory 信息，权限 gate 继续执行。技能内容进入模型上下文后，仍遵循工具 schema、执行模式和路径边界。
