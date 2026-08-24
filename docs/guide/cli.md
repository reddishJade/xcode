# CLI 命令行参数与子命令速查

Xcode 提供了简洁清晰的命令行调用入口。

---

## 1. 基础启动语法

```bash
xcode [SUBCOMMAND] [OPTIONS]
```

---

## 2. 核心子命令速查

### `xcode`（默认子命令：`tui`）
启动全屏终端交互界面（TUI）：
```bash
xcode
xcode tui
```

### `xcode cli`
启动标准多轮对话 REPL 模式：
```bash
xcode cli
```

### `xcode setup`
启动交互式配置向导，配置模型 Provider 与 API Key：
```bash
xcode setup
```

### `xcode config`
启动交互式配置浏览器，管理运行参数（执行模式、审批策略、Shell 等）：
```bash
xcode config
xcode config approval
```

---

## 3. 常用全局选项

| 选项 | 缩写 | 参数类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `--prompt` | `-p` | string | `None` | 单轮提问模式：执行单次提问并打印结果后立即退出 |
| `--resume` | `-r` | flag | `False` | 自动恢复当前项目最近一次历史会话 |
| `--config` | `-c` | path | `None` | 显式指定 `xcode.config.json` 或私有配置文件的路径 |
| `--project-root` | — | path | 当前工作目录 | 指定 Agent 执行的工作区根目录 |
| `--version` | `-v` | flag | — | 查看 Xcode 当前版本号 |
| `--help` | `-h` | flag | — | 查看完整的命令行帮助信息 |

---

## 4. 典型调用场景示例

```bash
# 场景 1：CI / 脚本化快速代码分析
xcode -p "检查当前 Git 暂存区的修改是否有明显语法问题"

# 场景 2：基于特定配置文件启动全屏 TUI
xcode --config ./configs/prod_settings.json

# 场景 3：指定特定项目目录并恢复历史会话
xcode --project-root /home/user/backend --resume
```

---

← **上一篇**：[Slash 命令完全参考手册 (slash-commands.md)](slash-commands.md) | **下一篇**：[实战场景与使用示例指南 (../examples.md)](../examples.md) →

