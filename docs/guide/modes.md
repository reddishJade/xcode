# 执行模式：Plan、Build、Act

执行模式同时控制工具可见性、默认规则、Shell 未决效果处理和审批路由。模式状态进入 session run state，恢复会话时继续使用记录中的模式。

## 1. 模式对照

| 模式 | 工具可见性 | 默认动作 | 适用场景 |
| --- | --- | --- | --- |
| Plan | 只读探索、搜索、Web、question；显式技能可激活 | 规则覆盖外的动作 deny | 研究架构、制定方案 |
| Build | 全部已注册工具 | 项目结构化写入 allow；Shell 与未匹配动作 ask | 自动完成编码和验证 |
| Act | 全部已注册工具 | 只读 allow；写入与 Shell ask | 逐项确认副作用 |

## 2. Plan

```text
/plan
/plan 分析当前 provider 结构并给出修改方案
```

Plan 可见 `read_file`、`glob_files`、`find_files`、`list_dir`、`grep_search`、`search_tools`、`webfetch`、`websearch` 和 `question`。`write_file`、`edit_file` 的默认允许目标是 `.xcode/plans/*.md`，`apply_patch`、bash 和其他写操作由模式 fallback 拒绝。

Plan investigation turn 默认上限为 8。达到上限后自动切换 Build，并向下一轮注入模式通知。

## 3. Build

```text
/build
```

Build 保持全部工具可见：

- 项目内 `write_file`、`edit_file`、`apply_patch` 由默认规则直接允许。
- 读取、搜索、技能、记忆和 MCP 工具由默认规则允许。
- Shell 默认进入自动审批 reviewer。
- 危险命令、敏感路径、restricted_dirs、项目外未授权路径仍然形成硬拒绝。

自动 reviewer 只授予当前动作的 once 权限。`security.approval_policy=never` 时，ask 约束转为确定性 deny。

## 4. Act

```text
/act
```

Act 默认允许只读工具，写工具和 Shell 请求用户审批。用户可以在授权面板选择：

- `Allow (once)`：当前动作。
- `Allow this session`：当前 session 的同类目标。
- `Always allow`：写入项目级永久授权。
- `Deny`：拒绝当前动作，并可向模型提供下一步建议。

规则、restricted_dirs 和危险命令优先于交互选择。

## 5. 模式切换

REPL 和 TUI 使用 `/plan`、`/build`、`/act`；Web 顶部模式按钮在下一次提交时携带模式。

当前 run 会捕获 ToolGate snapshot。模式切换后的规则在新的模型/工具边界生效，正在执行的单次工具调用保持原决策。

## 6. 自定义 ruleset

```json
{
  "execution_modes": {
    "default_mode": "build",
    "build": {
      "rules": [
        {
          "action": "bash",
          "command": "git",
          "subcommand": "push",
          "effect": "ask"
        },
        {
          "action": "write_file",
          "resource_pattern": "deploy/**",
          "effect": "deny"
        }
      ]
    }
  }
}
```

规则字段：

- `action`：工具名或通配符。
- `effect`：`allow`、`ask`、`deny`。
- Shell 条件：`command`、`subcommand`、`subcommand_in`、`flags_any`、`flags_all`。
- 文件或资源条件：`resource_pattern`。

规则按顺序匹配，后匹配规则覆盖同层前匹配规则。用户 ruleset 追加在默认 ruleset 后，因此可以收紧默认动作。
