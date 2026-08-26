# 模型与 Provider

Xcode 把模型调用统一为 `ModelProvider` 流式协议。provider 负责服务差异，Agent 只接收统一的文本、推理、工具、usage、结束和失败事件。

## 1. 支持的 transport

| transport | 适配器 | 说明 |
| --- | --- | --- |
| `openai_chat` | `OpenAIChatProvider` | OpenAI Chat Completions 及兼容网关 |
| `deepseek_chat` | `DeepSeekProvider` | DeepSeek Chat 与 reasoning content |
| `chatglm_chat` | `ChatGLMProvider` | ChatGLM Chat、thinking、tool stream |
| `mimo_chat` | `MiMoProvider` | Xiaomi MiMo Chat |
| `custom` | OpenAI Chat 适配器 | 自定义 base URL 的 OpenAI-compatible 网关 |

共享基类处理消息转换、工具 schema、thinking 参数、流式 chunk、usage 和在途请求中止；各 provider 处理自己的字段约束。

## 2. Profile 配置

运行时 profile 通常使用 `main`、`subagent`、`fallback` 和 `reviewer`：

- `main`：主 Agent。
- `subagent`：子代理。
- `fallback`：主 provider 连续失败后的回退 provider。
- `reviewer`：Build 自动审批 reviewer。

示例：

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "deepseek_chat",
        "chat_model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "thinking": true,
        "reasoning_effort": "high"
      },
      "reviewer": {
        "chat_model": "deepseek-v4-flash",
        "thinking": false
      },
      "subagent": "deepseek-v4-flash"
    }
  }
}
```

profile 可以从 `main` 继承。API key 可以写在 profile，也可以通过环境变量提供。常用变量：

```bash
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
MIMO_API_KEY=...
CHATGLM_API_KEY=...
ZHIPUAI_API_KEY=...
BIGMODEL_API_KEY=...
```

配置值优先于环境变量。provider 专用变量优先于通用 `OPENAI_API_KEY` 和 `API_KEY`。

## 3. 内置模型

内置模型注册表当前包含：

| provider | 模型 |
| --- | --- |
| OpenAI | `gpt-5.5`、`gpt-5.4`、`gpt-5.4-mini` |
| DeepSeek | `deepseek-v4-pro`、`deepseek-v4-flash` |
| ChatGLM | `glm-5.1`、`glm-5`、`glm-5-turbo`、`glm-4.7`、`glm-4.7-flash` |
| MiMo | `mimo-v2.5-pro`、`mimo-v2.5` |

网关可以提供自定义模型名。Web 工作台会优先请求当前 gateway 的 `/models`，失败时使用注册表和当前模型；custom transport 展示当前模型与自定义输入。

## 4. Thinking 与 reasoning effort

`thinking` 控制推理能力开关，`reasoning_effort` 控制 provider 支持的推理档位。两者分别保存：

- `/thinking off` 关闭 thinking，并清除本次模型切换中的 effort 覆盖。
- `/thinking on` 开启 thinking，并沿用 profile 当前 effort。
- `/effort off` 关闭 reasoning effort。
- `/effort LEVEL` 开启 thinking 并设置 effort。

当前界面档位：

- `openai_chat`、`custom`：`none`、`minimal`、`low`、`medium`、`high`、`xhigh`。
- `deepseek_chat`：`off`、`high`、`max`。
- ChatGLM 与 MiMo 主要使用 thinking 配置以及各自 extra body。

provider 可能返回 `reasoning_content`。跨 provider 使用历史消息时，归一化器会把 provider 专用字段转换为通用文本结构。

## 5. 模型切换

```text
/model
/model gpt-5.4-mini
/model main/gpt-5.4-mini:high
/model subagent/deepseek-v4-pro
/effort high
/thinking off
```

斜杠前的名称在当前 CLI 中作为 profile 使用，可用 profile 为 `main` 和 `subagent`。活动 run 存在时拒绝主模型替换；run 结束后再次执行切换。切换子代理模型只影响后续 child activation。

## 6. 缓存、成本与回退

provider 记录输入 token、输出 token、缓存命中/未命中 token、reasoning token 和累计成本。DeepSeek 优先读取原生缓存字段，OpenAI-compatible usage 使用 `prompt_tokens_details.cached_tokens` 回退字段。

模型注册表保存美元/百万 token 单价和可选 UTC 高峰时段。状态栏可以展示输入、输出、缓存读取、缓存写入、命中率和累计成本。

ProviderRuntime 对 429、500、502、503、529、连接错误和超时使用退避重试，并把状态码归类为可读错误。Fallback wrapper 在主 provider 连续三次失败后切换 fallback；fallback 连续三次成功后重新尝试主 provider。
