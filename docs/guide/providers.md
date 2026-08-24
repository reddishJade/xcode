# 模型与 Provider 配置

Xcode 内置了统一的 Provider 适配层，支持主流 OpenAI 协议兼容的大模型服务商（包括 DeepSeek、Xiaomi MiMo、ChatGLM、OpenAI 等），并针对 Thinking（推理链）、Reasoning Effort、流式输出与 Prompt Cache 进行了深度优化。

---

## 1. 快速配置向导 (Setup Wizard)

首次使用 Xcode 时，推荐直接运行内置的交互式配置向导：

```bash
xcode setup
```

向导会引导你选择模型提供商、输入 API Key 并选择默认聊天模型，生成的配置会自动保存至项目根目录的 `xcode.config.json` 或全局 `~/.xcode/settings.json`。

---

## 2. 环境变量快速注入

无需修改配置文件，也可直接通过标准环境变量提供 API 密钥：

```bash
# DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# Xiaomi MiMo
export MIMO_API_KEY="mimo-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# ChatGLM (Zhipu)
export ZHIPUAI_API_KEY="..."
```

---

## 3. 配置文件深度定制

在 `xcode.config.json` 中，通过 `provider.model_profiles` 声明不同的模型角色。Xcode 支持 4 种 Profile：
* `main`：主 Agent 使用的模型；
* `subagent`：委派给子代理（Subagent）的模型；
* `reviewer`：Build 模式下自动审查工具操作的独立轻量模型；
* `fallback`：主模型出现不可用或限流时的备用模型。

### 示例配置

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "deepseek_chat",
        "chat_model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "thinking": true,
        "reasoning_effort": "high"
      },
      "reviewer": {
        "transport": "deepseek_chat",
        "chat_model": "deepseek-v4-flash",
        "thinking": false
      },
      "subagent": "deepseek-v4-flash"
    }
  }
}
```

---

## 4. 各厂商适配特性

### 4.1 DeepSeek
* **默认 Base URL**：`https://api.deepseek.com`
* **Thinking 模式**：原生支持，默认自动启用 `extra_body={"thinking": {"type": "enabled"}}`。
* **Reasoning Effort**：默认 `"high"`，支持动态调优（`off`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`）。
* **Prompt 缓存统计**：精准识别原生 `prompt_cache_hit_tokens` 与 `prompt_cache_miss_tokens`。

### 4.2 Xiaomi MiMo
* **默认 Base URL**：`https://api.xiaomimimo.com/v1`
* **模型推荐**：`mimo-v2.5-pro`（默认开启思考链）/ `mimo-v2-flash`（高性价比快速模型）。
* **缓存统计**：兼容读取 `prompt_tokens_details.cached_tokens`。

### 4.3 ChatGLM (Zhipu AI)
* **默认 Base URL**：`https://open.bigmodel.cn/api/paas/v4/`
* **工具流式**：`glm-4.6`/`glm-4.7` 支持工具调用实时流式解析。

---

## 5. 运行时动态切换模型

在 REPL / TUI 会话中，可使用 `/model` 命令无缝切换当前模型，无需重启：

```text
# 查看当前正在使用的模型与配置
/model

# 切换为指定 Provider 与模型
/model deepseek_chat/deepseek-v4-pro:max
/model openai_chat/gpt-5.4-mini:high
/model mimo_chat/mimo-v2.5-pro
```

---

← **上一篇**：[安装与环境准备 (install.md)](install.md) | **下一篇**：[快速上手与交互模式 (quickstart.md)](quickstart.md) →

