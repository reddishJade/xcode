/* xcode workbench — 浏览器端逻辑。
   协议：WebSocket /ws；REST /api/info、/api/sessions。 */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);

  const els = {
    project: $("project-path"),
    branch: $("branch"),
    modelSelect: $("model-select"),
    sessionId: $("session-id"),
    modes: $("modes"),
    stream: $("stream"),
    empty: $("empty"),
    input: $("input"),
    send: $("send"),
    stop: $("stop"),
    scrollBottom: $("scroll-bottom"),
    sessionList: $("session-list"),
    newChat: $("new-chat"),
    workspaceSelect: $("workspace-select"),
    workspaceAdd: $("workspace-add"),
    workspaceModal: $("workspace-modal"),
    workspaceNewPath: $("workspace-new-path"),
    workspaceNewCancel: $("workspace-new-cancel"),
    workspaceNewConfirm: $("workspace-new-confirm"),
    modelModal: $("model-modal"),
    modelCustomInput: $("model-custom-input"),
    modelCustomCancel: $("model-custom-cancel"),
    modelCustomConfirm: $("model-custom-confirm"),
    toast: $("toast"),
    statsUsage: $("stats-usage"),
    statsContext: $("stats-context"),
    statsModel: $("stats-model"),
    approval: $("approval"),
    approvalTool: $("approval-tool"),
    approvalArgs: $("approval-args"),
    approvalReason: $("approval-reason"),
    approvalTranscript: $("approval-transcript"),
  };

  const MODES = ["plan", "build", "act"];
  let mode = localStorage.getItem("xcode.mode") || "build";
  if (!MODES.includes(mode)) mode = "build";

  let ws = null;
  let connected = false;
  let running = false;
  let reconnectTimer = null;
  let pendingApprovalId = null;
  let toastTimer = null;
  let viewedSessionId = null; // 当前视图展示的会话；null 表示实时空视图
  let currentWorkspace = ""; // 服务端当前工作区路径

  const steps = new Map(); // step -> {num, lamp, thinkingEl, thinkingText, assistantEl, toolsEl, buffer}
  let lastStepKey = 0;

  /* ── 工具函数 ── */

  function esc(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function markdown(text) {
    const codeBlocks = [];
    let body = String(text).replace(/```([\w+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      codeBlocks.push(code.replace(/^\n/, ""));
      return "\u0000CODE" + (codeBlocks.length - 1) + "\u0000";
    });
    body = esc(body);
    body = body.replace(/`([^`]+)`/g, (_, code) => "<code>" + code + "</code>");
    body = body.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    body = body.replace(/^\s*[-*]\s+/gm, "• ");
    body = body.replace(/^\s*(\d+)[.)]\s+/gm, (_m, n) => n + ". ");
    body = body.split("\n").reduce((out, line, i, all) => {
      const next = i + 1 < all.length ? all[i + 1] : "";
      const blank = line.trim() === "";
      const nextBlank = next.trim() === "";
      out.push(line);
      if (!blank && !nextBlank && i < all.length - 1) out.push("<br>");
      return out;
    }, []).join("");
    body = body.replace(/\u0000CODE(\d+)\u0000/g, (_, i) => {
      return '<pre><code class="code-block">' + esc(codeBlocks[+i]) + "</code></pre>";
    });
    return body;
  }

  function nearBottom() {
    const s = els.stream;
    return s.scrollHeight - s.scrollTop - s.clientHeight < 120;
  }

  function scrollToBottom(force) {
    if (force || nearBottom()) els.stream.scrollTop = els.stream.scrollHeight;
  }

  function toast(message, ms = 3200) {
    els.toast.textContent = message;
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (els.toast.hidden = true), ms);
  }

  function setMode(next) {
    mode = next;
    localStorage.setItem("xcode.mode", mode);
    els.modes.querySelectorAll(".mode").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.mode === mode);
    });

  }

  function setRunning(next) {
    running = next;
    els.send.disabled = next;
    els.stop.hidden = !next;
    els.scrollBottom.hidden = !next;
    if (next) els.empty?.remove();
    els.input.placeholder = next
      ? "运行中…（可输入，当前回合结束后处理；■ 停止立即中断）"
      : "描述任务…（Enter 发送，Shift+Enter 换行）";
  }

  /* ── 步骤/工具 DOM ── */

  function stepRail(lamp) {
    const rail = document.createElement("div");
    rail.className = "step__rail";
    const num = document.createElement("span");
    num.className = "step__num";
    rail.append(num, lamp);
    return { rail, num };
  }

  function ensureStep(stepNum) {
    if (steps.has(stepNum)) return steps.get(stepNum);
    const article = document.createElement("article");
    article.className = "step";

    const lamp = document.createElement("span");
    lamp.className = "step__lamp";
    lamp.dataset.lamp = "idle";
    const railParts = stepRail(lamp);
    article.appendChild(railParts.rail);

    const body = document.createElement("div");
    body.className = "step__body";

    const thinking = document.createElement("div");
    thinking.className = "thinking";
    thinking.hidden = true;
    const bar = document.createElement("button");
    bar.className = "thinking__bar";
    const thinkText = document.createElement("pre");
    thinkText.className = "thinking__text";
    thinking.append(bar, thinkText);

    const assistant = document.createElement("div");
    assistant.className = "assistant";

    const tools = document.createElement("div");
    tools.className = "tools";

    body.append(thinking, assistant, tools);
    article.appendChild(body);
    els.stream.appendChild(article);

    const record = {
      num: railParts.num,
      lamp: lamp,
      thinking: thinking,
      thinkingText: thinkText,
      thinkingBar: bar,
      thinkingCollapsed: false,
      thinkingTextHidden: false,
      assistant: assistant,
      assistantText: "",
      tools: tools,
      toolCards: new Map(),
      rendered: 0,
    };
    railParts.num.textContent = String(stepNum).padStart(2, "0");
    bar.addEventListener("click", () => toggleThinking(record));
    syncThinkingBar(record);
    steps.set(stepNum, record);
    lastStepKey = Math.max(lastStepKey, stepNum);
    scrollToBottom();
    return record;
  }

  function syncThinkingBar(record) {
    const label = record.thinkingCollapsed ? "▸ 展开" : "▾ 收起";
    record.thinkingBar.textContent = "◌ 思考 · " + label;
    record.thinkingText.hidden = record.thinkingTextHidden;
  }

  function toggleThinking(record) {
    record.thinkingCollapsed = !record.thinkingCollapsed;
    record.thinkingTextHidden = record.thinkingCollapsed;
    syncThinkingBar(record);
    scrollToBottom();
  }

  function addThinking(record, delta) {
    record.thinking.hidden = false;
    record.thinkingText.textContent += delta;
    lampState(record, "think");
    scrollToBottom();
  }

  function lampState(record, state) {
    record.lamp.dataset.lamp = state;
  }

  let flushQueued = false;
  function queueRenderAssist(record) {
    if (flushQueued) return;
    flushQueued = true;
    requestAnimationFrame(() => {
      flushQueued = false;
      renderAssist(record);
    });
  }

  function renderAssist(record) {
    const live = running && record === steps.get(lastStepKey);
    record.assistant.innerHTML =
      markdown(record.assistantText) + (live ? '<span class="cursor"></span>' : "");
    scrollToBottom();
  }

  function toolCard(record, callId, name, args) {
    const card = document.createElement("div");
    card.className = "tool-card";
    card.dataset.status = "running";

    const head = document.createElement("button");
    head.className = "tool-card__head";
    const status = document.createElement("span");
    status.className = "tool-card__status";
    status.textContent = "◌";
    const nm = document.createElement("span");
    nm.className = "tool-card__name";
    nm.textContent = name;
    const argsEl = document.createElement("span");
    argsEl.className = "tool-card__args";
    argsEl.textContent = JSON.stringify(args ?? {});
    const toggle = document.createElement("span");
    toggle.className = "tool-card__toggle";
    toggle.textContent = "▸ 输出";
    head.append(status, nm, argsEl, toggle);

    const output = document.createElement("div");
    output.className = "tool-card__output";
    output.hidden = true;
    const pre = document.createElement("pre");
    pre.textContent = "";
    output.appendChild(pre);

    const notice = document.createElement("div");
    notice.className = "tool-card__notice";
    notice.hidden = true;

    card.append(head, output, notice);
    record.tools.appendChild(card);

    head.addEventListener("click", () => {
      output.hidden = !output.hidden;
      toggle.textContent = output.hidden ? "▸ 输出" : "▾ 收起";
    });
    return { card, status, argsEl, pre, output, notice };
  }

  function systemNote(tag, text, cls = "") {
    const note = document.createElement("div");
    note.className = "system-note " + cls;
    note.innerHTML =
      '<span class="system-note__tag">' + esc(tag) + "</span><span>" + esc(text) + "</span>";
    els.stream.appendChild(note);
    scrollToBottom();
  }

  /* ── 事件处理 ── */

  function handleEvent(evt) {
    const stepNum = typeof evt.step === "number" ? evt.step : 0;
    const type = evt.type;
    const data = evt.data;

    if (type === "message_start") {
      const record = ensureStep(stepNum);
      if (data && Array.isArray(data.content)) {
        for (const block of data.content) {
          if (block && block.type === "text" && block.text) {
            record.assistantText += block.text;
            queueRenderAssist(record);
          } else if (block && (block.type === "tool_call" || (block.id && block.name))) {
            if (!record.toolCards.has(block.id)) {
              record.toolCards.set(block.id, toolCard(record, block.id, block.name, block.arguments));
            }
          }
        }
      }
      return;
    }

    if (type === "text_delta") {
      const record = ensureStep(stepNum);
      record.assistantText += data;
      queueRenderAssist(record);
      lampState(record, "think");
      return;
    }

    if (type === "reasoning_delta") {
      addThinking(ensureStep(stepNum), data);
      return;
    }

    if (type === "tool_use") {
      const record = ensureStep(stepNum);
      if (!record.toolCards.has(data.id)) {
        record.toolCards.set(data.id, toolCard(record, data.id, data.name, data.input));
      }
      lampState(record, "tool");
      scrollToBottom();
      return;
    }

    if (type === "tool_update") {
      const record = steps.get(stepNum);
      const card = record?.toolCards.get(data.tool_call_id);
      if (card && data.partial_result) {
        card.pre.textContent += data.partial_result;
        card.pre.classList.add("is-live");
        card.output.hidden = false;
        card.card.dataset.status = "running";
      }
      return;
    }

    if (type === "tool_result") {
      const record = steps.get(stepNum);
      const card = record?.toolCards.get(data.tool_use_id);
      if (card) {
        card.card.dataset.status = data.status === "error" ? "error" : "ok";
        card.status.textContent = data.status === "error" ? "✕" : "✓";
        card.pre.classList.remove("is-live");
        if (data.content) card.pre.textContent = data.content;
        card.output.hidden = data.content ? false : true;
        if (data.permission_notice) {
          card.notice.textContent = data.permission_notice;
          card.notice.hidden = false;
        }
      }
      return;
    }

    if (type === "assistant") {
      const record = ensureStep(stepNum);
      let text = "";
      for (const block of data) {
        if (block && typeof block.text === "string") text += block.text;
        else if (block && block.id && block.name) {
          if (!record.toolCards.has(block.id)) {
            record.toolCards.set(block.id, toolCard(record, block.id, block.name, block.input));
          }
        }
      }
      if (text) {
        record.assistantText = text;
        queueRenderAssist(record);
      }
      return;
    }

    if (type === "turn_end") {
      const record = steps.get(stepNum);
      if (record) lampState(record, "done");
      return;
    }

    if (type === "compaction") {
      systemNote(
        "压缩",
        `上下文已分层压缩：移除 ${data.messages_removed} 条消息，当前 ${data.messages_after} 条（触发：${data.trigger}）`
      );
      return;
    }

    if (type === "final") {
      const record = steps.get(stepNum);
      if (record) {
        lampState(record, "done");
        if (data.answer && record.assistantText.length < data.answer.length) {
          record.assistantText = data.answer;
          queueRenderAssist(record);
        }
      }
      renderFinalStrip(data);
      return;
    }
  }

  function renderFinalStrip(data) {
    const strip = document.createElement("div");
    strip.className = "final-strip";
    const chip = (label, value) =>
      "<span class='final-strip__chip'>" +
      esc(label) +
      " <b>" +
      esc(String(value ?? "—")) +
      "</b></span>";
    const metrics = data.metrics || {};
    const chips = [
      chip("steps", data.steps),
      chip("工具", (data.tool_calls || []).length),
      chip("结束", data.termination_reason || "completed"),
    ];
    if (metrics.llm_calls) chips.push(chip("llm", metrics.llm_calls));
    if (metrics.tool_time_ms) chips.push(chip("工具耗时", (metrics.tool_time_ms / 1000).toFixed(1) + "s"));
    strip.innerHTML = chips.join("");
    els.stream.appendChild(strip);
    scrollToBottom();
  }

  /* ── 审批 ── */

  function showApproval(payload) {
    pendingApprovalId = payload.id;
    els.approvalTool.textContent = payload.tool?.name || "未知工具";
    els.approvalArgs.textContent = JSON.stringify(payload.tool?.arguments ?? {}, null, 2);
    els.approvalReason.textContent = payload.reason || "";
    els.approvalTranscript.textContent = (payload.transcript || "").slice(0, 6000);
    els.approval.hidden = false;
  }

  function hideApproval() {
    els.approval.hidden = true;
    pendingApprovalId = null;
  }

  els.approval.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-approve]");
    if (!btn || !pendingApprovalId) return;
    const decision = btn.dataset.approve;
    send({
      type: "approval",
      id: pendingApprovalId,
      decision: decision === "deny" ? "deny" : "allow",
      scope: decision,
    });
    hideApproval();
  });

  /* ── WebSocket ── */

  function send(payload) {
    if (ws && connected) ws.send(JSON.stringify(payload));
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => {
      connected = true;
      refreshInfo();
      refreshModel();
      refreshSessions();
      refreshWorkspaces();
      refreshStats();
    };

    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (_err) {
        return;
      }
      handleMessage(msg);
    };

    ws.onclose = () => {
      connected = false;
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 1500);
    };

    ws.onerror = () => ws.close();
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case "hello":
        applyInfo(msg.info || {});
        setRunning(Boolean(msg.info?.busy));
        break;
      case "run_started":
        setRunning(true);
        break;
      case "run_idle":
        setRunning(false);
        refreshSessions();
        refreshStats();
        break;
      case "run_cancelled":
        systemNote("中断", "回合已取消");
        break;
      case "run_error":
        setRunning(false);
        systemNote("错误", msg.message || "未知错误", "system-note--error");
        break;
      case "user_message":
        renderUser(msg.text || "", "ws", msg.mode || mode);
        break;
      case "session_reset":
        resetView(true);
        refreshInfo();
        refreshSessions();
        refreshStats();
        break;
      case "session_switched":
        resetView(false);
        refreshInfo();
        refreshSessions();
        refreshStats();
        loadSession(msg.session_id || "", "live");
        break;
      case "workspace_switched":
        resetView(true);
        refreshInfo();
        refreshModel();
        refreshSessions();
        refreshWorkspaces();
        refreshStats();
        systemNote("工作区", "已切换到 " + (msg.info?.project || ""));
        break;
      case "approval_request":
        showApproval(msg);
        break;
      case "event":
        handleEvent(msg.event || {});
        break;
      case "pong":
      default:
        break;
    }
  }

  /* ── 提交 ── */

  async function submit() {
    const text = els.input.value.trim();
    if (!text || !connected || running) return;
    // 视图停留在历史会话时，先恢复该会话再提交（即“继续对话”）
    if (viewedSessionId) {
      const resumed = await tryResumeSession(viewedSessionId);
      if (!resumed) {
        toast("未能恢复该会话，已提交到当前实时会话");
      }
    }
    if (running) return; // 恢复期间可能被其他端的回合占用
    els.input.value = "";
    autoGrow();
    send({ type: "submit", text, mode });
  }

  els.send.addEventListener("click", submit);

  els.stop.addEventListener("click", () => {
    send({ type: "cancel" });
    els.stop.disabled = true;
    setTimeout(() => (els.stop.disabled = false), 800);
  });

  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });

  function autoGrow() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
  }
  els.input.addEventListener("input", autoGrow);

  async function switchModel(next) {
    const previous = els.modelSelect.dataset.current || "";
    if (!next || next === previous || next === "__custom__") return;
    try {
      const res = await fetch("/api/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: next }),
      });
      const data = await res.json();
      if (data.error) {
        toast(data.error);
        els.modelSelect.value = previous;
        return;
      }
      applyModel(data);
      toast("已切换模型 " + next);
      refreshSessions();
      refreshStats();
    } catch (_err) {
      toast("切换模型失败");
      els.modelSelect.value = previous;
    }
  }

  els.modelSelect.addEventListener("change", () => {
    const next = els.modelSelect.value;
    if (next === "__custom__") {
      els.modelModal.hidden = false;
      els.modelCustomInput.value = "";
      els.modelCustomInput.focus();
      return;
    }
    switchModel(next);
  });

  els.modelCustomCancel.addEventListener("click", () => {
    els.modelModal.hidden = true;
    els.modelSelect.value = els.modelSelect.dataset.current || "";
  });

  els.modelCustomConfirm.addEventListener("click", () => {
    const name = els.modelCustomInput.value.trim();
    els.modelModal.hidden = true;
    if (name) switchModel(name);
  });

  els.modelCustomInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") els.modelCustomConfirm.click();
  });

  els.modes.addEventListener("click", (e) => {
    const btn = e.target.closest(".mode");
    if (btn) setMode(btn.dataset.mode);
  });

  els.newChat.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/sessions", { method: "POST" });
      const data = await res.json();
      if (data.error) return toast(data.error);
      resetView(true);
      refreshInfo();
    } catch (_err) {
      toast("新建会话失败");
    }
  });

  els.scrollBottom.addEventListener("click", () => scrollToBottom(true));
  els.stream.addEventListener("scroll", () => {
    if (nearBottom()) els.scrollBottom.hidden = true;
  });

  async function refreshWorkspaces() {
    try {
      const res = await fetch("/api/workspaces");
      const data = await res.json();
      const current = data.current || "";
      currentWorkspace = current;
      els.workspaceSelect.innerHTML = "";
      for (const path of data.recent || []) {
        const opt = document.createElement("option");
        opt.value = path;
        const name = path.split(/[\\/]/).pop() || path;
        opt.textContent = path === current ? "● " + name : name;
        opt.title = path;
        if (path === current) opt.selected = true;
        els.workspaceSelect.appendChild(opt);
      }
    } catch (_err) {
      /* 忽略 */
    }
  }

  async function switchWorkspace(pathText) {
    const path = String(pathText || "").trim();
    if (!path) return toast("请输入工作区路径");
    try {
      const res = await fetch("/api/workspaces", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      const data = await res.json();
      if (data.error) {
        toast(data.error);
        return;
      }
      resetView(true);
      refreshInfo();
      refreshModel();
      refreshSessions();
      refreshStats();
      refreshWorkspaces();
      toast("已切换工作区 " + path);
    } catch (_err) {
      toast("切换工作区失败");
    }
  }

  els.workspaceSelect.addEventListener("change", () => {
    const next = els.workspaceSelect.value;
    if (!next || next === currentWorkspace) return;
    switchWorkspace(next);
  });

  els.workspaceAdd.addEventListener("click", () => {
    els.workspaceModal.hidden = false;
    els.workspaceNewPath.value = "";
    els.workspaceNewPath.focus();
  });

  els.workspaceNewCancel.addEventListener("click", () => {
    els.workspaceModal.hidden = true;
  });

  els.workspaceNewConfirm.addEventListener("click", () => {
    const path = els.workspaceNewPath.value.trim();
    els.workspaceModal.hidden = true;
    if (path) switchWorkspace(path);
  });

  els.workspaceNewPath.addEventListener("keydown", (e) => {
    if (e.key === "Enter") els.workspaceNewConfirm.click();
  });


  /* ── 会话列表 ── */

  async function refreshSessions() {
    try {
      const res = await fetch("/api/sessions");
      const data = await res.json();
      renderSessions(data);
    } catch (_err) {
      /* 服务暂不可用时静默 */
    }
  }

  function resetView(showEmpty) {
    steps.clear();
    lastStepKey = 0;
    els.stream.innerHTML = "";
    hideApproval();
    viewedSessionId = null;
    if (showEmpty) renderEmpty();
    els.input.focus();
  }

  function renderEmpty() {
    const wrap = document.createElement("div");
    wrap.className = "empty";
    wrap.id = "empty";
    wrap.innerHTML =
      '<p class="empty__eyebrow">xcode / web workbench</p>' +
      '<p class="empty__title">选择执行模式，向工作台提出第一个任务。</p>' +
      '<p class="empty__sub">事件流实时落盘为会话账本；刷新页面后可在左侧恢复最近会话。</p>';
    els.stream.appendChild(wrap);
    els.empty = wrap;
  }

  function renderSessions(data) {
    els.sessionList.innerHTML = "";
    const items = data.sessions || [];

    const li = document.createElement("li");
    const live = document.createElement("button");
    live.className = "session-item is-current";
    const liveTitle = document.createElement("span");
    liveTitle.className = "session-item__title";
    liveTitle.textContent = "● 实时会话";
    live.appendChild(liveTitle);
    live.addEventListener("click", () => {
      resetView(true);
      toast("已回到实时会话");
    });
    li.appendChild(live);
    els.sessionList.appendChild(li);

    if (!items.length) return;

    for (const item of items) {
      const li2 = document.createElement("li");
      const btn = document.createElement("button");
      btn.className =
        "session-item" + (item.id === data.current ? " is-current" : "");
      const title = document.createElement("span");
      title.className = "session-item__title";
      title.textContent = item.title || "未命名会话";
      btn.appendChild(title);
      btn.addEventListener("click", () => resumeSession(item.id));
      li2.appendChild(btn);
      els.sessionList.appendChild(li2);
    }
  }

  async function tryResumeSession(sessionId) {
    try {
      const res = await fetch("/api/sessions/resume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: sessionId }),
      });
      const data = await res.json();
      if (data.error) {
        toast(data.error);
        return false;
      }
      return true;
    } catch (_err) {
      return false;
    }
  }

  async function resumeSession(sessionId) {
    const ok = await tryResumeSession(sessionId);
    if (!ok) {
      resetView(true);
      return;
    }
    await loadSession(sessionId, "live");
    toast("已恢复会话，可以继续对话");
  }

  async function loadSession(sessionId, mode) {
    try {
      const res = await fetch("/api/sessions/" + sessionId);
      const data = await res.json();
      if (data.error) return toast(data.error);
      resetView(false);
      renderTranscript(data, { live: mode === "live" });
      if (mode !== "live") toast("已加载会话 " + sessionId);
    } catch (_err) {
      toast("加载会话失败");
    }
  }

  function renderTranscript(data, opts) {
    const live = Boolean(opts && opts.live);
    viewedSessionId = data.id;
    const banner = document.createElement("div");
    banner.className = "system-note";
    banner.innerHTML =
      '<span class="system-note__tag">' +
      (live ? "已恢复" : "历史") +
      "</span><span>会话 " +
      esc(data.id) +
      " · " +
      esc(data.title || "") +
      (live ? " · 可继续对话" : " · 只读回放，不影响实时会话") +
      "</span>";
    const backBtn = document.createElement("button");
    backBtn.className = "btn btn--ghost";
    backBtn.style.cssText = "margin:2px 0 12px 78px;";
    backBtn.textContent = live ? "清空视图" : "← 返回实时会话";
    backBtn.addEventListener("click", () => {
      resetView(true);
      toast(live ? "已清空视图" : "已回到实时会话");
    });
    els.stream.append(banner, backBtn);

    const entries = data.entries || [];
    for (const entry of entries) {
      const inner = entry.content || {};
      if (inner.type === "inbox/claimed") {
        renderUser(inner.data?.display_text || "", entry.id);
      } else if (inner.type === "assistant") {
        const blocks = inner.data;
        let text = "";
        if (typeof blocks === "string") text = blocks;
      else if (Array.isArray(blocks)) {
          text = blocks
            .map((b) => (b && (b.type === "text" ? b.text : null)) || (b && b.text) || "")
            .join("");
        }
        if (text) renderAssistantReadonly(text, entry.id);
      } else if (inner.type === "tool_use") {
        systemNote("工具", `${inner.data?.name || "tool"} — ${String(inner.data?.input ?? "")}`.slice(0, 300));
      } else if (inner.type === "tool_result") {
        systemNote("结果", `${inner.data?.status || "ok"} ${inner.data?.tool_use_id || ""}`.slice(0, 200));
      } else if (inner.type === "compaction") {
        systemNote("压缩", `生成 ${inner.data?.generation || "?"} · ${inner.data?.messages_after ?? "?"} 条消息`);
      }
    }
    scrollToBottom(true);
  }

  function renderUser(text, source, msgMode) {
    const wrap = document.createElement("div");
    wrap.className = "user-msg";
    wrap.dataset.entry = source;
    const chip =
      msgMode && ["plan", "build", "act"].includes(String(msgMode).toLowerCase())
        ? "<span class='user-msg__mode'>" + esc(String(msgMode).toLowerCase()) + "</span>"
        : "";
    wrap.innerHTML =
      '<span class="user-msg__mark">›</span><span class="user-msg__text">' +
      esc(text) +
      "</span>" +
      chip;
    els.stream.appendChild(wrap);
    scrollToBottom();
  }

  function renderAssistantReadonly(text, entryId) {
    const article = document.createElement("article");
    article.className = "step";
    article.dataset.entry = entryId;
    const lamp = document.createElement("span");
    lamp.className = "step__lamp";
    lamp.dataset.lamp = "done";
    const railParts = stepRail(lamp);
    article.appendChild(railParts.rail);
    const body = document.createElement("div");
    body.className = "step__body";
    const assistant = document.createElement("div");
    assistant.className = "assistant";
    assistant.innerHTML = markdown(text);
    body.appendChild(assistant);
    article.appendChild(body);
    els.stream.appendChild(article);
  }

  /* ── 信息 ── */

  async function refreshInfo() {
    try {
      const res = await fetch("/api/info");
      const info = await res.json();
      applyInfo(info);
    } catch (_err) {
      /* 忽略 */
    }
  }

  async function refreshStats() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();
      els.statsUsage.textContent = data.usage || "";
      els.statsContext.textContent = data.context || "";
      const model = data.model || "";
      const effort = data.effort ? " • " + data.effort : "";
      const provider = data.provider ? "(" + data.provider + ") " : "";
      els.statsModel.textContent = provider + model + effort;
    } catch (_err) {
      /* 忽略 */
    }
  }

  function applyInfo(info) {
    els.project.textContent = info.project || "—";
    els.branch.textContent = info.git_branch ? "⎇ " + info.git_branch : "";
    els.sessionId.textContent = "session " + (info.session_id || "");
  }

  async function refreshModel() {
    try {
      const res = await fetch("/api/model");
      const data = await res.json();
      applyModel(data);
    } catch (_err) {
      /* 忽略 */
    }
  }

  function applyModel(info) {
    const models =
      info.available && info.available.length ? info.available : [info.model || ""];
    const current = info.model || "";
    els.modelSelect.innerHTML = "";
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      if (m === current) opt.selected = true;
      els.modelSelect.appendChild(opt);
    }
    if (info.custom) {
      const custom = document.createElement("option");
      custom.value = "__custom__";
      custom.textContent = "＋ 自定义模型…";
      els.modelSelect.appendChild(custom);
    }
    els.modelSelect.dataset.current = current;
    els.modelSelect.title =
      current + (info.thinking === "on" ? " · thinking on" : "");
  }

  /* ── 启动 ── */

  setMode(mode);
  setRunning(false);
  connect();
})();
