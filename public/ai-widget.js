// public/ai-widget.js
// Usage on your page:
// <script src="/ai-widget.js"
//         data-api="http://127.0.0.1:8000/ask"
//         data-suggestions='["What does the cover letter say?","List my key skills","Summarize my experience","Any gaps to fix?"]'
//         defer></script>

(function () {
  const script = document.currentScript;
  const API = script?.dataset?.api || "http://127.0.0.1:8000/ask";

  // ---- suggestions config (from data-attributes or defaults) ----
  function getSuggestions() {
    // Try JSON first
    const raw = script?.dataset?.suggestions;
    if (raw) {
      try { 
        const arr = JSON.parse(raw);
        if (Array.isArray(arr)) return arr;
      } catch (e) { /* ignore parse error */ }
    }
    // Then individual data-s1, data-s2...
    const items = [];
    for (let i = 1; i <= 8; i++) {
      const v = script?.dataset?.[`s${i}`];
      if (v) items.push(v);
    }
    if (items.length) return items;

    // Fallback defaults
    return [
      "What does the cover letter say?",
      "List the main skills mentioned.",
      "Summarize the work experience.",
      "What should I improve in the cover letter?"
    ];
  }
  const SUGGESTIONS = getSuggestions();

  // ---- Styles ----
  const css = `
.ai-bubble { position: fixed; right: 20px; bottom: 20px; width: 56px; height: 56px;
  border-radius: 50%; border: none; background:#111; color:#fff; font-size:20px;
  box-shadow: 0 8px 24px rgba(0,0,0,.2); cursor:pointer; z-index: 999999; }
.ai-box { position: fixed; right: 20px; bottom: 90px; width: 360px; max-width: calc(100% - 40px);
  height: 520px; background: #fff; border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,.3);
  display: none; flex-direction: column; overflow: hidden; z-index: 999998; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.ai-head { padding: 10px 14px; font-weight: 600; border-bottom: 1px solid #eee; display:flex; justify-content:space-between; align-items:center;}
.ai-close { background: transparent; border: none; font-size: 18px; cursor: pointer; }
.ai-messages { padding: 12px; flex:1; overflow:auto; font-size: 14px; background: #fafafa; }
.ai-msg { margin: 8px 0; line-height: 1.5; white-space: pre-wrap; }
.ai-msg.user { text-align: right; }
.ai-suggestions { padding: 10px 12px; border-top: 1px solid #eee; background: #fff; display:flex; flex-wrap: wrap; gap:8px; }
.ai-suggestions button { border:1px solid #ddd; background:#f9f9f9; border-radius:999px; padding:6px 10px; font-size:13px; cursor:pointer; }
.ai-suggestions button:hover { background:#f0f0f0; }
.ai-input { display:flex; gap: 8px; padding: 12px; border-top: 1px solid #eee; background: #fff; }
.ai-input input { flex:1; padding:10px 12px; border-radius: 10px; border:1px solid #ddd; font-size:14px; }
.ai-input button { padding: 10px 14px; border-radius: 10px; border: none; background: #111; color:#fff; cursor:pointer; font-size:14px; }
@media (max-width: 600px) {
  .ai-box { right: 10px; left: 10px; bottom: 80px; width: auto; height: 70vh; }
}`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  // ---- Bubble ----
  const bubble = document.createElement("button");
  bubble.className = "ai-bubble";
  bubble.setAttribute("aria-label", "Open AI chat");
  bubble.textContent = "🤖";
  document.body.appendChild(bubble);

  // ---- Panel ----
  const box = document.createElement("div");
  box.className = "ai-box";
  box.setAttribute("role", "dialog");
  box.setAttribute("aria-modal", "true");
  box.setAttribute("aria-label", "AI chat");

  box.innerHTML = `
    <div class="ai-head">
      <span>Ask our AI</span>
      <button class="ai-close" aria-label="Close">✕</button>
    </div>
    <div class="ai-messages" id="aiMessages"></div>
    <div class="ai-suggestions" id="aiSuggestions" aria-label="Suggested questions"></div>
    <div class="ai-input">
      <input id="aiInput" type="text" placeholder="Type your question…" />
      <button id="aiSend">Send</button>
    </div>
  `;
  document.body.appendChild(box);

  const closeBtn = box.querySelector(".ai-close");
  const messages = box.querySelector("#aiMessages");
  const input = box.querySelector("#aiInput");
  const send = box.querySelector("#aiSend");
  const sugEl = box.querySelector("#aiSuggestions");

  function append(text, who) {
    const div = document.createElement("div");
    div.className = `ai-msg ${who || "bot"}`;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  async function ask(q) {
    append(q, "user");
    append("…", "bot");
    // hide suggestions after first use (change to keep showing if you want)
    if (sugEl) sugEl.style.display = "none";
    try {
      const r = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q })
      });
      const data = await r.json();
      messages.lastChild.textContent = (data && data.answer) ? data.answer : "(no answer)";
    } catch (e) {
      messages.lastChild.textContent = "Network error. Please try again.";
    }
  }

  function renderSuggestions() {
    if (!sugEl) return;
    if (!SUGGESTIONS || !SUGGESTIONS.length) { sugEl.style.display = "none"; return; }
    sugEl.innerHTML = "";
    SUGGESTIONS.slice(0, 8).forEach((text) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      b.addEventListener("click", () => ask(text));
      sugEl.appendChild(b);
    });
  }

  bubble.addEventListener("click", () => {
    box.style.display = "flex";
    renderSuggestions();
    setTimeout(() => input.focus(), 0);
  });
  closeBtn.addEventListener("click", () => { box.style.display = "none"; });

  send.addEventListener("click", () => {
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    ask(q);
  });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send.click(); });
})();
