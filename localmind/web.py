"""Локальный веб-интерфейс LocalMind.

Простой браузерный чат на стандартной библиотеке (http.server), без внешних
зависимостей. Важное отличие от «браузер → Ollama напрямую»: страницу и API
обслуживает наш собственный сервер, а он уже ходит в Ollama на стороне сервера.
Поэтому CORS и переменная OLLAMA_ORIGINS не нужны — типовая ошибка
«нет доступа из браузера» здесь не возникает.

Запуск:  python -m localmind web   (откроет http://127.0.0.1:8770)
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


# ── чистая логика запросов (тестируется без поднятия сервера) ─────────────
def status_payload(app: Any) -> dict[str, Any]:
    ok = app.llm.available()
    model = app.cfg.llm["model"]
    models = app.llm.installed_models() if ok else []
    model_installed = ok and _model_present(model, models)
    image_ok = False
    image_host = ""
    media = getattr(app, "media", None)
    if media is not None:
        image_host = getattr(media, "image_host", "")
        try:
            image_ok = media.image_available()
        except Exception:
            image_ok = False
    return {
        "ollama": ok,
        "host": app.cfg.llm["host"],
        "model": model,
        "models": models,
        "model_installed": model_installed,
        "tools": app.registry.names(),
        "skills": list(app.loaded_skills),
        "memory": app.memory.stats(),
        "image_server": image_ok,
        "image_host": image_host,
    }


def _model_present(model: str, installed: list) -> bool:
    """Модель считается установленной при точном совпадении или совпадении
    базового имени (без тега), т.к. 'qwen2.5' и 'qwen2.5:latest' — одно и то же."""
    if model in installed:
        return True
    base = model.split(":", 1)[0]
    return any(m.split(":", 1)[0] == base for m in installed)


def handle_chat(app: Any, state: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    message = str(data.get("message", "")).strip()
    if not message:
        return {"error": "Пустое сообщение."}
    recall = app.memory.recall_block(message)
    turn = message if not recall else f"{recall}\n\nВопрос: {message}"
    convo = state["history"] + [{"role": "user", "content": turn}]
    try:
        reply = app.llm.chat(convo)
    except Exception as exc:  # нет связи с Ollama и пр.
        return {"error": str(exc)}
    # В историю кладём чистое сообщение пользователя (без блока воспоминаний).
    state["history"].append({"role": "user", "content": message})
    state["history"].append({"role": "assistant", "content": reply})
    app.memory.add(f"Пользователь: {message}\nОтвет: {reply}", kind="dialog")
    return {"reply": reply}


def handle_agent(app: Any, data: dict[str, Any]) -> dict[str, Any]:
    task = str(data.get("task", "")).strip()
    if not task:
        return {"error": "Пустая задача."}
    events: list[dict[str, str]] = []

    def on_event(kind: str, text: str) -> None:
        events.append({"kind": kind, "text": text})

    try:
        result = app.agent.run(task, on_event=on_event)
    except Exception as exc:
        return {"error": str(exc), "events": events}
    return {"result": result, "events": events}


def handle_media(app: Any, data: dict[str, Any]) -> dict[str, Any]:
    kind = str(data.get("kind", "image"))
    prompt = str(data.get("prompt", "")).strip()
    file = str(data.get("file", "")).strip()
    try:
        if kind == "image":
            if not prompt:
                return {"error": "Нужен текстовый запрос для изображения."}
            paths = app.media.generate_image(prompt, count=int(data.get("count", 1) or 1))
        elif kind == "img2img":
            if not file or not prompt:
                return {"error": "Нужны исходная картинка и запрос."}
            paths = app.media.image_to_image(prompt, init_image=file)
        elif kind == "upscale":
            if not file:
                return {"error": "Нет изображения для апскейла."}
            paths = app.media.upscale_image(file, scale=float(data.get("scale", 2) or 2))
        elif kind == "img2video":
            if not file:
                return {"error": "Нет изображения для анимации."}
            paths = app.media.image_to_video(init_image=file, prompt=prompt)
        else:
            return {"error": f"Неизвестный режим медиа: {kind}"}
    except Exception as exc:
        return {"error": str(exc)}
    return {"files": _media_files(app, paths)}


def _media_files(app: Any, paths: list) -> list[dict[str, str]]:
    workspace = app.media.guard.workspace
    out: list[dict[str, str]] = []
    for p in paths:
        try:
            rel = str(p.relative_to(workspace))
        except ValueError:
            rel = str(p)
        out.append({"name": p.name, "url": "/media/" + p.name, "path": rel})
    return out


_CTYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".mp4": "video/mp4",
}


def new_state() -> dict[str, Any]:
    return {"history": [{
        "role": "system",
        "content": ("Ты — LocalMind, локальный дружелюбный ассистент. "
                    "Отвечай по-русски, кратко и по делу."),
    }]}


# ── HTTP-обвязка ─────────────────────────────────────────────────────────
def make_handler(app: Any) -> type[BaseHTTPRequestHandler]:
    state = new_state()

    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalMind/0.1"

        def log_message(self, *args: Any) -> None:  # тишина в консоли
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: dict[str, Any], code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(status_payload(app))
            elif path.startswith("/media/"):
                self._serve_media(path[len("/media/"):])
            else:
                self._json({"error": "not found"}, 404)

        def _serve_media(self, name: str) -> None:
            if not name or "/" in name or "\\" in name or ".." in name:
                self._json({"error": "bad name"}, 400)
                return
            try:
                target = app.media.guard.resolve_path(app.media.save_dir + "/" + name)
            except Exception:
                self._json({"error": "forbidden"}, 403)
                return
            if not target.exists() or not target.is_file():
                self._json({"error": "not found"}, 404)
                return
            ctype = _CTYPES.get(target.suffix.lower(), "application/octet-stream")
            self._send(200, target.read_bytes(), ctype)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            data = self._read_json()
            if path == "/api/chat":
                self._json(handle_chat(app, state, data))
            elif path == "/api/agent":
                self._json(handle_agent(app, data))
            elif path == "/api/media":
                self._json(handle_media(app, data))
            elif path == "/api/reset":
                state.clear()
                state.update(new_state())
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def serve(app: Any, host: str = "127.0.0.1", port: int = 8770,
          open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}"
    print(f"LocalMind — веб-интерфейс: {url}")
    print("Остановить: Ctrl+C")
    if not app.llm.available():
        print(f"[!] Ollama пока недоступен на {app.cfg.llm['host']}. "
              "Запустите `ollama serve` — страница подхватит связь автоматически.")
    if open_browser:
        threading.Timer(0.8, lambda: _safe_open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        httpd.server_close()


def _safe_open(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


# ── одностраничный интерфейс ─────────────────────────────────────────────
INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalMind</title>
<style>
  :root {
    --bg:#f6f7f9; --panel:#ffffff; --text:#1a1c1e; --muted:#6b7280;
    --border:#e3e6ea; --accent:#4f46e5; --user:#4f46e5; --user-text:#fff;
    --bot:#eef0f4; --bot-text:#1a1c1e; --warn-bg:#fdecea; --warn-text:#8a1c14;
    --ok:#1a7f37; --bad:#c0392b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#0f1115; --panel:#171a21; --text:#e6e8eb; --muted:#9aa3af;
      --border:#242833; --accent:#7c74ff; --user:#5b54e6; --user-text:#fff;
      --bot:#212632; --bot-text:#e6e8eb; --warn-bg:#3a1d1a; --warn-text:#ffb4ab;
      --ok:#3fb950; --bad:#ff6b5e;
    }
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--text); height:100vh; display:flex;
         flex-direction:column; }
  header { padding:12px 18px; border-bottom:1px solid var(--border);
           background:var(--panel); display:flex; align-items:center; gap:14px; }
  header h1 { font-size:17px; margin:0; font-weight:650; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:6px; }
  .dot.ok { background:var(--ok); } .dot.bad { background:var(--bad); }
  #status { font-size:13px; color:var(--muted); margin-left:auto; }
  .modes { display:flex; gap:4px; background:var(--bot); border-radius:9px; padding:3px; }
  .modes button { border:0; background:transparent; color:var(--muted); padding:5px 12px;
                  border-radius:7px; font-size:13px; cursor:pointer; }
  .modes button.active { background:var(--panel); color:var(--text); font-weight:600;
                         box-shadow:0 1px 2px rgba(0,0,0,.12); }
  #warn { display:none; background:var(--warn-bg); color:var(--warn-text);
          padding:10px 18px; font-size:13px; line-height:1.5; border-bottom:1px solid var(--border); }
  #warn code { background:rgba(0,0,0,.12); padding:1px 5px; border-radius:4px; }
  main { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:14px; }
  .wrap { max-width:820px; width:100%; margin:0 auto; display:flex; flex-direction:column; gap:14px; }
  .msg { display:flex; }
  .msg.user { justify-content:flex-end; }
  .bubble { max-width:78%; padding:11px 14px; border-radius:14px; font-size:15px;
            line-height:1.5; white-space:pre-wrap; word-wrap:break-word; }
  .user .bubble { background:var(--user); color:var(--user-text); border-bottom-right-radius:5px; }
  .bot .bubble { background:var(--bot); color:var(--bot-text); border-bottom-left-radius:5px; }
  .trace { max-width:78%; font-size:12.5px; color:var(--muted); border:1px solid var(--border);
           border-radius:10px; padding:8px 10px; background:var(--panel); }
  .trace summary { cursor:pointer; color:var(--text); }
  .trace .step { margin-top:6px; }
  .trace .k { color:var(--accent); font-weight:600; }
  footer { border-top:1px solid var(--border); background:var(--panel); padding:12px 18px; }
  .inrow { max-width:820px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; border:1px solid var(--border); border-radius:11px;
             padding:11px 13px; font:inherit; font-size:15px; background:var(--bg);
             color:var(--text); max-height:160px; }
  textarea:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  .send { border:0; background:var(--accent); color:#fff; border-radius:11px; padding:0 18px;
          height:44px; font-size:15px; font-weight:600; cursor:pointer; }
  .send:disabled { opacity:.5; cursor:default; }
  .reset { border:1px solid var(--border); background:transparent; color:var(--muted);
           border-radius:11px; height:44px; padding:0 12px; cursor:pointer; font-size:13px; }
  .hint { max-width:820px; margin:6px auto 0; font-size:12px; color:var(--muted); }
  .bubble.media { background:var(--panel); border:1px solid var(--border); max-width:88%;
                  display:flex; flex-direction:column; gap:12px; }
  .mediaItem { display:flex; flex-direction:column; gap:6px; }
  .thumb { max-width:100%; border-radius:9px; border:1px solid var(--border); }
  .acts { display:flex; gap:6px; flex-wrap:wrap; }
  .acts button { border:1px solid var(--border); background:var(--bg); color:var(--text);
                 border-radius:8px; padding:4px 10px; font-size:12.5px; cursor:pointer; }
  .acts button:hover { border-color:var(--accent); color:var(--accent); }
  .caption { font-size:11.5px; color:var(--muted); word-break:break-all; }
</style>
</head>
<body>
<header>
  <h1>🧠 LocalMind</h1>
  <div class="modes">
    <button id="mChat" class="active" onclick="setMode('chat')">Чат</button>
    <button id="mAgent" onclick="setMode('agent')">Агент</button>
    <button id="mMedia" onclick="setMode('media')">Медиа</button>
  </div>
  <span id="status"><span class="dot bad"></span>проверка…</span>
</header>
<div id="warn"></div>
<main><div class="wrap" id="messages"></div></main>
<footer>
  <div class="inrow">
    <textarea id="input" rows="1" placeholder="Спросите что-нибудь…"></textarea>
    <button class="send" id="send" onclick="send()">Отправить</button>
    <button class="reset" onclick="resetChat()" title="Очистить диалог">Сброс</button>
  </div>
  <div class="hint" id="hint">Режим «Чат» — обычный диалог. Режим «Агент» — задача с инструментами и памятью.</div>
</footer>
<script>
let mode = "chat";
const messages = document.getElementById("messages");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

function setMode(m){
  mode = m;
  document.getElementById("mChat").classList.toggle("active", m==="chat");
  document.getElementById("mAgent").classList.toggle("active", m==="agent");
  document.getElementById("mMedia").classList.toggle("active", m==="media");
  input.placeholder = m==="chat" ? "Спросите что-нибудь…"
    : m==="agent" ? "Опишите задачу для агента…"
    : "Опишите картинку для генерации…";
  document.getElementById("hint").textContent = m==="media"
    ? "Режим «Медиа»: опишите картинку и нажмите Отправить. У готовых картинок есть кнопки Апскейл и Оживить."
    : "Режим «Чат» — обычный диалог. Режим «Агент» — задача с инструментами и памятью.";
  updateBanner();
}
input.addEventListener("input", ()=>{ input.style.height="auto"; input.style.height=Math.min(input.scrollHeight,160)+"px"; });
input.addEventListener("keydown", e=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); send(); }});

function addMsg(text, who){
  const row = document.createElement("div"); row.className = "msg "+who;
  const b = document.createElement("div"); b.className="bubble"; b.textContent=text;
  row.appendChild(b); messages.appendChild(row); scroll(); return b;
}
function addTrace(events){
  if(!events || !events.length) return;
  const d = document.createElement("details"); d.className="trace";
  const s = document.createElement("summary"); s.textContent="Ход рассуждений ("+events.length+")";
  d.appendChild(s);
  for(const e of events){
    const p = document.createElement("div"); p.className="step";
    p.innerHTML = '<span class="k">'+e.kind+':</span> ';
    p.appendChild(document.createTextNode(e.text));
    d.appendChild(p);
  }
  messages.appendChild(d); scroll();
}
function scroll(){ messages.parentElement.scrollTop = messages.parentElement.scrollHeight; }

function post(url, body){
  return fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json());
}
function mkBtn(label, fn){ const b=document.createElement("button"); b.textContent=label; b.onclick=fn; return b; }

function addMedia(files){
  const row=document.createElement("div"); row.className="msg bot";
  const box=document.createElement("div"); box.className="bubble media";
  for(const f of files){
    const item=document.createElement("div"); item.className="mediaItem";
    let el;
    if(/\.(mp4)$/i.test(f.name)){ el=document.createElement("video"); el.src=f.url; el.controls=true; }
    else { el=document.createElement("img"); el.src=f.url; el.loading="lazy"; }
    el.className="thumb"; item.appendChild(el);
    const acts=document.createElement("div"); acts.className="acts";
    const a=document.createElement("a"); a.href=f.url; a.download=f.name; a.appendChild(mkBtn("Скачать", ()=>{}));
    acts.appendChild(a);
    if(!/\.(mp4)$/i.test(f.name)){
      acts.appendChild(mkBtn("Апскейл ×2", ()=>mediaAction("upscale", f.path)));
      acts.appendChild(mkBtn("Оживить", ()=>mediaAction("img2video", f.path)));
    }
    item.appendChild(acts);
    const cap=document.createElement("div"); cap.className="caption"; cap.textContent=f.path; item.appendChild(cap);
    box.appendChild(item);
  }
  row.appendChild(box); messages.appendChild(row); scroll();
}

async function mediaAction(kind, file){
  const pending = addMsg("… "+(kind==="upscale"?"апскейл":"анимация"), "bot");
  try {
    const j = await post("/api/media",{kind:kind, file:file, prompt:input.value.trim()});
    pending.remove();
    if(j.error){ addMsg("⚠ "+j.error, "bot"); } else { addMedia(j.files); }
  } catch(err){ pending.textContent="⚠ Ошибка сети: "+err; }
}

async function send(){
  const text = input.value.trim(); if(!text) return;
  addMsg(text, "user"); input.value=""; input.style.height="auto";
  sendBtn.disabled = true;
  const pending = addMsg("…", "bot");
  try {
    if(mode==="chat"){
      const j = await post("/api/chat",{message:text});
      pending.textContent = j.error ? ("⚠ "+j.error) : j.reply;
    } else if(mode==="agent"){
      const j = await post("/api/agent",{task:text});
      pending.textContent = j.error ? ("⚠ "+j.error) : j.result;
      addTrace(j.events);
    } else {
      const j = await post("/api/media",{kind:"image", prompt:text});
      pending.remove();
      if(j.error){ addMsg("⚠ "+j.error, "bot"); } else { addMedia(j.files); }
    }
  } catch(err){
    pending.textContent = "⚠ Ошибка сети: "+err;
  } finally {
    sendBtn.disabled = false; input.focus();
  }
}
async function resetChat(){
  await fetch("/api/reset",{method:"POST"});
  messages.innerHTML=""; input.focus();
}
let lastStatus = {};
function updateBanner(){
  const st = document.getElementById("status");
  const warn = document.getElementById("warn");
  const j = lastStatus;
  // 1) Ollama важнее всего: без него не работают чат и агент.
  if(j.ollama === false){
    st.innerHTML = '<span class="dot bad"></span>Ollama не подключён';
    warn.style.display="block";
    warn.innerHTML = 'Нет связи с Ollama. Откройте PowerShell и запустите: '+
      '<code>ollama serve</code> — затем страница подхватит связь сама. '+
      'Модель ставится командой <code>ollama pull '+(j.model||'')+'</code>.';
    return;
  }
  if(j.ollama === true){
    st.innerHTML = '<span class="dot ok"></span>Ollama · '+j.model;
  }
  // 2) Ollama работает, но нужная модель не установлена → 404 при запросе.
  if(j.ollama === true && j.model_installed === false){
    warn.style.display="block";
    warn.innerHTML = 'Ollama запущен, но модель <b>'+j.model+'</b> не установлена '+
      '(поэтому запросы падают с 404). Выполните в PowerShell: '+
      '<code>ollama pull '+j.model+'</code> — затем страница подхватит модель сама.';
    return;
  }
  // 3) В режиме «Медиа» подсказываем про отдельный сервер картинок.
  if(mode==="media" && j.image_server === false){
    warn.style.display="block";
    warn.innerHTML = 'Для генерации картинок нужен ОТДЕЛЬНЫЙ сервер '+
      '<b>Stable Diffusion</b> ('+(j.image_host||'http://localhost:7860')+'), это НЕ Ollama. '+
      'Запустите Automatic1111/SD.Next с флагом <code>--api</code> — см. раздел «Медиа» в README. '+
      'Чат и Агент работают без него.';
  } else {
    warn.style.display="none";
  }
}
async function refreshStatus(){
  try {
    lastStatus = await (await fetch("/api/status")).json();
    updateBanner();
  } catch(e){ /* сервер перезапускается */ }
}
refreshStatus(); setInterval(refreshStatus, 5000); input.focus();
</script>
</body>
</html>
"""
