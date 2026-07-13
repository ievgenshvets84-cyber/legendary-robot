/* LLM-Forge – Frontend-Anwendungslogik.
   Verdrahtet die Oberfläche mit der API, verwaltet Tabs, Live-Training und Chat. */

// ---------- Hilfsfunktionen ----------
const $ = (id) => document.getElementById(id);

function toast(message, type = "") {
    const el = $("toast");
    el.textContent = message;
    el.className = `toast show ${type}`;
    setTimeout(() => { el.className = "toast"; }, 3200);
}

function fmt(n, digits = 3) {
    return (n === null || n === undefined) ? "–" : Number(n).toFixed(digits);
}

// ---------- Tab-Navigation ----------
document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        $(`tab-${tab.dataset.tab}`).classList.add("active");
    });
});

// ---------- Modelle ----------
async function refreshModels() {
    try {
        const models = await API.listModels();
        renderModelList(models);
        populateModelSelects(models);
    } catch (e) { toast(e.message, "error"); }
}

function renderModelList(models) {
    const list = $("modelList");
    list.innerHTML = "";
    if (models.length === 0) {
        list.innerHTML = '<p class="hint">Noch keine Modelle angelegt.</p>';
        return;
    }
    for (const m of models) {
        const params = (m.num_parameters / 1e6).toFixed(2);
        const div = document.createElement("div");
        div.className = "list-item";
        div.innerHTML = `
            <div>
                <div class="name">${m.name}</div>
                <div class="meta">${m.config.num_layers}L · ${m.config.hidden_size}d ·
                    ${m.config.num_heads}/${m.config.num_kv_heads} Heads · ${params}M Params
                    · Tokenizer: ${m.tokenizer_path ? "✓" : "✗"}</div>
            </div>
            <button class="btn ghost small" data-del="${m.name}">Löschen</button>`;
        list.appendChild(div);
    }
    list.querySelectorAll("[data-del]").forEach((btn) => {
        btn.addEventListener("click", async () => {
            if (!confirm(`Modell '${btn.dataset.del}' wirklich löschen?`)) return;
            try { await API.deleteModel(btn.dataset.del); toast("Modell gelöscht.", "success"); refreshModels(); }
            catch (e) { toast(e.message, "error"); }
        });
    });
}

function populateModelSelects(models) {
    const selects = ["tk_model", "tr_model", "chat_model", "ev_model", "qz_model", "ex_model"];
    for (const id of selects) {
        const sel = $(id);
        if (!sel) continue;
        const current = sel.value;
        sel.innerHTML = models.map((m) => `<option value="${m.name}">${m.name}</option>`).join("");
        if (current) sel.value = current;
    }
}

// Parameter-Schätzung live (grobe Formel entsprechend Backend)
function estimateParams() {
    const h = +$("m_hidden").value, L = +$("m_layers").value, v = +$("m_vocab").value;
    const i = +$("m_ffn").value, heads = +$("m_heads").value, kv = +$("m_kv").value;
    if (!h || !heads) return;
    const headDim = h / heads;
    const kvDim = kv * headDim;
    const embed = v * h;
    const attn = h * h + 2 * h * kvDim + h * h;
    const ffn = 2 * h * i + i * h;
    const norms = L * 2 * h + h;
    const tie = $("m_tie").checked ? 0 : v * h;
    const total = embed + L * (attn + ffn) + norms + tie;
    $("paramEstimate").textContent =
        `Geschätzte Parameter: ${(total / 1e6).toFixed(2)} M (~${(total * 4 / 1024 / 1024).toFixed(1)} MB fp32)`;
}
["m_hidden", "m_layers", "m_vocab", "m_ffn", "m_heads", "m_kv", "m_tie"].forEach((id) => {
    document.addEventListener("DOMContentLoaded", () => $(id) && $(id).addEventListener("input", estimateParams));
});

$("modelForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const arch = {
        name: $("m_name").value.trim(),
        hidden_size: +$("m_hidden").value,
        num_layers: +$("m_layers").value,
        num_heads: +$("m_heads").value,
        num_kv_heads: +$("m_kv").value,
        intermediate_size: +$("m_ffn").value,
        max_seq_len: +$("m_ctx").value,
        vocab_size: +$("m_vocab").value,
        rope_theta: +$("m_theta").value,
        tie_word_embeddings: $("m_tie").checked,
    };
    try {
        await API.createModel(arch);
        toast(`Modell '${arch.name}' angelegt.`, "success");
        refreshModels();
    } catch (e) { toast(e.message, "error"); }
});
$("refreshModels").addEventListener("click", refreshModels);

// ---------- Datensätze ----------
async function refreshDatasets() {
    try {
        const data = await API.listDatasets();
        const list = $("datasetList");
        list.innerHTML = "";
        if (data.raw.length === 0) {
            list.innerHTML = '<p class="hint">Noch keine Datensätze hochgeladen.</p>';
        }
        for (const f of data.raw) {
            const kb = (f.size_bytes / 1024).toFixed(1);
            const div = document.createElement("div");
            div.className = "list-item";
            div.innerHTML = `<div><span class="name">${f.name}</span>
                <div class="meta">${kb} KB</div></div>`;
            list.appendChild(div);
        }
    } catch (e) { toast(e.message, "error"); }
}

$("uploadBtn").addEventListener("click", async () => {
    const file = $("datasetFile").files[0];
    if (!file) { toast("Bitte eine Datei auswählen.", "error"); return; }
    try {
        await API.uploadDataset(file);
        toast(`'${file.name}' hochgeladen.`, "success");
        refreshDatasets();
    } catch (e) { toast(e.message, "error"); }
});

$("tokenizerForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
        const info = await API.trainTokenizer({
            model_name: $("tk_model").value,
            vocab_size: +$("tk_vocab").value,
            min_frequency: +$("tk_minfreq").value,
        });
        $("tokenizerResult").textContent = `Tokenizer trainiert · Vokabular: ${info.vocab_size}`;
        toast("Tokenizer trainiert.", "success");
        refreshModels();
    } catch (e) { toast(e.message, "error"); }
});

$("prepareBtn").addEventListener("click", async () => {
    try {
        const meta = await API.prepareDataset($("tk_model").value);
        $("prepareResult").textContent =
            `Fertig · ${meta.num_tokens.toLocaleString()} Token (${meta.dtype})`;
        toast("Datensatz aufbereitet.", "success");
    } catch (e) { toast(e.message, "error"); }
});

// ---------- Training ----------
let chart;
let statusTimer = null;

$("trainForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const req = {
        model_name: $("tr_model").value,
        dataset_name: "train",
        params: {
            learning_rate: +$("tr_lr").value,
            max_steps: +$("tr_steps").value,
            batch_size: +$("tr_batch").value,
            grad_accum_steps: +$("tr_accum").value,
            seq_len: +$("tr_seq").value,
            device: $("tr_device").value,
        },
        use_lora: $("tr_lora").checked,
    };
    try {
        chart.reset();
        await API.startTraining(req);
        toast("Training gestartet.", "success");
        startStatusPolling();
    } catch (e) { toast(e.message, "error"); }
});

$("pauseBtn").addEventListener("click", () => API.pauseTraining().then(() => toast("Pausiert.")).catch((e) => toast(e.message, "error")));
$("resumeBtn").addEventListener("click", () => API.resumeTraining().then(() => toast("Fortgesetzt.")).catch((e) => toast(e.message, "error")));
$("stopBtn").addEventListener("click", () => API.stopTraining().then(() => toast("Gestoppt.")).catch((e) => toast(e.message, "error")));

let lastPlottedStep = -1;
function startStatusPolling() {
    if (statusTimer) clearInterval(statusTimer);
    lastPlottedStep = -1;
    statusTimer = setInterval(updateTrainingStatus, 1000);
}

async function updateTrainingStatus() {
    try {
        const s = await API.trainingStatus();
        $("st_status").textContent = s.status;
        $("st_step").textContent = `${s.step}/${s.max_steps}`;
        $("st_loss").textContent = fmt(s.train_loss);
        $("st_val").textContent = fmt(s.val_loss);
        $("st_tps").textContent = fmt(s.tokens_per_sec, 0);
        $("progressBar").style.width = `${(s.progress * 100).toFixed(1)}%`;

        // Metrikreihe des aktiven Laufs nachladen und plotten
        if (s.run_id) {
            const metrics = await API.runMetrics(s.run_id);
            chart.reset();
            for (const m of metrics) {
                chart.push("Train-Loss", m.step, m.train_loss);
                if (m.val_loss !== null) chart.push("Val-Loss", m.step, m.val_loss);
            }
            chart.draw();
        }

        // Bei Abschluss Polling beenden
        if (["completed", "stopped", "error", "idle"].includes(s.status) && s.step >= s.max_steps || s.status === "error") {
            if (["completed", "stopped", "error"].includes(s.status)) {
                clearInterval(statusTimer); statusTimer = null;
                if (s.message) toast(s.message, s.status === "error" ? "error" : "success");
            }
        }
    } catch (e) { /* still tolerant */ }
}

// ---------- GPU-Anzeige ----------
async function updateGpu() {
    try {
        const gpus = await API.gpuInfo();
        const el = $("gpuInfo");
        if (!gpus || gpus.length === 0) {
            el.textContent = "Keine GPU erkannt (CPU-Modus).";
            return;
        }
        el.innerHTML = gpus.map((g) => {
            const util = g.gpu_util_percent !== null ? `${g.gpu_util_percent}%` : "n/a";
            return `<div>
                <strong>GPU ${g.index}: ${g.name}</strong> · Auslastung ${util}
                <div class="gpu-bar"><div class="gpu-bar-fill" style="width:${g.mem_percent}%"></div>
                    <span class="gpu-bar-label">VRAM ${g.mem_used_mb} / ${g.mem_total_mb} MB (${g.mem_percent}%)</span></div>
            </div>`;
        }).join("");
    } catch (e) { /* ignorieren */ }
}

// ---------- Chat ----------
function addMessage(text, cls) {
    const win = $("chatWindow");
    const div = document.createElement("div");
    div.className = `msg ${cls}`;
    div.textContent = text;
    win.appendChild(div);
    win.scrollTop = win.scrollHeight;
    return div;
}

$("chatForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = $("chatPrompt").value.trim();
    if (!prompt) return;
    addMessage(prompt, "user");
    $("chatPrompt").value = "";

    const botDiv = addMessage("", "bot");
    // Streaming über Server-Sent-Events (fetch + ReadableStream)
    try {
        const res = await fetch("/api/inference/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_name: $("chat_model").value,
                checkpoint: $("chat_ckpt").value,
                prompt: prompt,
                max_new_tokens: +$("chat_max").value,
                temperature: +$("chat_temp").value,
            }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || "Fehler");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith("data:")) continue;
                const payload = JSON.parse(line.slice(5).trim());
                if (payload.token) { botDiv.textContent += payload.token; $("chatWindow").scrollTop = 1e9; }
                if (payload.error) botDiv.textContent += `\n[Fehler: ${payload.error}]`;
            }
        }
    } catch (e) {
        botDiv.textContent = `[Fehler: ${e.message}]`;
    }
});

// ---------- Werkzeuge ----------
$("evalBtn").addEventListener("click", async () => {
    try {
        const r = await API.evaluate({ model_name: $("ev_model").value, checkpoint: "best", dataset_name: "train", max_batches: 20 });
        $("evalResult").textContent = `Loss: ${fmt(r.loss)} · Perplexität: ${fmt(r.perplexity)} · Token: ${r.num_tokens}`;
    } catch (e) { toast(e.message, "error"); }
});

$("quantBtn").addEventListener("click", async () => {
    try {
        const r = await API.quantize({ model_name: $("qz_model").value, checkpoint: "best", bits: +$("qz_bits").value });
        $("quantResult").textContent = `Gespeichert: ${r.path}\nVorher ${r.size_before_mb} MB → nachher ${r.size_after_mb} MB`;
        toast("Quantisierung abgeschlossen.", "success");
    } catch (e) { toast(e.message, "error"); }
});

$("exportBtn").addEventListener("click", async () => {
    try {
        const r = await API.exportModel({ model_name: $("ex_model").value, checkpoint: "best", format: $("ex_format").value });
        $("exportResult").textContent = `Exportiert nach: ${r.path}`;
        toast(`Export (${r.format}) abgeschlossen.`, "success");
    } catch (e) { toast(e.message, "error"); }
});

// ---------- Initialisierung ----------
async function init() {
    chart = new LineChart($("lossChart"));
    chart.addSeries("Train-Loss", "#ff6b35");
    chart.addSeries("Val-Loss", "#4c9aff");
    chart.draw();

    estimateParams();
    await refreshModels();
    await refreshDatasets();

    // Systeminfo
    try {
        const info = await API.systemInfo();
        $("deviceBadge").textContent = `Gerät: ${info.device}`;
        $("torchBadge").textContent = `PyTorch: ${info.torch_version}`;
        if (info.training_active) startStatusPolling();
    } catch (e) { /* ignorieren */ }

    // GPU regelmäßig aktualisieren
    updateGpu();
    setInterval(updateGpu, 3000);
}

document.addEventListener("DOMContentLoaded", init);
