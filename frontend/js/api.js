/* LLM-Forge – dünne API-Client-Schicht.
   Kapselt alle fetch-Aufrufe an das FastAPI-Backend. */

const API = {
    async request(method, path, body) {
        const opts = { method, headers: {} };
        if (body !== undefined) {
            opts.headers["Content-Type"] = "application/json";
            opts.body = JSON.stringify(body);
        }
        const res = await fetch(path, opts);
        if (!res.ok) {
            let detail;
            try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
            throw new Error(detail || `HTTP ${res.status}`);
        }
        // 204 / leere Antworten abfangen
        const text = await res.text();
        return text ? JSON.parse(text) : null;
    },

    get(path) { return this.request("GET", path); },
    post(path, body) { return this.request("POST", path, body); },
    patch(path, body) { return this.request("PATCH", path, body); },
    delete(path) { return this.request("DELETE", path); },

    // ---- Modelle ----
    listModels() { return this.get("/api/models"); },
    createModel(arch) { return this.post("/api/models", arch); },
    getModel(name) { return this.get(`/api/models/${encodeURIComponent(name)}`); },
    deleteModel(name) { return this.delete(`/api/models/${encodeURIComponent(name)}`); },

    // ---- Daten & Tokenizer ----
    listDatasets() { return this.get("/api/datasets"); },
    async uploadDataset(file) {
        const fd = new FormData();
        fd.append("file", file);
        const res = await fetch("/api/datasets/upload", { method: "POST", body: fd });
        if (!res.ok) throw new Error((await res.json()).detail || "Upload fehlgeschlagen");
        return res.json();
    },
    prepareDataset(model, output = "train") {
        return this.post(`/api/datasets/prepare?model_name=${encodeURIComponent(model)}&output_name=${output}`);
    },
    trainTokenizer(req) { return this.post("/api/tokenizer/train", req); },

    // ---- Training ----
    startTraining(req) { return this.post("/api/training/start", req); },
    trainingStatus() { return this.get("/api/training/status"); },
    pauseTraining() { return this.post("/api/training/pause"); },
    resumeTraining() { return this.post("/api/training/resume"); },
    stopTraining() { return this.post("/api/training/stop"); },
    runMetrics(runId) { return this.get(`/api/training/runs/${runId}/metrics`); },

    // ---- Inferenz ----
    evaluate(req) { return this.post("/api/inference/evaluate", req); },

    // ---- Management ----
    quantize(req) { return this.post("/api/quantize", req); },
    exportModel(req) { return this.post("/api/export", req); },
    systemInfo() { return this.get("/api/system/info"); },
    gpuInfo() { return this.get("/api/system/gpu"); },
};
