# 🔥 LLM-Forge

Ein vollständiges, lauffähiges Framework, um ein **eigenes großes Sprachmodell (LLM) von Grund auf** zu entwickeln, zu trainieren und zu betreiben – mit eigener Architektur, eigenem Tokenizer, eigenen Gewichten und einer transparenten Trainingspipeline.

Offene Modelle (Gemma, Llama, Mistral) dienen **ausschließlich als technische Referenz** für die Architektur und optional zur Wissensdistillation. Es werden **keine fremden Gewichte kopiert oder übernommen** – jedes mit LLM-Forge trainierte Modell ist eigenständig.

---

## Inhalt

- [Funktionsumfang](#funktionsumfang)
- [Architektur des Modells](#architektur-des-modells)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Schnellstart – Webinterface](#schnellstart--webinterface)
- [Schnellstart – Kommandozeile](#schnellstart--kommandozeile)
- [Docker](#docker)
- [REST-API](#rest-api)
- [Tests](#tests)
- [Konfiguration](#konfiguration)

---

## Funktionsumfang

| # | Komponente | Modul |
|---|------------|-------|
| 1 | Eigener Byte-Level-BPE-Tokenizer | `tokenizer/bpe_tokenizer.py` |
| 2 | Datensatzverwaltung (Upload, Packing, Memory-Mapping) | `training/dataset.py` |
| 3 | Datenbereinigung (Unicode, Dedup, Filter) | `training/data_cleaning.py` |
| 4 | Trainingspipeline (AdamW, Warmup-Kosinus, AMP, Grad-Accum) | `training/trainer.py` |
| 5 | Decoder-Transformer (RoPE, RMSNorm, GQA, GeGLU, Residuen) | `models/` |
| 6 | Inferenz mit KV-Cache und Sampling (Top-k/Top-p) | `inference/generator.py` |
| 7 | Checkpoint-System (best/latest, Pruning) | `training/checkpoint.py` |
| 8 | Quantisierung 4/8 Bit (gruppenweise, eigenständig) | `inference/quantization.py` |
| 9 | Fine-Tuning (ab Checkpoint fortsetzen) | `backend/services/forge_service.py` |
| 10 | LoRA-Unterstützung | `models/lora.py` |
| 11 | Modellbewertung (Perplexität) | `evaluation/metrics.py` |
| 12 | Export nach ONNX und GGUF | `inference/export.py` |
| + | Optionale Distillation von offenen Referenzmodellen | `training/distillation.py` |
| + | Web-Backend (FastAPI) + SQLite | `backend/` |
| + | Web-Frontend (HTML/CSS/JS, Live-Charts, Chat) | `frontend/` |

---

## Architektur des Modells

LLM-Forge implementiert einen modernen **Decoder-only-Transformer** – vollständig eigenständig, ohne Übernahme fremder Gewichte:

```
Token-Embedding
  └─ N × DecoderBlock
        ├─ RMSNorm → Grouped-Query-Attention (RoPE) → + Residuum
        └─ RMSNorm → GeGLU-Feed-Forward            → + Residuum
  └─ finale RMSNorm
  └─ LM-Head (optional an das Embedding gebunden)
```

- **Rotary Positional Embeddings (RoPE)** kodieren die Position durch Rotation der Query/Key-Vektoren.
- **RMSNorm** normalisiert ohne Mittelwertzentrierung und Bias (schnell und stabil).
- **Grouped Query Attention (GQA)** teilt sich Key/Value-Köpfe über mehrere Query-Köpfe (`num_kv_heads < num_heads`); bei Gleichheit klassische Multi-Head-Attention.
- **GeGLU-Feed-Forward** nutzt eine GELU-gegatete Projektion.
- **KV-Cache** beschleunigt die autoregressive Generierung.

Die gesamte Architektur wird über eine einzige `ModelConfig` (`models/config.py`) beschrieben und skaliert von wenigen Millionen bis zu mehreren Milliarden Parametern ohne Codeänderung.

---

## Projektstruktur

```
legendary-robot/
├── frontend/        # Webinterface (HTML, CSS, Vanilla-JS, Canvas-Charts)
├── backend/         # FastAPI-App, Router, Dienste, SQLite-Schicht
│   ├── routers/     # API-Endpunkte (Modelle, Daten, Training, Inferenz, …)
│   └── services/    # Zentrale Geschäftslogik (ForgeService)
├── models/          # Transformer-Architektur (config, layers, transformer, lora)
├── tokenizer/       # Eigener BPE-Tokenizer
├── training/        # Dataset, Bereinigung, Trainer, Checkpoints, Distillation
├── inference/       # Generierung, Quantisierung, Export (ONNX/GGUF)
├── evaluation/      # Metriken (Perplexität)
├── datasets/        # raw/ (Uploads) und processed/ (Token-Ströme); samples/
├── checkpoints/     # Gespeicherte Modell-Checkpoints
├── exports/         # ONNX-/GGUF-/quantisierte Ausgaben
├── config/          # YAML-Konfigurationen + SQLite-Datenbank
├── logs/            # Logdateien
├── tests/           # Unit- und Integrationstests (pytest)
└── scripts/         # CLI-Werkzeuge (End-to-End-Demo)
```

---

## Installation

**Voraussetzung:** Python 3.12+

### Automatisch (empfohlen)

```bash
./install.sh          # erkennt NVIDIA-GPU automatisch und wählt CUDA-Torch
./install.sh --cpu    # erzwingt CPU-Variante
./install.sh --dev    # zusätzlich Entwicklungswerkzeuge (pytest, ruff, mypy)
```

### Manuell

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # CPU
# GPU (CUDA 12.1):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Optionale Zusatzfunktionen:

```bash
pip install ".[export]"   # ONNX
pip install ".[distill]"  # Distillation von offenen Referenzmodellen
pip install ".[gpu]"      # GPU-Telemetrie im Webinterface (nvidia-ml)
pip install regex         # exaktere Unicode-Vortokenisierung im Tokenizer
```

---

## Schnellstart – Webinterface

```bash
source .venv/bin/activate
python -m backend.main
```

Anschließend im Browser öffnen: **http://localhost:8000**

Der Arbeitsablauf im Webinterface folgt den Tabs:

1. **Modell** – neue Architektur konfigurieren (Layer, Hidden Size, Heads, GQA, Kontextlänge …) und anlegen.
2. **Daten & Tokenizer** – Textdateien (`.txt`/`.jsonl`) hochladen, Tokenizer trainieren, Datensatz bereinigen und tokenisieren.
3. **Training** – Hyperparameter setzen, Training starten/pausieren/fortsetzen/stoppen, Live-Loss-Kurve und GPU-Auslastung beobachten.
4. **Chat & Test** – mit dem trainierten Modell chatten (Streaming, Temperatur, Top-k/Top-p).
5. **Bewertung & Export** – Perplexität berechnen, quantisieren (4/8 Bit), nach ONNX/GGUF exportieren.

---

## Schnellstart – Kommandozeile

Kompletter End-to-End-Durchlauf (Modell → Tokenizer → Daten → Training → Generierung) auf dem mitgelieferten Beispielkorpus:

```bash
python -m scripts.train_cli demo --steps 200 --device cpu
```

---

## Docker

```bash
# CPU
docker compose up llm-forge

# GPU (NVIDIA Container Toolkit erforderlich)
docker compose --profile gpu up llm-forge-gpu
```

Das Webinterface ist danach unter http://localhost:8000 erreichbar. Daten, Checkpoints und Exporte werden über Volumes persistent gehalten.

---

## REST-API

Die interaktive API-Dokumentation (Swagger/OpenAPI) ist nach dem Start unter **http://localhost:8000/docs** verfügbar. Wichtige Endpunkte:

| Methode | Pfad | Zweck |
|---------|------|-------|
| `POST` | `/api/models` | Neues Modell anlegen |
| `PATCH`| `/api/models/{name}` | Architektur bearbeiten |
| `POST` | `/api/datasets/upload` | Datensatz hochladen |
| `POST` | `/api/tokenizer/train` | Tokenizer trainieren |
| `POST` | `/api/datasets/prepare` | Bereinigen & tokenisieren |
| `POST` | `/api/training/start` | Training starten (Hintergrund-Thread) |
| `GET`  | `/api/training/status` | Live-Status |
| `POST` | `/api/training/{pause,resume,stop}` | Steuerung |
| `POST` | `/api/inference/stream` | Chat-Streaming (SSE) |
| `POST` | `/api/inference/evaluate` | Perplexität |
| `POST` | `/api/quantize` | 4/8-Bit-Quantisierung |
| `POST` | `/api/export` | Export nach ONNX/GGUF |
| `GET`  | `/api/system/info` | Gerät, CUDA, GPU-Auslastung |

---

## Tests

```bash
source .venv/bin/activate
pip install pytest httpx
pytest -q
```

Die Suite deckt Tokenizer (inkl. Unicode-Roundtrip), Modell-Bausteine, KV-Cache-Konsistenz, LoRA, Quantisierung, Datenbereinigung, die komplette Trainings-/Generierungs-/Export-Pipeline sowie die FastAPI-Endpunkte ab.

---

## Konfiguration

Alle Standardwerte liegen als YAML unter `config/`:

- `config/model_default.yaml` – Modellarchitektur.
- `config/training_default.yaml` – Trainingshyperparameter.
- `config/server.yaml` – Server- und Pfadeinstellungen.

Umgebungsvariablen überschreiben die Serverkonfiguration: `LLM_FORGE_HOST`, `LLM_FORGE_PORT`, `LLM_FORGE_DB`.

---

## Hinweis zur Eigenständigkeit

LLM-Forge erzeugt Modelle mit **eigener Architektur, eigenen Gewichten und eigener Trainingspipeline**. Offene Modelle dürfen laut Projektziel nur als technische Referenz oder – optional – für Distillation dienen (`training/distillation.py` überträgt dabei ausschließlich weiche Ausgabewahrscheinlichkeiten als Trainingssignal, niemals Gewichte).

---

## Lizenz

Siehe [LICENSE](LICENSE).
