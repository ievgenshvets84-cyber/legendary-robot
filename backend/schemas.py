"""Pydantic-Schemata für die REST-API von LLM-Forge.

Diese Modelle validieren ein- und ausgehende Daten der FastAPI-Endpunkte
und dienen zugleich als API-Dokumentation (OpenAPI/Swagger).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Modelle / Architektur
# ---------------------------------------------------------------------------
class ModelArchitecture(BaseModel):
    """Konfigurierbare Architekturparameter eines neuen Modells."""

    name: str = Field(..., min_length=1, max_length=64, description="Eindeutiger Modellname")
    vocab_size: int = Field(8192, ge=256, le=200_000)
    hidden_size: int = Field(512, ge=32, le=8192)
    num_layers: int = Field(8, ge=1, le=80)
    num_heads: int = Field(8, ge=1, le=64)
    num_kv_heads: int = Field(4, ge=1, le=64)
    intermediate_size: int = Field(1536, ge=32, le=32_768)
    max_seq_len: int = Field(1024, ge=8, le=131_072)
    rope_theta: float = Field(10000.0, gt=0)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    rms_norm_eps: float = Field(1e-5, gt=0)
    tie_word_embeddings: bool = True


class ModelInfo(BaseModel):
    """Zusammenfassende Modellinformation für die Übersicht."""

    name: str
    config: dict
    tokenizer_path: str | None = None
    num_parameters: int | None = None
    created_at: float | None = None


class UpdateArchitecture(BaseModel):
    """Teilaktualisierung der Architektur (nur gesetzte Felder wirken)."""

    hidden_size: int | None = Field(None, ge=32, le=8192)
    num_layers: int | None = Field(None, ge=1, le=80)
    num_heads: int | None = Field(None, ge=1, le=64)
    num_kv_heads: int | None = Field(None, ge=1, le=64)
    intermediate_size: int | None = Field(None, ge=32, le=32_768)
    max_seq_len: int | None = Field(None, ge=8, le=131_072)
    rope_theta: float | None = Field(None, gt=0)
    dropout: float | None = Field(None, ge=0.0, lt=1.0)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class TokenizerTrainRequest(BaseModel):
    """Anfrage zum Training eines Tokenizers."""

    model_name: str = Field(..., description="Modell, dem der Tokenizer zugeordnet wird")
    vocab_size: int = Field(8192, ge=260, le=200_000)
    min_frequency: int = Field(2, ge=1)
    dataset_files: list[str] | None = Field(
        None, description="Optionale Auswahl an Rohdateien; Standard: alle"
    )


class TokenizerInfo(BaseModel):
    vocab_size: int
    path: str


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainingParams(BaseModel):
    """Hyperparameter eines Trainingslaufs (Standardwerte aus der YAML)."""

    learning_rate: float = Field(3e-4, gt=0)
    max_steps: int = Field(2000, ge=1)
    warmup_steps: int = Field(100, ge=0)
    batch_size: int = Field(8, ge=1)
    grad_accum_steps: int = Field(4, ge=1)
    seq_len: int = Field(1024, ge=8)
    eval_interval: int = Field(200, ge=1)
    checkpoint_interval: int = Field(500, ge=1)
    weight_decay: float = Field(0.1, ge=0)
    grad_clip: float = Field(1.0, ge=0)
    mixed_precision: bool = True
    device: str = Field("auto", pattern="^(auto|cuda|cpu)$")


class StartTrainingRequest(BaseModel):
    model_name: str
    dataset_name: str = "train"
    params: TrainingParams = Field(default_factory=TrainingParams)
    # Optionales Fine-Tuning ab einem Checkpoint / mit LoRA
    resume_from: str | None = None
    use_lora: bool = False
    lora_rank: int = Field(8, ge=1, le=256)


class TrainingStatus(BaseModel):
    status: str
    step: int = 0
    max_steps: int = 0
    progress: float = 0.0
    train_loss: float | None = None
    val_loss: float | None = None
    learning_rate: float | None = None
    tokens_per_sec: float | None = None
    elapsed_sec: float | None = None
    best_val_loss: float | None = None
    message: str = ""
    run_id: int | None = None


# ---------------------------------------------------------------------------
# Inferenz / Chat
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """Anfrage zur Textgenerierung / für das Chatfenster."""

    model_name: str
    prompt: str = Field(..., min_length=1)
    checkpoint: str = Field("best", description="'best', 'latest' oder Dateiname")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: int = Field(40, ge=0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, gt=0.0)
    stream: bool = True


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    num_tokens: int


# ---------------------------------------------------------------------------
# Export / Quantisierung / Evaluation
# ---------------------------------------------------------------------------
class ExportRequest(BaseModel):
    model_name: str
    checkpoint: str = "best"
    format: str = Field("onnx", pattern="^(onnx|gguf)$")


class QuantizeRequest(BaseModel):
    model_name: str
    checkpoint: str = "best"
    bits: int = Field(8, ge=4, le=8)
    group_size: int = Field(64, ge=8)


class EvaluateRequest(BaseModel):
    model_name: str
    checkpoint: str = "best"
    dataset_name: str = "train"
    max_batches: int = Field(50, ge=1)


class MessageResponse(BaseModel):
    """Generische Erfolgs-/Statusantwort."""

    ok: bool = True
    message: str = ""
    data: dict | None = None
