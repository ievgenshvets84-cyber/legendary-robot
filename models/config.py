"""Architektur-Konfiguration des LLM-Forge-Modells.

Die Konfiguration ist eine reine Datenklasse und kann verlustfrei nach/aus
YAML und JSON serialisiert werden. Sie ist die einzige Quelle der Wahrheit
für die Modellarchitektur – Checkpoints, Exporte und das Webinterface
arbeiten alle mit dieser Struktur.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Beschreibt die Architektur eines Decoder-Transformers.

    Attributes:
        name: Anzeigename des Modells.
        vocab_size: Größe des Tokenizer-Vokabulars.
        hidden_size: Breite der Embeddings und Residual-Ströme.
        num_layers: Anzahl der Decoder-Blöcke.
        num_heads: Anzahl der Query-Attention-Köpfe.
        num_kv_heads: Anzahl der Key/Value-Köpfe. Ist der Wert kleiner als
            ``num_heads``, wird Grouped Query Attention (GQA) verwendet;
            bei Gleichheit klassische Multi-Head Attention.
        intermediate_size: Innere Dimension des GeGLU-Feed-Forward-Netzes.
        max_seq_len: Maximale Kontextlänge in Token.
        rope_theta: Basisfrequenz der Rotary Positional Embeddings.
        dropout: Dropout-Wahrscheinlichkeit (0 = deaktiviert).
        rms_norm_eps: Numerisches Epsilon der RMSNorm.
        tie_word_embeddings: Bindet Eingabe-Embedding und Ausgabeprojektion.
    """

    name: str = "forge-base"
    vocab_size: int = 8192
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 1536
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    # Freies Feld für Zusatzinformationen (z. B. Tokenizer-Pfad, Notizen)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Validierung
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Prüft die Konfiguration auf Konsistenz und wirft ``ValueError``."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) muss durch num_heads "
                f"({self.num_heads}) teilbar sein."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) muss durch num_kv_heads "
                f"({self.num_kv_heads}) teilbar sein (GQA-Gruppierung)."
            )
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"Die Kopfdimension ({head_dim}) muss gerade sein, "
                "damit RoPE angewendet werden kann."
            )
        for name in ("vocab_size", "hidden_size", "num_layers", "num_heads",
                     "num_kv_heads", "intermediate_size", "max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} muss eine positive Ganzzahl sein, erhalten: {value!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout muss in [0, 1) liegen, erhalten: {self.dropout}")

    # ------------------------------------------------------------------
    # Abgeleitete Größen
    # ------------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        """Dimension eines einzelnen Attention-Kopfes."""
        return self.hidden_size // self.num_heads

    def num_parameters(self) -> int:
        """Schätzt die Gesamtzahl der Parameter der Architektur (exakt)."""
        h, v, i = self.hidden_size, self.vocab_size, self.intermediate_size
        kv_dim = self.num_kv_heads * self.head_dim
        embed = v * h
        # Attention: q, k, v, o Projektionen (ohne Bias)
        attn = h * h + 2 * h * kv_dim + h * h
        # GeGLU-FFN: gate, up, down
        ffn = 2 * h * i + i * h
        # Zwei RMSNorm-Gewichte pro Block + finale Norm
        norms = self.num_layers * 2 * h + h
        lm_head = 0 if self.tie_word_embeddings else v * h
        return embed + self.num_layers * (attn + ffn) + norms + lm_head

    # ------------------------------------------------------------------
    # Serialisierung
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Konfiguration als einfaches Dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Erzeugt eine Konfiguration aus einem Dictionary.

        Unbekannte Schlüssel werden ignoriert, damit alte Checkpoints
        auch nach Erweiterungen der Konfiguration ladbar bleiben.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save_yaml(self, path: str | Path) -> None:
        """Schreibt die Konfiguration als YAML-Datei."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Lädt eine Konfiguration aus einer YAML-Datei."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Ungültige Modellkonfiguration in {path}")
        return cls.from_dict(data)

    def to_json(self) -> str:
        """Konfiguration als JSON-Zeichenkette (für die Datenbank)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "ModelConfig":
        """Erzeugt eine Konfiguration aus einer JSON-Zeichenkette."""
        return cls.from_dict(json.loads(data))
