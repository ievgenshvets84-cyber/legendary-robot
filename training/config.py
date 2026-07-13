"""Trainingskonfiguration von LLM-Forge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Alle Hyperparameter und Ablaufeinstellungen eines Trainingslaufs.

    Siehe ``config/training_default.yaml`` für dokumentierte Standardwerte.
    """

    # Optimierung
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Ablauf
    max_steps: int = 2000
    warmup_steps: int = 100
    batch_size: int = 8
    grad_accum_steps: int = 4
    seq_len: int = 1024

    # Auswertung / Checkpoints
    eval_interval: int = 200
    eval_batches: int = 20
    checkpoint_interval: int = 500
    keep_checkpoints: int = 3
    log_interval: int = 10

    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    compile: bool = False
    num_workers: int = 2
    seed: int = 42

    # Daten
    val_split: float = 0.01

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Prüft die Konfiguration auf sinnvolle Wertebereiche."""
        if self.max_steps <= 0:
            raise ValueError("max_steps muss positiv sein.")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps muss in [0, max_steps) liegen.")
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size und grad_accum_steps müssen positiv sein.")
        if not 0.0 <= self.val_split < 0.5:
            raise ValueError("val_split muss in [0, 0.5) liegen.")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("device muss 'auto', 'cuda' oder 'cpu' sein.")

    @property
    def effective_batch_size(self) -> int:
        """Effektive Batchgröße nach Gradientenakkumulation."""
        return self.batch_size * self.grad_accum_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data or {})
