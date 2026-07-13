"""Modellbewertung: Perplexität und abgeleitete Kennzahlen.

Perplexität ist das Standardmaß für Sprachmodelle: der exponenzierte
mittlere Cross-Entropy-Verlust. Niedrigere Werte bedeuten, dass das Modell
den Validierungstext besser vorhersagt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from backend.utils import autocast_context
from models.transformer import DecoderLM


@dataclass
class EvaluationResult:
    """Ergebnis einer Modellbewertung."""

    loss: float
    perplexity: float
    num_tokens: int
    num_batches: int

    def as_dict(self) -> dict:
        return {
            "loss": round(self.loss, 5),
            "perplexity": round(self.perplexity, 3),
            "num_tokens": self.num_tokens,
            "num_batches": self.num_batches,
        }


@torch.no_grad()
def evaluate_perplexity(
    model: DecoderLM,
    dataset: Dataset,
    device: torch.device | str = "cpu",
    batch_size: int = 8,
    max_batches: int | None = None,
    mixed_precision: bool = False,
) -> EvaluationResult:
    """Berechnet Verlust und Perplexität auf einem Datensatz.

    Args:
        model: Das zu bewertende Modell.
        dataset: Datensatz, der ``(x, y)``-Tupel liefert.
        device: Rechengerät.
        batch_size: Batchgröße für die Auswertung.
        max_batches: Optionale Obergrenze der ausgewerteten Batches.
        mixed_precision: Mixed Precision aktivieren (nur CUDA wirksam).

    Returns:
        Ein :class:`EvaluationResult`.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    total_loss = 0.0
    total_batches = 0
    total_tokens = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with autocast_context(device, mixed_precision):
            _, loss = model(x, targets=y)
        total_loss += loss.item()
        total_batches += 1
        total_tokens += int(y.numel())

    if total_batches == 0:
        raise ValueError("Der Datensatz lieferte keine Batches für die Bewertung.")

    mean_loss = total_loss / total_batches
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return EvaluationResult(
        loss=mean_loss,
        perplexity=perplexity,
        num_tokens=total_tokens,
        num_batches=total_batches,
    )
