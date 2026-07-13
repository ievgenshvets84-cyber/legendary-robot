"""LoRA (Low-Rank Adaptation) für parametereffizientes Fine-Tuning.

LoRA friert die Originalgewichte ein und lernt pro Zielschicht nur zwei
kleine Matrizen A (down) und B (up) mit Rang ``r``:

    W_eff = W + (alpha / r) · B @ A

Dadurch sinkt die Zahl der trainierbaren Parameter typischerweise auf
unter 1 % des Gesamtmodells. Die Adapter können nach dem Training in die
Basisgewichte eingerechnet ("gemergt") oder separat gespeichert werden.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from models.transformer import DecoderLM

# Standardmäßig adaptierte Projektionen (Attention-Q/V hat sich bewährt)
DEFAULT_TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj")


@dataclass
class LoRAConfig:
    """Konfiguration eines LoRA-Adapters.

    Attributes:
        r: Rang der Niedrigrang-Zerlegung (typisch 4–64).
        alpha: Skalierungsfaktor; wirksam ist ``alpha / r``.
        dropout: Dropout auf dem LoRA-Pfad während des Trainings.
        target_modules: Namensendungen der zu adaptierenden Linearschichten.
    """

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"LoRA-Rang r muss positiv sein, erhalten: {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA-alpha muss positiv sein, erhalten: {self.alpha}")

    def to_dict(self) -> dict:
        """Konfiguration als Dictionary (für Checkpoints)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LoRAConfig":
        """Erzeugt eine Konfiguration aus einem Dictionary."""
        return cls(**data)


class LoRALinear(nn.Module):
    """Lineare Schicht mit additivem LoRA-Pfad.

    Umhüllt eine bestehende (eingefrorene) ``nn.Linear``-Schicht und
    addiert den Niedrigrang-Korrekturterm zur Ausgabe.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.r = config.r
        self.scaling = config.alpha / config.r
        self.dropout = nn.Dropout(config.dropout)

        # LoRA-Matrizen: A wird zufällig, B mit Null initialisiert,
        # sodass der Adapter zu Beginn die Identität ist (kein Effekt).
        self.lora_a = nn.Parameter(torch.empty(config.r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.r))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

        # Basisgewichte einfrieren – nur der Adapter lernt
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Originalpfad + skalierter Niedrigrang-Pfad
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        """Rechnet den Adapter in die Basisgewichte ein und gibt diese zurück."""
        delta = (self.lora_b @ self.lora_a) * self.scaling
        self.base.weight += delta.to(self.base.weight.dtype)
        return self.base


def inject_lora(model: DecoderLM, config: LoRAConfig) -> int:
    """Ersetzt Ziel-Linearschichten des Modells durch LoRA-Varianten.

    Alle übrigen Parameter werden eingefroren, sodass anschließend nur
    die LoRA-Matrizen trainierbar sind.

    Args:
        model: Das zu adaptierende Modell (wird in-place verändert).
        config: LoRA-Konfiguration.

    Returns:
        Anzahl der ersetzten Schichten.

    Raises:
        ValueError: Wenn keine passende Zielschicht gefunden wurde.
    """
    # Zuerst alles einfrieren; LoRALinear registriert dann trainierbare Adapter
    for param in model.parameters():
        param.requires_grad_(False)

    replaced = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            is_target = isinstance(child, nn.Linear) and any(
                child_name.endswith(t) for t in config.target_modules
            )
            if is_target:
                setattr(module, child_name, LoRALinear(child, config))
                replaced += 1

    if replaced == 0:
        raise ValueError(
            f"Keine Zielschicht gefunden für target_modules={config.target_modules}. "
            "Gültige Namen sind z. B. q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj."
        )
    return replaced


def merge_lora(model: DecoderLM) -> int:
    """Rechnet alle LoRA-Adapter in die Basisgewichte ein.

    Ersetzt jede :class:`LoRALinear` wieder durch die zugrunde liegende
    ``nn.Linear`` mit aktualisierten Gewichten. Danach ist das Modell
    wieder ein reines Basismodell (z. B. für Export oder Quantisierung).

    Returns:
        Anzahl der zusammengeführten Adapter.
    """
    merged = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merge_into_base())
                merged += 1
    # Nach dem Merge alle Parameter wieder trainierbar machen
    for param in model.parameters():
        param.requires_grad_(True)
    return merged


def lora_state_dict(model: DecoderLM) -> dict[str, torch.Tensor]:
    """Extrahiert nur die LoRA-Parameter (für kompakte Adapter-Checkpoints)."""
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if "lora_a" in name or "lora_b" in name
    }
