"""Gewichtsquantisierung (4-Bit und 8-Bit) für kompaktere Modelle.

Implementiert eine eigenständige, gruppenweise affine Quantisierung der
linearen Schichten. Ziel ist die Reduktion des Speicherbedarfs für
Inferenz und Export. Die Quantisierung erfolgt post-training (PTQ) und
ohne externe Abhängigkeiten.

Verfahren pro Gewichtszeile (bzw. Gruppe):
    q = round(w / scale) + zero_point,   scale = (max - min) / (2^bits - 1)
    w ≈ (q - zero_point) * scale

Eine :class:`QuantizedLinear`-Schicht speichert die gepackten Ganzzahlwerte
sowie Skalen/Nullpunkte und dequantisiert beim Vorwärtsdurchlauf.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class QuantConfig:
    """Konfiguration der Quantisierung.

    Attributes:
        bits: Bit-Breite (4 oder 8).
        group_size: Anzahl der Gewichte pro Skalengruppe entlang der
            Eingangsdimension (kleinere Gruppen = genauer, mehr Overhead).
    """

    bits: int = 8
    group_size: int = 64

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError("Nur 4-Bit- oder 8-Bit-Quantisierung wird unterstützt.")
        if self.group_size <= 0:
            raise ValueError("group_size muss positiv sein.")


def _quantize_tensor(
    weight: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantisiert eine 2D-Gewichtsmatrix gruppenweise (affin, asymmetrisch).

    Args:
        weight: Gewichtsmatrix der Form (out_features, in_features).
        bits: Bit-Breite.
        group_size: Gruppengröße entlang der Eingangsdimension.

    Returns:
        Tupel (quant, scale, zero_point):
            quant: int-Tensor gleicher Form wie ``weight``.
            scale/zero_point: Form (out_features, num_groups).
    """
    out_features, in_features = weight.shape
    # Auf ganze Gruppen auffüllen
    pad = (-in_features) % group_size
    if pad:
        weight = torch.cat([weight, weight.new_zeros(out_features, pad)], dim=1)
    padded_in = weight.shape[1]
    num_groups = padded_in // group_size

    # In (out, groups, group_size) umformen und je Gruppe Min/Max bestimmen
    grouped = weight.view(out_features, num_groups, group_size)
    max_val = grouped.amax(dim=2)
    min_val = grouped.amin(dim=2)

    qmax = (1 << bits) - 1
    scale = (max_val - min_val).clamp(min=1e-8) / qmax
    zero_point = torch.round(-min_val / scale)

    quant = torch.round(grouped / scale.unsqueeze(2) + zero_point.unsqueeze(2))
    quant = quant.clamp(0, qmax).to(torch.uint8)
    quant = quant.view(out_features, padded_in)[:, :in_features]
    return quant, scale, zero_point


def _dequantize_tensor(
    quant: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int,
    in_features: int,
) -> torch.Tensor:
    """Rekonstruiert die float-Gewichtsmatrix aus quantisierten Werten."""
    out_features = quant.shape[0]
    pad = (-in_features) % group_size
    if pad:
        quant = torch.cat([quant, quant.new_zeros(out_features, pad)], dim=1)
    num_groups = quant.shape[1] // group_size
    grouped = quant.view(out_features, num_groups, group_size).float()
    deq = (grouped - zero_point.unsqueeze(2)) * scale.unsqueeze(2)
    return deq.view(out_features, -1)[:, :in_features]


class QuantizedLinear(nn.Module):
    """Lineare Schicht mit quantisiert gespeicherten Gewichten.

    Die Gewichte werden komprimiert gehalten und beim Vorwärtsdurchlauf
    dequantisiert. Für 4 Bit werden zwei Werte pro Byte gepackt.
    """

    def __init__(self, base: nn.Linear, config: QuantConfig) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bits = config.bits
        self.group_size = config.group_size

        quant, scale, zero_point = _quantize_tensor(
            base.weight.data.float(), config.bits, config.group_size
        )
        # 4-Bit-Werte paarweise in ein Byte packen
        if config.bits == 4:
            quant = self._pack_4bit(quant)
        self.register_buffer("quant", quant)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        if base.bias is not None:
            self.register_buffer("bias", base.bias.data.clone())
        else:
            self.bias = None

    @staticmethod
    def _pack_4bit(quant: torch.Tensor) -> torch.Tensor:
        """Packt zwei 4-Bit-Werte in ein uint8-Byte."""
        out_features, in_features = quant.shape
        if in_features % 2 == 1:
            quant = torch.cat([quant, quant.new_zeros(out_features, 1)], dim=1)
        low = quant[:, 0::2]
        high = quant[:, 1::2]
        return (low | (high << 4)).to(torch.uint8)

    def _unpack_4bit(self) -> torch.Tensor:
        """Entpackt gepackte 4-Bit-Werte in eine (out, in)-Matrix."""
        low = self.quant & 0x0F
        high = (self.quant >> 4) & 0x0F
        # Verschachteln: low0, high0, low1, high1, ...
        out_features = self.quant.shape[0]
        interleaved = torch.stack([low, high], dim=2).view(out_features, -1)
        return interleaved[:, : self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quant = self._unpack_4bit() if self.bits == 4 else self.quant
        weight = _dequantize_tensor(
            quant, self.scale, self.zero_point, self.group_size, self.in_features
        ).to(x.dtype)
        return nn.functional.linear(x, weight, self.bias)

    def memory_bytes(self) -> int:
        """Speicherbedarf der quantisierten Puffer in Bytes."""
        total = self.quant.numel() * self.quant.element_size()
        total += self.scale.numel() * self.scale.element_size()
        total += self.zero_point.numel() * self.zero_point.element_size()
        return total


def quantize_model(model: nn.Module, config: QuantConfig) -> nn.Module:
    """Ersetzt alle ``nn.Linear``-Schichten durch :class:`QuantizedLinear`.

    Die LM-Head-/Embedding-Bindung wird dabei aufgelöst; das Modell ist
    nach der Quantisierung für Inferenz und Export gedacht, nicht für
    weiteres Training.

    Args:
        model: Das zu quantisierende Modell (wird in-place verändert).
        config: Quantisierungskonfiguration.

    Returns:
        Das quantisierte Modell (dieselbe Instanz).
    """
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, child_name, QuantizedLinear(child, config))
    return model
