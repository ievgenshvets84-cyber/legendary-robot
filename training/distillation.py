"""Optionale Wissensdistillation von einem offenen Referenzmodell.

Dieses Modul ist **optional** und dient ausschließlich dazu, ein
eigenständiges LLM-Forge-Modell durch "weiche" Zielverteilungen eines
offenen Lehrermodells (z. B. Gemma, Llama, Mistral über Hugging Face)
schneller lernen zu lassen. Es werden **keine Gewichte** des Lehrermodells
übernommen – nur dessen Ausgabewahrscheinlichkeiten fließen als
zusätzliches Trainingssignal ein.

Die Abhängigkeit ``transformers`` ist optional; ist sie nicht installiert,
bleibt der Rest des Projekts uneingeschränkt nutzbar.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DistillationConfig:
    """Konfiguration der Distillation.

    Attributes:
        teacher_model: Name oder Pfad des offenen Lehrermodells (nur als
            Referenzsignal, keine Gewichtsübernahme).
        temperature: Temperatur zur Weichzeichnung der Verteilungen.
        alpha: Gewichtung zwischen hartem (Daten-)Verlust und weichem
            (Lehrer-)Verlust: ``loss = (1-alpha)*hard + alpha*soft``.
    """

    teacher_model: str = "google/gemma-2-2b"
    temperature: float = 2.0
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha muss in [0, 1] liegen.")
        if self.temperature <= 0:
            raise ValueError("temperature muss positiv sein.")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    config: DistillationConfig,
) -> torch.Tensor:
    """Kombiniert harten Cross-Entropy- und weichen KL-Distillationsverlust.

    Args:
        student_logits: Logits des eigenen Modells (batch, seq, vocab).
        teacher_logits: Logits des Lehrermodells (batch, seq, vocab).
            Bei abweichender Vokabulargröße wird auf das Minimum beschnitten.
        targets: Ziel-Token-IDs (batch, seq); ``-100`` wird ignoriert.
        config: Distillationskonfiguration.

    Returns:
        Skalarer Gesamtverlust.
    """
    # Harter Verlust gegen die echten Ziele
    hard = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )

    # Weicher Verlust: KL-Divergenz zwischen weichgezeichneten Verteilungen.
    # Bei unterschiedlichen Vokabularen auf die gemeinsame Größe beschränken.
    vocab = min(student_logits.size(-1), teacher_logits.size(-1))
    t = config.temperature
    s_log_probs = F.log_softmax(student_logits[..., :vocab] / t, dim=-1)
    t_probs = F.softmax(teacher_logits[..., :vocab] / t, dim=-1)
    soft = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (t * t)

    return (1.0 - config.alpha) * hard + config.alpha * soft


def load_teacher(config: DistillationConfig, device: str = "cpu"):
    """Lädt ein offenes Lehrermodell über ``transformers`` (optional).

    Raises:
        ImportError: Wenn ``transformers`` nicht installiert ist.
    """
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except ImportError as exc:  # pragma: no cover - optionale Abhängigkeit
        raise ImportError(
            "Für Distillation wird 'transformers' benötigt. "
            "Installation: pip install .[distill]"
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(config.teacher_model)
    model.eval().to(device)
    # Lehrer wird nicht trainiert – Gradienten deaktivieren
    for param in model.parameters():
        param.requires_grad_(False)
    return model
