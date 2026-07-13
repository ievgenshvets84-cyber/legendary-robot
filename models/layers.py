"""Grundbausteine des LLM-Forge-Transformers.

Alle Bausteine sind von Grund auf implementiert (keine Übernahme fremder
Gewichte) und orientieren sich architektonisch an modernen offenen
Decoder-Modellen wie Llama, Mistral und Gemma:

* :class:`RMSNorm` – Root-Mean-Square-Normalisierung ohne Bias.
* :class:`RotaryEmbedding` – Rotary Positional Embeddings (RoPE).
* :class:`GroupedQueryAttention` – Multi-Head- bzw. Grouped-Query-Attention
  mit kausaler Maske und KV-Cache für schnelle Inferenz.
* :class:`GeGLUFeedForward` – Feed-Forward-Netz mit GeGLU-Aktivierung.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer-Normalisierung.

    Normalisiert den Eingang über die letzte Dimension anhand des
    quadratischen Mittelwerts und skaliert mit einem lernbaren Gewicht.
    Im Gegensatz zu LayerNorm gibt es weder Mittelwertzentrierung noch Bias,
    was Rechenzeit spart und sich in LLMs bewährt hat.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Lernbarer Skalierungsfaktor, initialisiert mit 1 (Identität)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In float32 normalisieren, um numerische Stabilität unter AMP zu sichern
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Kodiert Positionen, indem Query- und Key-Vektoren paarweise in der
    komplexen Ebene rotiert werden. Die Rotationsfrequenzen werden einmal
    vorberechnet und als Puffer gehalten (kein lernbarer Parameter).
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim muss gerade sein, erhalten: {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # Inverse Frequenzen: theta^(-2i/d) für i = 0 .. d/2-1
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Winkel für jede Position vorberechnen: (max_seq_len, head_dim/2)
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)
        # Als nicht-persistente Puffer registrieren (nicht Teil des state_dict)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wendet die Rotation auf Query und Key an.

        Args:
            q: Query-Tensor der Form (batch, heads, seq, head_dim).
            k: Key-Tensor der Form (batch, kv_heads, seq, head_dim).
            position_offset: Startposition (für KV-Cache-Inferenz > 0).

        Returns:
            Rotierte (q, k)-Tensoren in der Eingangsform.
        """
        seq_len = q.shape[-2]
        end = position_offset + seq_len
        # Kontextgrenze nur im Eager-Modus prüfen; unter torch.jit-Tracing
        # (z. B. ONNX-Export) würde der Tensorvergleich eine Warnung auslösen.
        if not torch.jit.is_tracing() and end > self.max_seq_len:
            raise ValueError(
                f"Sequenzposition {end} überschreitet die maximale "
                f"Kontextlänge {self.max_seq_len}."
            )
        cos = self.cos_cached[position_offset:end].to(q.dtype)  # (seq, head_dim/2)
        sin = self.sin_cached[position_offset:end].to(q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotiert die Kopfdimension paarweise: (x1, x2) -> (x1·cos − x2·sin, x1·sin + x2·cos)."""
    # Paare bilden: (..., seq, head_dim) -> (..., seq, head_dim/2, 2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated_1 = x1 * cos - x2 * sin
    rotated_2 = x1 * sin + x2 * cos
    # Paare wieder verschachteln
    out = torch.stack((rotated_1, rotated_2), dim=-1)
    return out.flatten(-2)


class KVCache:
    """Einfacher Key/Value-Cache für autoregressive Inferenz.

    Hält pro Schicht die bisher berechneten Keys und Values, sodass bei der
    Generierung pro Schritt nur das neue Token verarbeitet werden muss.
    """

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        """Anzahl der bereits zwischengespeicherten Positionen."""
        return 0 if self.keys is None else self.keys.shape[-2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hängt neue Keys/Values an und gibt den Gesamtverlauf zurück."""
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = torch.cat([self.keys, k], dim=-2)
            self.values = torch.cat([self.values, v], dim=-2)
        return self.keys, self.values


class GroupedQueryAttention(nn.Module):
    """Kausale Self-Attention mit Grouped-Query-Unterstützung.

    Bei ``num_kv_heads == num_heads`` entspricht das klassischer Multi-Head
    Attention; bei weniger KV-Köpfen teilen sich mehrere Query-Köpfe ein
    Key/Value-Paar (GQA), was Speicher und KV-Cache-Größe reduziert.
    Die eigentliche Attention nutzt PyTorchs fused
    ``scaled_dot_product_attention`` (FlashAttention, wo verfügbar).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        kv_dim = self.num_kv_heads * self.head_dim
        # Projektionen ohne Bias (üblich in modernen LLMs)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Berechnet kausale Self-Attention.

        Args:
            x: Eingang der Form (batch, seq, hidden).
            rope: Rotary-Embedding-Modul (wird vom Modell geteilt).
            kv_cache: Optionaler KV-Cache für autoregressive Inferenz.

        Returns:
            Attention-Ausgabe der Form (batch, seq, hidden).
        """
        batch, seq_len, _ = x.shape

        # Projektionen und Aufteilung in Köpfe: (batch, heads, seq, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Positionskodierung relativ zum bisherigen Cache-Inhalt
        offset = kv_cache.seq_len if kv_cache is not None else 0
        q, k = rope(q, k, position_offset=offset)

        # Bei Inferenz: neue Keys/Values an den Cache anhängen
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # KV-Köpfe auf Query-Köpfe verteilen (GQA-Broadcast)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Kausalität: Ohne Cache (Training/Prefill/Export) wird stets kausal
        # maskiert. Mit Cache (inkrementelle Dekodierung) darf das neue Token
        # den gesamten bisherigen Kontext sehen; nur bei Mehrtoken-Eingaben
        # wird untereinander kausal maskiert. ``is_causal`` bleibt so ein reiner
        # Python-Bool und ist damit auch beim ONNX-Tracing gültig.
        if kv_cache is None:
            is_causal = True
        else:
            is_causal = q.shape[-2] > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Köpfe zusammenführen und zurückprojizieren
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class GeGLUFeedForward(nn.Module):
    """Feed-Forward-Netz mit GeGLU-Aktivierung.

    GeGLU kombiniert eine GELU-aktivierte "Gate"-Projektion multiplikativ
    mit einer linearen "Up"-Projektion:  FFN(x) = W_down( GELU(W_gate x) ⊙ W_up x ).
    Diese Gating-Variante verbessert die Qualität gegenüber klassischem
    ReLU/GELU-FFN bei gleicher Parameterzahl.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU mit tanh-Näherung (schnell und exportfreundlich)
        gated = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class DecoderBlock(nn.Module):
    """Ein Decoder-Block: Pre-Norm-Attention und Pre-Norm-FFN mit Residuen.

    Struktur (Pre-Normalisierung wie in Llama/Mistral):
        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = GeGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        # Residualverbindung 1: Attention
        x = x + self.attention(self.attn_norm(x), rope, kv_cache)
        # Residualverbindung 2: Feed-Forward
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


def init_weights(module: nn.Module) -> None:
    """Initialisiert Gewichte nach dem GPT-2-Schema (Normalverteilung, std=0.02).

    Die zusätzliche 1/sqrt(2·num_layers)-Skalierung der Residual-
    Ausgangsprojektionen übernimmt das Gesamtmodell, da dort die
    Parameternamen bekannt sind (siehe ``DecoderLM``).
    """
    std = 0.02
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
