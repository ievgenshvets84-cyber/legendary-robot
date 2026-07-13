"""Das vollständige Decoder-Transformer-Sprachmodell von LLM-Forge.

Architektur (eigenständig implementiert, keine fremden Gewichte):

    Token-Embedding
        → N × DecoderBlock (RMSNorm → GQA-Attention → Residuum,
                            RMSNorm → GeGLU-FFN   → Residuum)
        → finale RMSNorm
        → LM-Head (optional mit dem Embedding gebunden)

Positionsinformation wird ausschließlich über Rotary Positional
Embeddings (RoPE) in der Attention eingebracht.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig
from models.layers import DecoderBlock, KVCache, RMSNorm, RotaryEmbedding, init_weights


class DecoderLM(nn.Module):
    """Autoregressives Decoder-Sprachmodell.

    Args:
        config: Architektur-Konfiguration (siehe :class:`ModelConfig`).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token-Embedding: bildet Token-IDs auf den Residual-Strom ab
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Gemeinsames RoPE-Modul für alle Schichten (nur Puffer, keine Parameter)
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        # Stapel der Decoder-Blöcke
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))

        # Finale Normalisierung vor dem LM-Head
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Ausgabeprojektion auf das Vokabular
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # Gewichtsbindung: LM-Head teilt sich die Matrix mit dem Embedding
            self.lm_head.weight = self.embed_tokens.weight

        self._init_all_weights()

    # ------------------------------------------------------------------
    # Initialisierung
    # ------------------------------------------------------------------
    def _init_all_weights(self) -> None:
        """Initialisiert alle Gewichte des Modells.

        Standard: Normalverteilung mit std=0.02. Die Ausgangsprojektionen
        der Residualpfade (Attention ``o_proj`` und FFN ``down_proj``)
        werden mit 1/sqrt(2·num_layers) herunterskaliert, damit die
        Aktivierungsvarianz über die Tiefe konstant bleibt (GPT-2-Schema).
        """
        self.apply(init_weights)
        residual_scale = 1.0 / math.sqrt(2 * self.config.num_layers)
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 * residual_scale)

    # ------------------------------------------------------------------
    # Vorwärtsdurchlauf
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Berechnet Logits und optional den Trainingsverlust.

        Args:
            input_ids: Token-IDs der Form (batch, seq).
            targets: Optionale Zielausgaben (batch, seq) für den
                Cross-Entropy-Verlust; Positionen mit ``-100`` werden
                ignoriert (Padding/Prompt-Maskierung).
            kv_caches: Optionale KV-Caches (eine Instanz pro Schicht)
                für autoregressive Inferenz.

        Returns:
            Tupel aus Logits (batch, seq, vocab) und Verlust (Skalar
            oder ``None``, wenn keine Ziele übergeben wurden).
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids muss 2-dimensional sein, erhalten: {tuple(input_ids.shape)}")
        if kv_caches is not None and len(kv_caches) != len(self.layers):
            raise ValueError(
                f"Es werden {len(self.layers)} KV-Caches benötigt, "
                f"erhalten: {len(kv_caches)}"
            )

        # Embedding-Lookup und Dropout
        x = self.embed_dropout(self.embed_tokens(input_ids))

        # Decoder-Stapel
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, self.rope, cache)

        x = self.final_norm(x)

        if targets is not None:
            # Training: Logits über die volle Sequenz und Verlust berechnen
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return logits, loss

        # Inferenz: Nur die letzte Position wird für das nächste Token benötigt –
        # das spart bei langen Prompts erheblich Rechenzeit und Speicher.
        logits = self.lm_head(x[:, -1:, :])
        return logits, None

    # ------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------
    def new_kv_caches(self) -> list[KVCache]:
        """Erzeugt leere KV-Caches für eine neue Generierungssitzung."""
        return [KVCache() for _ in self.layers]

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Zählt die (trainierbaren) Parameter des Modells."""
        params = (
            p for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
        # Bei gebundenen Gewichten zählt PyTorch die Matrix nur einmal
        return sum(p.numel() for p in params)

    @torch.no_grad()
    def estimate_memory_mb(self) -> float:
        """Schätzt den Speicherbedarf der Gewichte in Megabyte (float32)."""
        return self.num_parameters() * 4 / (1024 ** 2)

    def __repr__(self) -> str:  # pragma: no cover - nur Anzeige
        cfg = self.config
        return (
            f"DecoderLM(name={cfg.name!r}, layers={cfg.num_layers}, "
            f"hidden={cfg.hidden_size}, heads={cfg.num_heads}/{cfg.num_kv_heads}, "
            f"ctx={cfg.max_seq_len}, params={self.num_parameters():,})"
        )
