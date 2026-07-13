"""Autoregressive Textgenerierung mit KV-Cache.

Implementiert gängige Sampling-Strategien (Temperatur, Top-k, Top-p/Nucleus,
Wiederholungsstrafe) sowie eine Streaming-Variante, die Token für Token
liefert – ideal für das Chatfenster des Webinterfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from models.transformer import DecoderLM
from tokenizer.bpe_tokenizer import BPETokenizer


@dataclass
class GenerationConfig:
    """Parameter der Textgenerierung.

    Attributes:
        max_new_tokens: Maximale Anzahl neu erzeugter Token.
        temperature: Skaliert die Logits; 0 erzwingt gieriges Dekodieren.
        top_k: Beschränkt auf die k wahrscheinlichsten Token (0 = aus).
        top_p: Nucleus-Sampling-Schwelle (1.0 = aus).
        repetition_penalty: Bestraft bereits erzeugte Token (1.0 = aus).
        seed: Optionaler Zufalls-Seed für reproduzierbare Ausgaben.
    """

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    seed: int | None = None

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens muss positiv sein.")
        if self.temperature < 0:
            raise ValueError("temperature darf nicht negativ sein.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p muss in (0, 1] liegen.")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty muss positiv sein.")


class TextGenerator:
    """Erzeugt Text mit einem trainierten :class:`DecoderLM`."""

    def __init__(
        self,
        model: DecoderLM,
        tokenizer: BPETokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_seq_len = model.config.max_seq_len

    # ------------------------------------------------------------------
    # Logit-Verarbeitung
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, generated: list[int], penalty: float
    ) -> torch.Tensor:
        """Reduziert die Wahrscheinlichkeit bereits erzeugter Token."""
        if penalty == 1.0 or not generated:
            return logits
        unique = torch.tensor(sorted(set(generated)), device=logits.device)
        selected = logits[unique]
        # Positive Logits werden geteilt, negative multipliziert (HF-Konvention)
        logits[unique] = torch.where(selected > 0, selected / penalty, selected * penalty)
        return logits

    @staticmethod
    def _filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """Wendet Top-k- und Top-p-Filterung an (setzt gefilterte Logits auf -inf)."""
        logits = logits.clone()
        # Top-k
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k).values[..., -1, None]
            logits[logits < threshold] = float("-inf")
        # Top-p (Nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            # Token oberhalb der kumulierten Schwelle entfernen, erstes stets behalten
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits[remove] = float("-inf")
            # Zurücksortieren in die Originalreihenfolge
            logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
        return logits

    def _next_token(
        self, logits: torch.Tensor, generated: list[int], cfg: GenerationConfig
    ) -> int:
        """Wählt das nächste Token gemäß Sampling-Konfiguration."""
        logits = logits.squeeze()  # (vocab,)
        logits = self._apply_repetition_penalty(logits, generated, cfg.repetition_penalty)

        # Gieriges Dekodieren bei Temperatur 0
        if cfg.temperature == 0:
            return int(torch.argmax(logits).item())

        logits = logits / cfg.temperature
        logits = self._filter_top_k_top_p(logits, cfg.top_k, cfg.top_p)
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------
    # Generierung
    # ------------------------------------------------------------------
    @torch.no_grad()
    def stream(self, prompt: str, cfg: GenerationConfig | None = None) -> Iterator[str]:
        """Erzeugt Text und liefert die dekodierten Teilstücke als Strom.

        Yields:
            Neu dekodierte Textfragmente (jeweils das zuletzt erzeugte Token).
        """
        cfg = cfg or GenerationConfig()
        cfg.validate()
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        # Prompt kodieren (mit BOS) und ggf. auf Kontextlänge kürzen
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = input_ids[-self.max_seq_len :]

        kv_caches = self.model.new_kv_caches()
        generated: list[int] = list(input_ids)

        # Prefill: gesamten Prompt in einem Durchlauf verarbeiten
        tokens = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(tokens, kv_caches=kv_caches)

        decoded_so_far = ""
        new_tokens: list[int] = []
        for _ in range(cfg.max_new_tokens):
            next_id = self._next_token(logits[:, -1, :], generated, cfg)

            if next_id == self.tokenizer.eos_id:
                break

            generated.append(next_id)
            new_tokens.append(next_id)

            # Inkrementelles Dekodieren: nur den neuen Textzuwachs liefern.
            # (Byte-BPE kann ein Zeichen über mehrere Token verteilen, daher
            #  wird der Gesamtstring verglichen statt Token einzeln dekodiert.)
            full = self.tokenizer.decode(new_tokens)
            delta = full[len(decoded_so_far):]
            if delta:
                decoded_so_far = full
                yield delta

            # Kontextgrenze beachten
            if kv_caches[0].seq_len >= self.max_seq_len:
                break

            # Nächsten Schritt nur mit dem neuen Token (KV-Cache hält den Rest)
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            logits, _ = self.model(next_tensor, kv_caches=kv_caches)

    @torch.no_grad()
    def generate(self, prompt: str, cfg: GenerationConfig | None = None) -> str:
        """Erzeugt Text und gibt die vollständige Fortsetzung als String zurück."""
        return "".join(self.stream(prompt, cfg))

    # ------------------------------------------------------------------
    # Konstruktion aus Checkpoint
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict,
        tokenizer: BPETokenizer,
        device: torch.device | str = "cpu",
    ) -> "TextGenerator":
        """Erzeugt einen Generator direkt aus einem geladenen Checkpoint-Dict."""
        from models.config import ModelConfig

        config = ModelConfig.from_dict(checkpoint["model_config"])
        model = DecoderLM(config)
        model.load_state_dict(checkpoint["model_state"])
        return cls(model, tokenizer, device)
