"""Byte-Level Byte-Pair-Encoding-Tokenizer (BPE), von Grund auf implementiert.

Der Tokenizer arbeitet auf UTF-8-Bytes und kann daher jede Zeichenkette
verlustfrei kodieren (kein "unbekanntes Zeichen"). Das Training lernt
iterativ die häufigsten benachbarten Symbolpaare (Merges) bis zur
gewünschten Vokabulargröße – dasselbe Grundprinzip, das auch GPT-, Llama-
und Mistral-Tokenizer verwenden, hier jedoch komplett eigenständig.

Ablauf:
    1. Text mit einem Regex-Muster in "Worte" vortokenisieren (GPT-2-Stil).
    2. Jedes Wort als Folge von Bytes darstellen.
    3. Iterativ das global häufigste Byte-/Symbolpaar zu einem neuen Symbol
       verschmelzen, bis ``vocab_size`` erreicht ist.
    4. Merges und Vokabular als JSON speichern.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SpecialTokens:
    """Sondertokens des Tokenizers.

    Diese Tokens erhalten feste IDs am Anfang des Vokabulars und werden
    beim BPE-Training nicht zerlegt.
    """

    pad: str = "<|pad|>"
    bos: str = "<|bos|>"      # Begin of sequence
    eos: str = "<|eos|>"      # End of sequence
    unk: str = "<|unk|>"      # Reserve (Byte-Level braucht es eigentlich nicht)

    def as_list(self) -> list[str]:
        return [self.pad, self.bos, self.eos, self.unk]


# Vortokenisierungs-Regex im GPT-2-Stil: trennt Wörter, Zahlen, Satzzeichen
# und Leerzeichen sinnvoll voneinander.
#
# Die \p{L}-/\p{N}-Unicode-Klassen benötigen das optionale `regex`-Modul.
# Ist es nicht installiert, wird ein gleichwertiges Muster mit den
# Unicode-fähigen Klassen der Standardbibliothek (\w, \d) verwendet.
try:  # pragma: no cover - abhängig von optionaler Installation
    import regex as _regex

    _PRETOKEN_PATTERN = _regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""",
    )
except ImportError:
    _PRETOKEN_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+""",
        re.UNICODE,
    )


def _pretokenize(text: str) -> list[str]:
    """Zerlegt Text in Vortokens (Wörter/Zahlen/Satzzeichen/Leerräume)."""
    return _PRETOKEN_PATTERN.findall(text)


class BPETokenizer:
    """Trainierbarer Byte-Level-BPE-Tokenizer.

    Attributes:
        special_tokens: Die verwendeten Sondertokens.
        vocab: Abbildung Token-ID -> Byte-Folge.
        merges: Gelernte Merges als geordnete Liste von Byte-Paaren.
    """

    def __init__(self, special_tokens: SpecialTokens | None = None) -> None:
        self.special_tokens = special_tokens or SpecialTokens()
        # Token-ID -> bytes  und  bytes -> Token-ID
        self.vocab: dict[int, bytes] = {}
        self._token_to_id: dict[bytes, int] = {}
        # Rang eines Merges (kleiner = früher gelernt = höhere Priorität)
        self.merges: dict[tuple[int, int], int] = {}
        self._special_to_id: dict[str, int] = {}
        self._trained = False

    # ------------------------------------------------------------------
    # Eigenschaften
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Aktuelle Größe des Vokabulars."""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self._special_to_id[self.special_tokens.pad]

    @property
    def bos_id(self) -> int:
        return self._special_to_id[self.special_tokens.bos]

    @property
    def eos_id(self) -> int:
        return self._special_to_id[self.special_tokens.eos]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        progress_callback=None,
    ) -> None:
        """Trainiert den Tokenizer auf einem Textkorpus.

        Args:
            texts: Iterierbare Textzeilen/Dokumente.
            vocab_size: Angestrebte Vokabulargröße (inkl. Sondertokens und
                der 256 Basis-Bytes).
            min_frequency: Mindesthäufigkeit eines Paares, damit es
                verschmolzen wird.
            progress_callback: Optionale Funktion ``fn(step, total)`` zur
                Fortschrittsanzeige.

        Raises:
            ValueError: Wenn ``vocab_size`` zu klein für die Basis ist.
        """
        specials = self.special_tokens.as_list()
        base_size = len(specials) + 256
        if vocab_size < base_size:
            raise ValueError(
                f"vocab_size ({vocab_size}) muss mindestens {base_size} betragen "
                f"({len(specials)} Sondertokens + 256 Byte-Werte)."
            )

        # 1) Vokabular mit Sondertokens und den 256 Byte-Werten initialisieren
        self._init_base_vocab()

        # 2) Korpus vortokenisieren und Wortfrequenzen zählen
        word_freqs: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for piece in _pretokenize(text):
                # Jedes Vortoken als Tupel seiner Byte-Werte (verschoben um Offset)
                symbol_ids = tuple(b + self._byte_offset for b in piece.encode("utf-8"))
                if symbol_ids:
                    word_freqs[symbol_ids] += 1

        if not word_freqs:
            raise ValueError("Der Trainingskorpus enthält keinen verwertbaren Text.")

        # 3) Iterativ das häufigste Paar verschmelzen
        num_merges = vocab_size - base_size
        words: dict[tuple[int, ...], int] = dict(word_freqs)
        for step in range(num_merges):
            pair_counts = self._count_pairs(words)
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            # Neues Symbol anlegen: Verkettung der beiden Byte-Folgen
            new_id = len(self.vocab)
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_bytes
            self._token_to_id[merged_bytes] = new_id
            self.merges[best_pair] = len(self.merges)

            # Alle Wörter mit dem neuen Symbol aktualisieren
            words = self._apply_merge(words, best_pair, new_id)

            if progress_callback is not None:
                progress_callback(step + 1, num_merges)

        self._trained = True

    def _init_base_vocab(self) -> None:
        """Legt Sondertokens (IDs 0..k-1) und 256 Byte-Tokens an."""
        self.vocab.clear()
        self._token_to_id.clear()
        self.merges.clear()
        self._special_to_id.clear()

        idx = 0
        for tok in self.special_tokens.as_list():
            # Sondertokens als eigene Byte-Repräsentation (kollidiert nicht mit Rohbytes)
            key = tok.encode("utf-8")
            self.vocab[idx] = key
            self._token_to_id[key] = idx
            self._special_to_id[tok] = idx
            idx += 1

        # Offset, ab dem die 256 rohen Bytewerte liegen
        self._byte_offset = idx
        for b in range(256):
            self.vocab[idx] = bytes([b])
            # Achtung: Ein-Byte-Schlüssel könnten theoretisch mit Sondertoken-Bytes
            # kollidieren; da Sondertokens mehrbytig sind, ist das ausgeschlossen.
            self._token_to_id[bytes([b])] = idx
            idx += 1

    @staticmethod
    def _count_pairs(words: dict[tuple[int, ...], int]) -> Counter:
        """Zählt benachbarte Symbolpaare gewichtet mit der Wortfrequenz."""
        counts: Counter = Counter()
        for symbols, freq in words.items():
            for a, b in zip(symbols, symbols[1:]):
                counts[(a, b)] += freq
        return counts

    @staticmethod
    def _apply_merge(
        words: dict[tuple[int, ...], int],
        pair: tuple[int, int],
        new_id: int,
    ) -> dict[tuple[int, ...], int]:
        """Ersetzt jedes Vorkommen von ``pair`` durch ``new_id`` in allen Wörtern."""
        new_words: dict[tuple[int, ...], int] = {}
        first, second = pair
        for symbols, freq in words.items():
            merged: list[int] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == first and symbols[i + 1] == second:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            new_words[tuple(merged)] = new_words.get(tuple(merged), 0) + freq
        return new_words

    # ------------------------------------------------------------------
    # Kodierung / Dekodierung
    # ------------------------------------------------------------------
    def _encode_chunk(self, piece: str) -> list[int]:
        """Kodiert ein einzelnes Vortoken zu Token-IDs durch Anwenden der Merges."""
        symbols = [b + self._byte_offset for b in piece.encode("utf-8")]
        if len(symbols) < 2:
            return symbols

        # Greedy nach Merge-Rang: immer das Paar mit dem kleinsten Rang zuerst
        while len(symbols) >= 2:
            # Bestes (frühestes) Merge-Paar in der aktuellen Symbolfolge finden
            best_rank: int | None = None
            best_pos = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
            if best_rank is None:
                break
            # Paar an best_pos verschmelzen
            merged_bytes = self.vocab[symbols[best_pos]] + self.vocab[symbols[best_pos + 1]]
            new_id = self._token_to_id[merged_bytes]
            symbols = symbols[:best_pos] + [new_id] + symbols[best_pos + 2:]
        return symbols

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Kodiert eine Zeichenkette in eine Liste von Token-IDs.

        Args:
            text: Eingabetext.
            add_bos: Fügt am Anfang das BOS-Token ein.
            add_eos: Hängt am Ende das EOS-Token an.
        """
        self._ensure_trained()
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in _pretokenize(text):
            ids.extend(self._encode_chunk(piece))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Dekodiert Token-IDs zurück in eine Zeichenkette."""
        self._ensure_trained()
        special_ids = set(self._special_to_id.values())
        buffer = bytearray()
        for tid in ids:
            if tid not in self.vocab:
                continue
            if skip_special and tid in special_ids:
                continue
            buffer.extend(self.vocab[tid])
        # Ungültige Byte-Folgen tolerant dekodieren
        return buffer.decode("utf-8", errors="replace")

    def _ensure_trained(self) -> None:
        if not self._trained and not self.vocab:
            raise RuntimeError(
                "Der Tokenizer wurde weder trainiert noch geladen. "
                "Zuerst train() oder load() aufrufen."
            )

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Speichert Vokabular und Merges als JSON-Datei."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "special_tokens": {
                "pad": self.special_tokens.pad,
                "bos": self.special_tokens.bos,
                "eos": self.special_tokens.eos,
                "unk": self.special_tokens.unk,
            },
            "byte_offset": self._byte_offset,
            # Vokabular als {id: latin-1-dekodierte Bytes} – latin-1 ist bijektiv zu Bytes
            "vocab": {str(k): v.decode("latin-1") for k, v in self.vocab.items()},
            # Merges als Liste [a, b, rang], nach Rang sortiert
            "merges": [[a, b, rank] for (a, b), rank in sorted(self.merges.items(), key=lambda x: x[1])],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Lädt einen Tokenizer aus einer JSON-Datei."""
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        st = payload["special_tokens"]
        tok = cls(SpecialTokens(pad=st["pad"], bos=st["bos"], eos=st["eos"], unk=st["unk"]))
        tok._byte_offset = payload["byte_offset"]
        tok.vocab = {int(k): v.encode("latin-1") for k, v in payload["vocab"].items()}
        tok._token_to_id = {v: k for k, v in tok.vocab.items()}
        tok.merges = {(a, b): rank for a, b, rank in payload["merges"]}
        tok._special_to_id = {
            st["pad"]: 0, st["bos"]: 1, st["eos"]: 2, st["unk"]: 3,
        }
        tok._trained = True
        return tok
