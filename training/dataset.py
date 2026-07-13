"""Datensatzverwaltung und -aufbereitung für das Training.

Enthält:
* :class:`DatasetManager` – verwaltet Rohdateien im ``datasets``-Verzeichnis,
  bereinigt und tokenisiert sie zu einem gepackten Token-Strom auf der Platte.
* :class:`PackedTextDataset` – ein ``torch.utils.data.Dataset``, das den
  gepackten Token-Strom in überlappungsfreie Blöcke fester Länge schneidet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer.bpe_tokenizer import BPETokenizer
from training.data_cleaning import CleaningStats, TextCleaner


class DatasetManager:
    """Verwaltet Rohdatensätze und erzeugt tokenisierte Token-Ströme.

    Rohdateien (``.txt``/``.jsonl``) liegen unter ``<root>/raw``; das Ergebnis
    der Tokenisierung wird als ``.bin`` (uint16/uint32) unter
    ``<root>/processed`` abgelegt und ist per Memory-Mapping ladbar.
    """

    def __init__(self, root: str | Path = "datasets") -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Rohdaten
    # ------------------------------------------------------------------
    def list_raw(self) -> list[dict]:
        """Listet alle Rohdateien mit Größe (Bytes) auf."""
        files = []
        for path in sorted(self.raw_dir.glob("*")):
            if path.is_file():
                files.append({"name": path.name, "size_bytes": path.stat().st_size})
        return files

    def read_lines(self, path: Path) -> Iterator[str]:
        """Liest Zeilen aus einer ``.txt``- oder ``.jsonl``-Datei.

        Bei ``.jsonl`` wird das Feld ``text`` (oder der erste String-Wert)
        jedes Objekts extrahiert.
        """
        with open(path, encoding="utf-8", errors="replace") as fh:
            if path.suffix == ".jsonl":
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        text = obj.get("text") or next(
                            (v for v in obj.values() if isinstance(v, str)), None
                        )
                        if text:
                            yield text
                    elif isinstance(obj, str):
                        yield obj
            else:
                yield from fh

    def iter_corpus(self, cleaner: TextCleaner | None = None) -> Iterator[str]:
        """Iteriert (optional bereinigt) über alle Rohzeilen des Korpus."""
        cleaner = cleaner or TextCleaner()
        for path in sorted(self.raw_dir.glob("*")):
            if path.is_file() and path.suffix in (".txt", ".jsonl"):
                yield from cleaner.clean_stream(self.read_lines(path))

    # ------------------------------------------------------------------
    # Tokenisierung / Packen
    # ------------------------------------------------------------------
    def tokenize_corpus(
        self,
        tokenizer: BPETokenizer,
        output_name: str = "train",
        cleaner: TextCleaner | None = None,
        add_eos_between_docs: bool = True,
        progress_callback=None,
    ) -> dict:
        """Tokenisiert den gesamten Korpus und speichert ihn als ``.bin``.

        Jedes Dokument wird kodiert und – zur Trennung – mit einem
        EOS-Token abgeschlossen. Das Ergebnis ist ein flacher Token-Strom.

        Args:
            tokenizer: Trainierter Tokenizer.
            output_name: Basisname der Ausgabedatei (ohne Endung).
            cleaner: Optionaler Textbereiniger.
            add_eos_between_docs: EOS-Token zwischen Dokumenten einfügen.
            progress_callback: Optionale Funktion ``fn(num_tokens)``.

        Returns:
            Metadaten-Dictionary (Pfad, Tokenanzahl, dtype, Vokabulargröße).
        """
        cleaner = cleaner or TextCleaner()
        # dtype je nach Vokabulargröße wählen (spart Plattenplatz)
        dtype = np.uint16 if tokenizer.vocab_size <= 65_536 else np.uint32
        out_path = self.processed_dir / f"{output_name}.bin"

        total_tokens = 0
        buffer: list[int] = []
        with open(out_path, "wb") as out:
            for line in self.iter_corpus(cleaner):
                ids = tokenizer.encode(line, add_eos=add_eos_between_docs)
                buffer.extend(ids)
                total_tokens += len(ids)
                # In Blöcken auf die Platte schreiben, um RAM zu schonen
                if len(buffer) >= 1_000_000:
                    np.asarray(buffer, dtype=dtype).tofile(out)
                    buffer.clear()
                    if progress_callback is not None:
                        progress_callback(total_tokens)
            if buffer:
                np.asarray(buffer, dtype=dtype).tofile(out)
                if progress_callback is not None:
                    progress_callback(total_tokens)

        if total_tokens == 0:
            raise ValueError(
                "Nach der Bereinigung blieben keine Token übrig. "
                "Bitte Rohdaten und Bereinigungseinstellungen prüfen."
            )

        meta = {
            "path": str(out_path),
            "num_tokens": total_tokens,
            "dtype": np.dtype(dtype).name,
            "vocab_size": tokenizer.vocab_size,
        }
        meta_path = self.processed_dir / f"{output_name}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return meta

    def load_meta(self, output_name: str = "train") -> dict:
        """Lädt die Metadaten eines tokenisierten Datensatzes."""
        meta_path = self.processed_dir / f"{output_name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Kein tokenisierter Datensatz '{output_name}' gefunden. "
                "Zuerst tokenize_corpus() ausführen."
            )
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)


class PackedTextDataset(Dataset):
    """Dataset über einen gepackten Token-Strom für Sprachmodellierung.

    Der Token-Strom wird per ``numpy.memmap`` gelesen (kein vollständiges
    Laden in den RAM) und in Blöcke der Länge ``seq_len + 1`` geschnitten.
    Aus jedem Block ergeben sich Eingabe ``x`` (Token 0..n-1) und Ziel
    ``y`` (Token 1..n), also die klassische Next-Token-Vorhersage.
    """

    def __init__(self, bin_path: str | Path, seq_len: int, dtype: str = "uint16") -> None:
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Token-Datei nicht gefunden: {self.bin_path}")
        self._dtype = np.dtype(dtype)
        self.data = np.memmap(self.bin_path, dtype=self._dtype, mode="r")
        # Anzahl vollständiger, nicht überlappender Blöcke
        self.num_blocks = max(0, (len(self.data) - 1) // self.seq_len)
        if self.num_blocks == 0:
            raise ValueError(
                f"Token-Strom ist zu kurz ({len(self.data)} Token) für "
                f"seq_len={seq_len}. Mehr Daten bereitstellen oder seq_len senken."
            )

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        # +1 Token, damit x und y gegeneinander verschoben werden können
        chunk = self.data[start : start + self.seq_len + 1]
        # Nach int64 kopieren (Embedding erwartet Long-Indices)
        chunk = torch.from_numpy(chunk.astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
