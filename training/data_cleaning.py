"""Datenbereinigung für Trainingskorpora.

Bietet konfigurierbare Filter- und Normalisierungsschritte, die typische
Probleme in Rohtextdaten beheben: Steuerzeichen, Unicode-Inkonsistenzen,
Duplikate, zu kurze/zu lange Zeilen und übermäßige Wiederholungen.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass
class CleaningStats:
    """Statistik über einen Bereinigungslauf."""

    total: int = 0
    kept: int = 0
    removed_empty: int = 0
    removed_short: int = 0
    removed_long: int = 0
    removed_duplicate: int = 0
    removed_repetitive: int = 0

    @property
    def removed(self) -> int:
        return self.total - self.kept

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "kept": self.kept,
            "removed": self.removed,
            "removed_empty": self.removed_empty,
            "removed_short": self.removed_short,
            "removed_long": self.removed_long,
            "removed_duplicate": self.removed_duplicate,
            "removed_repetitive": self.removed_repetitive,
        }


@dataclass
class TextCleaner:
    """Konfigurierbarer Textbereiniger.

    Attributes:
        normalize_unicode: Wendet NFC-Unicode-Normalisierung an.
        strip_control_chars: Entfernt nicht druckbare Steuerzeichen.
        collapse_whitespace: Reduziert Folgen von Leerraum auf ein Leerzeichen.
        min_chars: Mindestlänge einer Zeile (kürzere werden verworfen).
        max_chars: Maximallänge einer Zeile (längere werden verworfen).
        dedupe: Entfernt exakte Duplikate (über Hash).
        max_repetition_ratio: Höchstanteil des häufigsten Zeichens; darüber
            gilt die Zeile als "repetitiv" und wird verworfen (z. B. "aaaa...").
    """

    normalize_unicode: bool = True
    strip_control_chars: bool = True
    collapse_whitespace: bool = True
    min_chars: int = 1
    max_chars: int = 100_000
    dedupe: bool = True
    max_repetition_ratio: float = 0.5

    # Muster für Steuerzeichen (ohne Tab/Zeilenumbruch)
    _control_re: re.Pattern = field(
        default_factory=lambda: re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
        repr=False,
    )
    _ws_re: re.Pattern = field(default_factory=lambda: re.compile(r"\s+"), repr=False)

    def clean_line(self, line: str) -> str:
        """Normalisiert eine einzelne Zeile (ohne Filterentscheidung)."""
        if self.normalize_unicode:
            line = unicodedata.normalize("NFC", line)
        if self.strip_control_chars:
            line = self._control_re.sub("", line)
        if self.collapse_whitespace:
            line = self._ws_re.sub(" ", line)
        return line.strip()

    def _is_repetitive(self, line: str) -> bool:
        """Erkennt Zeilen, die von einem einzelnen Zeichen dominiert werden."""
        if len(line) < 10:
            return False
        most_common = max(line.count(c) for c in set(line))
        return most_common / len(line) > self.max_repetition_ratio

    def clean(self, lines: Iterable[str]) -> tuple[list[str], CleaningStats]:
        """Bereinigt und filtert eine Sammlung von Zeilen.

        Returns:
            Tupel aus bereinigter Zeilenliste und :class:`CleaningStats`.
        """
        stats = CleaningStats()
        seen: set[str] = set()
        result: list[str] = []

        for raw in lines:
            stats.total += 1
            line = self.clean_line(raw)

            if not line:
                stats.removed_empty += 1
                continue
            if len(line) < self.min_chars:
                stats.removed_short += 1
                continue
            if len(line) > self.max_chars:
                stats.removed_long += 1
                continue
            if self._is_repetitive(line):
                stats.removed_repetitive += 1
                continue
            if self.dedupe:
                digest = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
                if digest in seen:
                    stats.removed_duplicate += 1
                    continue
                seen.add(digest)

            result.append(line)
            stats.kept += 1

        return result, stats

    def clean_stream(self, lines: Iterable[str]) -> Iterator[str]:
        """Wie :meth:`clean`, aber als Generator (speicherschonend, ohne Dedup-Rückgabe)."""
        seen: set[str] = set()
        for raw in lines:
            line = self.clean_line(raw)
            if not line or len(line) < self.min_chars or len(line) > self.max_chars:
                continue
            if self._is_repetitive(line):
                continue
            if self.dedupe:
                digest = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
                if digest in seen:
                    continue
                seen.add(digest)
            yield line
