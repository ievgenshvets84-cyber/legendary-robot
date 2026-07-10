"""Долговременная память и «обучение» через накопление опыта (RAG).

Это основной механизм обучения без переобучения весов: агент записывает
факты, выводы и результаты действий, а затем извлекает релевантные из них
семантическим поиском. Если модель эмбеддингов недоступна, используется
запасной лексический поиск, чтобы память работала всегда.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .llm import LLMClient


class Memory:
    def __init__(self, path: str | Path, llm: LLMClient, top_k: int = 5,
                 min_score: float = 0.15) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self._entries: list[dict[str, Any]] = self._load()

    # ── запись ───────────────────────────────────────────────────────────
    def add(self, text: str, tags: list[str] | None = None,
            kind: str = "note") -> None:
        text = text.strip()
        if not text:
            return
        entry: dict[str, Any] = {
            "text": text,
            "tags": tags or [],
            "kind": kind,
            "ts": time.time(),
            "embedding": self._safe_embed(text),
        }
        self._entries.append(entry)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── чтение ───────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not self._entries:
            return []
        k = top_k or self.top_k
        q_emb = self._safe_embed(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self._entries:
            if q_emb and entry.get("embedding"):
                score = _cosine(q_emb, entry["embedding"])
            else:
                score = _lexical(query, entry["text"])
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [e for s, e in scored[:k] if s >= self.min_score]

    def recall_block(self, query: str) -> str:
        """Готовый текстовый блок с релевантными воспоминаниями для промпта."""
        hits = self.search(query)
        if not hits:
            return ""
        lines = [f"- ({h['kind']}) {h['text']}" for h in hits]
        return "Релевантные воспоминания:\n" + "\n".join(lines)

    def all(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for e in self._entries:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        return {"total": len(self._entries), "by_kind": kinds,
                "embedded": sum(1 for e in self._entries if e.get("embedding"))}

    # ── служебное ────────────────────────────────────────────────────────
    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _safe_embed(self, text: str) -> list[float] | None:
        try:
            return self.llm.embed(text)
        except Exception:
            return None


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _lexical(query: str, text: str) -> float:
    """Запасная метрика: доля пересечения слов (Жаккар)."""
    q = set(_tokens(query))
    t = set(_tokens(text))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def _tokens(text: str) -> list[str]:
    return [w for w in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split() if len(w) > 2]
