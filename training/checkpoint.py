"""Checkpoint-System von LLM-Forge.

Ein Checkpoint bündelt Modellgewichte, Optimierer-Zustand, Trainingsschritt,
Modell- und Trainingskonfiguration sowie die zuletzt erreichten Metriken.
Der :class:`CheckpointManager` verwaltet mehrere Checkpoints eines Modells,
begrenzt ihre Anzahl und markiert den besten (nach Validierungsverlust).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from models.config import ModelConfig


class CheckpointManager:
    """Verwaltet Checkpoints eines einzelnen Modells in einem Verzeichnis.

    Layout::

        checkpoints/<model_name>/
            step_000500.pt
            step_001000.pt
            best.pt            (Kopie des besten Checkpoints)
            latest.pt          (Kopie des jüngsten Checkpoints)
    """

    def __init__(self, root: str | Path, model_name: str) -> None:
        self.dir = Path(root) / model_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Speichern
    # ------------------------------------------------------------------
    def save(
        self,
        *,
        model: torch.nn.Module,
        model_config: ModelConfig,
        step: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler_state: dict | None = None,
        training_config: dict | None = None,
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Schreibt einen Checkpoint und aktualisiert ``latest``/``best``.

        Returns:
            Pfad der geschriebenen Checkpoint-Datei.
        """
        payload: dict[str, Any] = {
            "format_version": 1,
            "model_name": self.model_name,
            "step": step,
            "model_config": model_config.to_dict(),
            "model_state": model.state_dict(),
            "training_config": training_config or {},
            "metrics": metrics or {},
            "extra": extra or {},
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if scheduler_state is not None:
            payload["scheduler_state"] = scheduler_state

        path = self.dir / f"step_{step:06d}.pt"
        torch.save(payload, path)

        # Bequeme Verweise als Kopien aktualisieren
        shutil.copyfile(path, self.dir / "latest.pt")
        if is_best:
            shutil.copyfile(path, self.dir / "best.pt")

        return path

    def prune(self, keep: int) -> list[Path]:
        """Behält nur die ``keep`` jüngsten schrittbasierten Checkpoints.

        ``best.pt`` und ``latest.pt`` sind eigenständige Kopien und bleiben
        immer erhalten.

        Returns:
            Liste der gelöschten Dateipfade.
        """
        step_files = sorted(
            self.dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        to_delete = step_files[:-keep] if keep > 0 else []
        for path in to_delete:
            path.unlink(missing_ok=True)
        return to_delete

    # ------------------------------------------------------------------
    # Laden / Auflisten
    # ------------------------------------------------------------------
    def list_checkpoints(self) -> list[dict]:
        """Listet vorhandene Checkpoints mit Schritt und Größe auf."""
        result = []
        for path in sorted(self.dir.glob("step_*.pt")):
            result.append(
                {
                    "name": path.name,
                    "step": int(path.stem.split("_")[1]),
                    "size_bytes": path.stat().st_size,
                }
            )
        return result

    def path_for(self, name: str = "latest") -> Path:
        """Löst einen Checkpoint-Namen (``latest``/``best``/Dateiname) zu einem Pfad auf."""
        if name in ("latest", "best"):
            candidate = self.dir / f"{name}.pt"
        else:
            candidate = self.dir / name
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {candidate}")
        return candidate

    @staticmethod
    def load(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
        """Lädt einen Checkpoint als Dictionary.

        ``weights_only=False`` ist nötig, da neben Tensoren auch die
        Konfigurations-Dictionaries gespeichert sind. Nur eigenen,
        vertrauenswürdigen Checkpoints laden.
        """
        return torch.load(path, map_location=map_location, weights_only=False)

    def export_config(self, path: str | Path) -> None:
        """Schreibt eine Zusammenfassung der Checkpoints als JSON."""
        info = {"model_name": self.model_name, "checkpoints": self.list_checkpoints()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
