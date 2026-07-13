"""Die Trainingsschleife von LLM-Forge.

Der :class:`Trainer` kapselt einen vollständigen (Fine-)Trainingslauf mit
AdamW, Warmup-Kosinus-Lernratenplan, Gradientenakkumulation, Mixed
Precision, Gradient Clipping, periodischer Validierung und Checkpointing.
Er ist so gestaltet, dass er sowohl per Skript als auch aus dem
FastAPI-Backend heraus (in einem Hintergrund-Thread) betrieben werden kann.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from backend.utils import autocast_context, resolve_device, set_seed, setup_logger
from models.config import ModelConfig
from models.transformer import DecoderLM
from training.checkpoint import CheckpointManager
from training.config import TrainingConfig


@dataclass
class TrainingState:
    """Beobachtbarer Zustand eines Trainingslaufs (für UI und Logging)."""

    status: str = "idle"          # idle | running | paused | completed | error | stopped
    step: int = 0
    max_steps: int = 0
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    learning_rate: float = 0.0
    tokens_per_sec: float = 0.0
    elapsed_sec: float = 0.0
    best_val_loss: float = float("inf")
    message: str = ""
    history: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        """Serialisiert den Zustand (ohne die volle Historie) für die API."""
        return {
            "status": self.status,
            "step": self.step,
            "max_steps": self.max_steps,
            "progress": (self.step / self.max_steps) if self.max_steps else 0.0,
            "train_loss": _safe(self.train_loss),
            "val_loss": _safe(self.val_loss),
            "learning_rate": self.learning_rate,
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "best_val_loss": _safe(self.best_val_loss),
            "message": self.message,
        }


def _safe(x: float) -> float | None:
    """Wandelt NaN/Inf in ``None`` für saubere JSON-Ausgabe."""
    return None if (x is None or math.isnan(x) or math.isinf(x)) else round(x, 5)


class Trainer:
    """Führt einen Trainings- oder Fine-Tuning-Lauf aus.

    Args:
        model: Das zu trainierende Modell.
        model_config: Architektur-Konfiguration (für Checkpoints).
        train_config: Trainingshyperparameter.
        train_dataset: Trainingsdatensatz (liefert ``(x, y)``-Tupel).
        val_dataset: Optionaler Validierungsdatensatz.
        checkpoint_manager: Verwaltung der Checkpoints.
        on_event: Optionaler Callback ``fn(event: dict)`` für Metriken/Status.
    """

    def __init__(
        self,
        model: DecoderLM,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.cfg = train_config
        self.model_config = model_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.ckpt = checkpoint_manager
        self.on_event = on_event
        self.logger = setup_logger("trainer")

        set_seed(self.cfg.seed)
        self.device = resolve_device(self.cfg.device)
        self.model = model.to(self.device)

        if self.cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        self.optimizer = self._build_optimizer()
        # GradScaler nur bei float16 auf CUDA nötig (bfloat16 braucht ihn nicht)
        use_scaler = (
            self.cfg.mixed_precision
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        )
        # torch.amp.GradScaler ist die aktuelle API (torch>=2.3); der Scaler ist
        # nur bei float16 auf CUDA aktiv und ansonsten ein No-Op.
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        self.state = TrainingState(status="idle", max_steps=self.cfg.max_steps)
        # Steuerflags für Pause/Stopp aus einem anderen Thread
        self._pause = threading.Event()
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Optimierer und Lernratenplan
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Baut AdamW mit Weight-Decay-Trennung (keine Decay auf Norm/Bias)."""
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # 1D-Parameter (Norm-Gewichte, Bias) sowie Embeddings ohne Decay
            if param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )

    def _lr_at(self, step: int) -> float:
        """Berechnet die Lernrate: lineares Warmup, dann Kosinus-Abkühlung."""
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
        # Kosinus-Abkühlung von peak auf min_lr
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(1.0, progress)
        min_lr = cfg.learning_rate * cfg.min_lr_ratio
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (cfg.learning_rate - min_lr) * coeff

    # ------------------------------------------------------------------
    # Steuerung (Thread-sicher)
    # ------------------------------------------------------------------
    def pause(self) -> None:
        """Pausiert die Schleife nach dem aktuellen Schritt."""
        self._pause.set()
        self.state.status = "paused"

    def resume(self) -> None:
        """Setzt einen pausierten Lauf fort."""
        self._pause.clear()
        if self.state.status == "paused":
            self.state.status = "running"

    def stop(self) -> None:
        """Fordert einen sauberen Abbruch der Schleife an."""
        self._stop.set()
        self._pause.clear()

    # ------------------------------------------------------------------
    # Datenlader
    # ------------------------------------------------------------------
    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

    @staticmethod
    def _infinite(loader: DataLoader):
        """Wiederholt einen DataLoader endlos (für schrittbasiertes Training)."""
        while True:
            yield from loader

    # ------------------------------------------------------------------
    # Trainingsschleife
    # ------------------------------------------------------------------
    def train(self) -> TrainingState:
        """Führt den Trainingslauf bis ``max_steps`` oder Stopp aus."""
        self.logger.info("Starte Training auf %s: %r", self.device, self.model_config.name)
        self.state.status = "running"
        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        data_iter = self._infinite(train_loader)

        start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for step in range(self.cfg.max_steps):
                # Pause-Handling: aktiv warten, bis fortgesetzt oder gestoppt wird
                while self._pause.is_set() and not self._stop.is_set():
                    time.sleep(0.1)
                if self._stop.is_set():
                    self.state.status = "stopped"
                    self.state.message = "Training vom Benutzer gestoppt."
                    break

                # Lernrate für diesen Schritt setzen
                lr = self._lr_at(step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                step_loss = self._train_step(data_iter)

                # Zustand aktualisieren
                self.state.step = step + 1
                self.state.train_loss = step_loss
                self.state.learning_rate = lr
                self.state.elapsed_sec = time.time() - start
                tokens = self.cfg.effective_batch_size * self.cfg.seq_len
                self.state.tokens_per_sec = tokens / max(1e-6, self._last_step_time)

                if (step + 1) % self.cfg.log_interval == 0:
                    self._emit_metrics(step + 1)

                # Validierung
                if self.val_dataset is not None and (step + 1) % self.cfg.eval_interval == 0:
                    val_loss = self.evaluate()
                    self.state.val_loss = val_loss
                    is_best = val_loss < self.state.best_val_loss
                    if is_best:
                        self.state.best_val_loss = val_loss
                    self._emit_metrics(step + 1, val_loss=val_loss)
                    if self.ckpt is not None and is_best:
                        self._save_checkpoint(step + 1, is_best=True)

                # Regelmäßiger Checkpoint
                if self.ckpt is not None and (step + 1) % self.cfg.checkpoint_interval == 0:
                    self._save_checkpoint(step + 1, is_best=False)

            else:
                self.state.status = "completed"
                self.state.message = "Training abgeschlossen."

            # Abschluss-Checkpoint
            if self.ckpt is not None and self.state.status in ("completed", "stopped"):
                self._save_checkpoint(self.state.step, is_best=False)

        except Exception as exc:  # pragma: no cover - defensiver Pfad
            self.state.status = "error"
            self.state.message = f"Fehler im Training: {exc}"
            self.logger.exception("Training fehlgeschlagen")
            self._emit_status()
            raise

        self._emit_status()
        self.logger.info("Training beendet mit Status '%s'.", self.state.status)
        return self.state

    def _train_step(self, data_iter) -> float:
        """Führt einen Optimierungsschritt inkl. Gradientenakkumulation aus."""
        step_start = time.time()
        self.model.train()
        total_loss = 0.0

        for _ in range(self.cfg.grad_accum_steps):
            x, y = next(data_iter)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with autocast_context(self.device, self.cfg.mixed_precision):
                _, loss = self.model(x, targets=y)
                # Verlust über Akkumulationsschritte mitteln
                loss = loss / self.cfg.grad_accum_steps

            self.scaler.scale(loss).backward()
            total_loss += loss.item()

        # Gradient Clipping (nach Unscale, damit die Norm korrekt ist)
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self._last_step_time = time.time() - step_start
        return total_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """Berechnet den mittleren Validierungsverlust über ``eval_batches``."""
        if self.val_dataset is None:
            return float("nan")
        self.model.eval()
        loader = self._make_loader(self.val_dataset, shuffle=False)
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= self.cfg.eval_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with autocast_context(self.device, self.cfg.mixed_precision):
                _, loss = self.model(x, targets=y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    # ------------------------------------------------------------------
    # Ereignisse / Checkpoints
    # ------------------------------------------------------------------
    def _emit_metrics(self, step: int, val_loss: float | None = None) -> None:
        record = {
            "type": "metrics",
            "step": step,
            "train_loss": _safe(self.state.train_loss),
            "val_loss": _safe(val_loss if val_loss is not None else self.state.val_loss),
            "learning_rate": self.state.learning_rate,
            "tokens_per_sec": round(self.state.tokens_per_sec, 1),
            # Perplexität als anschauliches Qualitätsmaß
            "perplexity": _safe(math.exp(self.state.train_loss)) if not math.isnan(self.state.train_loss) else None,
        }
        self.state.history.append(record)
        self.logger.info(
            "step %d | loss %.4f | lr %.2e | %.0f tok/s",
            step, self.state.train_loss, self.state.learning_rate, self.state.tokens_per_sec,
        )
        if self.on_event is not None:
            self.on_event(record)

    def _emit_status(self) -> None:
        if self.on_event is not None:
            self.on_event({"type": "status", **self.state.snapshot()})

    def _save_checkpoint(self, step: int, is_best: bool) -> None:
        assert self.ckpt is not None
        # Bei torch.compile das originale Modul speichern
        model = getattr(self.model, "_orig_mod", self.model)
        self.ckpt.save(
            model=model,
            model_config=self.model_config,
            step=step,
            optimizer=self.optimizer,
            training_config=self.cfg.to_dict(),
            metrics={
                "train_loss": _safe(self.state.train_loss) or 0.0,
                "val_loss": _safe(self.state.val_loss) or 0.0,
            },
            is_best=is_best,
        )
        self.ckpt.prune(self.cfg.keep_checkpoints)
        self.logger.info("Checkpoint gespeichert (step %d, best=%s).", step, is_best)

    # Wird in _train_step gesetzt; hier vordeklariert für Klarheit
    _last_step_time: float = 1e-6
