"""Zentrale Geschäftslogik von LLM-Forge.

Der :class:`ForgeService` verbindet API-Schicht und Kernmodule und verwaltet
den globalen Zustand: registrierte Modelle, den (einzelnen) aktiven
Trainingslauf in einem Hintergrund-Thread, Tokenizer, Inferenz, Export,
Quantisierung und Evaluation. Die Klasse ist thread-sicher gegenüber dem
Trainings-Thread ausgelegt.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

from backend.database import Database
from backend.settings import ServerSettings
from backend.utils import gpu_stats, resolve_device, setup_logger
from evaluation.metrics import evaluate_perplexity
from inference.export import export_gguf, export_onnx
from inference.generator import GenerationConfig, TextGenerator
from inference.quantization import QuantConfig, quantize_model
from models.config import ModelConfig
from models.lora import LoRAConfig, inject_lora
from models.transformer import DecoderLM
from tokenizer.bpe_tokenizer import BPETokenizer
from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.dataset import DatasetManager, PackedTextDataset
from training.trainer import Trainer


class ForgeService:
    """Koordiniert alle Kernoperationen des Frameworks."""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.logger = setup_logger("forge", log_dir=settings.logs_dir)
        self.db = Database(settings.root / settings.database)
        self.datasets = DatasetManager(settings.root / settings.datasets_dir)

        # Zustand des aktiven Trainings
        self._trainer: Trainer | None = None
        self._train_thread: threading.Thread | None = None
        self._active_run_id: int | None = None
        self._active_model: str | None = None
        self._lock = threading.Lock()

    # ==================================================================
    # Modelle
    # ==================================================================
    def create_model(self, arch: dict[str, Any]) -> dict:
        """Registriert ein neues Modell aus Architekturparametern."""
        config = ModelConfig.from_dict(arch)  # validiert implizit
        if self.db.get_model(config.name) is not None:
            raise ValueError(f"Ein Modell mit dem Namen '{config.name}' existiert bereits.")
        self.db.create_model(config.name, config.to_dict())
        self.logger.info("Modell '%s' angelegt (%d Parameter geschätzt).",
                         config.name, config.num_parameters())
        return self._model_summary(config.name)

    def list_models(self) -> list[dict]:
        """Listet alle registrierten Modelle mit Parameterzahl."""
        result = []
        for entry in self.db.list_models():
            config = ModelConfig.from_dict(entry["config"])
            result.append(
                {
                    "name": entry["name"],
                    "config": entry["config"],
                    "tokenizer_path": entry["tokenizer_path"],
                    "num_parameters": config.num_parameters(),
                    "created_at": entry["created_at"],
                }
            )
        return result

    def get_model_config(self, name: str) -> ModelConfig:
        """Lädt die :class:`ModelConfig` eines registrierten Modells."""
        entry = self.db.get_model(name)
        if entry is None:
            raise KeyError(f"Modell '{name}' nicht gefunden.")
        return ModelConfig.from_dict(entry["config"])

    def update_architecture(self, name: str, updates: dict[str, Any]) -> dict:
        """Aktualisiert Architekturparameter eines noch untrainierten Modells."""
        config = self.get_model_config(name)
        data = config.to_dict()
        data.update({k: v for k, v in updates.items() if v is not None})
        new_config = ModelConfig.from_dict(data)  # revalidiert
        self.db.update_model(name, new_config.to_dict())
        self.logger.info("Architektur von '%s' aktualisiert.", name)
        return self._model_summary(name)

    def delete_model(self, name: str) -> None:
        """Entfernt ein Modell aus der Datenbank."""
        if self._active_model == name:
            raise RuntimeError("Modell wird gerade trainiert und kann nicht gelöscht werden.")
        self.db.delete_model(name)

    def _model_summary(self, name: str) -> dict:
        entry = self.db.get_model(name)
        assert entry is not None
        config = ModelConfig.from_dict(entry["config"])
        return {
            "name": name,
            "config": entry["config"],
            "tokenizer_path": entry["tokenizer_path"],
            "num_parameters": config.num_parameters(),
            "created_at": entry["created_at"],
        }

    # ==================================================================
    # Tokenizer
    # ==================================================================
    def _tokenizer_path(self, model_name: str) -> Path:
        return self.settings.root / self.settings.tokenizer_dir / f"{model_name}.json"

    def train_tokenizer(
        self,
        model_name: str,
        vocab_size: int,
        min_frequency: int = 2,
        dataset_files: list[str] | None = None,
    ) -> dict:
        """Trainiert einen BPE-Tokenizer auf dem Korpus und speichert ihn."""
        entry = self.db.get_model(model_name)
        if entry is None:
            raise KeyError(f"Modell '{model_name}' nicht gefunden.")

        # Korpuszeilen sammeln (optional auf ausgewählte Dateien beschränkt)
        def corpus():
            files = sorted(self.datasets.raw_dir.glob("*"))
            for path in files:
                if not path.is_file() or path.suffix not in (".txt", ".jsonl"):
                    continue
                if dataset_files and path.name not in dataset_files:
                    continue
                yield from self.datasets.read_lines(path)

        tokenizer = BPETokenizer()
        tokenizer.train(corpus(), vocab_size=vocab_size, min_frequency=min_frequency)

        path = self._tokenizer_path(model_name)
        tokenizer.save(path)

        # Vokabulargröße im Modell-Config nachziehen (Tokenizer bestimmt sie)
        config = ModelConfig.from_dict(entry["config"])
        config.vocab_size = tokenizer.vocab_size
        self.db.update_model(model_name, config.to_dict(), tokenizer_path=str(path))
        self.logger.info("Tokenizer für '%s' trainiert (vocab=%d).", model_name, tokenizer.vocab_size)
        return {"vocab_size": tokenizer.vocab_size, "path": str(path)}

    def load_tokenizer(self, model_name: str) -> BPETokenizer:
        """Lädt den zu einem Modell gehörenden Tokenizer."""
        path = self._tokenizer_path(model_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Für Modell '{model_name}' wurde noch kein Tokenizer trainiert."
            )
        return BPETokenizer.load(path)

    # ==================================================================
    # Datensätze
    # ==================================================================
    def list_datasets(self) -> dict:
        """Listet Roh- und verarbeitete Datensätze."""
        processed = []
        for meta_file in sorted(self.datasets.processed_dir.glob("*.meta.json")):
            import json
            with open(meta_file, encoding="utf-8") as fh:
                processed.append(json.load(fh))
        return {"raw": self.datasets.list_raw(), "processed": processed}

    def prepare_dataset(self, model_name: str, output_name: str = "train") -> dict:
        """Bereinigt und tokenisiert den Rohkorpus für ein Modell."""
        tokenizer = self.load_tokenizer(model_name)
        meta = self.datasets.tokenize_corpus(tokenizer, output_name=output_name)
        self.logger.info("Datensatz '%s' vorbereitet: %d Token.", output_name, meta["num_tokens"])
        return meta

    # ==================================================================
    # Training
    # ==================================================================
    def is_training(self) -> bool:
        """Gibt an, ob gerade ein Training läuft."""
        return self._train_thread is not None and self._train_thread.is_alive()

    def start_training(
        self,
        model_name: str,
        dataset_name: str,
        params: dict[str, Any],
        resume_from: str | None = None,
        use_lora: bool = False,
        lora_rank: int = 8,
    ) -> dict:
        """Startet einen Trainingslauf in einem Hintergrund-Thread."""
        with self._lock:
            if self.is_training():
                raise RuntimeError("Es läuft bereits ein Training. Bitte zuerst stoppen.")

            entry = self.db.get_model(model_name)
            if entry is None:
                raise KeyError(f"Modell '{model_name}' nicht gefunden.")

            model_config = ModelConfig.from_dict(entry["config"])
            train_config = TrainingConfig.from_dict(params)

            # Datensatz laden
            meta = self.datasets.load_meta(dataset_name)
            full_ds = PackedTextDataset(meta["path"], train_config.seq_len, dtype=meta["dtype"])
            train_ds, val_ds = self._split_dataset(full_ds, train_config.val_split)

            # Modell instanziieren und ggf. Checkpoint laden
            model = DecoderLM(model_config)
            ckpt_manager = CheckpointManager(
                self.settings.root / self.settings.checkpoints_dir, model_name
            )
            if resume_from:
                checkpoint = CheckpointManager.load(ckpt_manager.path_for(resume_from))
                model.load_state_dict(checkpoint["model_state"])
                self.logger.info("Fortsetzung ab Checkpoint '%s'.", resume_from)

            # Optionales LoRA-Fine-Tuning
            if use_lora:
                num = inject_lora(model, LoRAConfig(r=lora_rank))
                self.logger.info("LoRA aktiviert: %d Schichten adaptiert.", num)

            # Trainingslauf in der DB anlegen
            run_id = self.db.create_run(entry["id"], train_config.to_dict(), dataset_name)
            self._active_run_id = run_id
            self._active_model = model_name

            def on_event(event: dict) -> None:
                # Metriken persistieren
                if event.get("type") == "metrics":
                    self.db.add_metric(run_id, event)

            trainer = Trainer(
                model=model,
                model_config=model_config,
                train_config=train_config,
                train_dataset=train_ds,
                val_dataset=val_ds,
                checkpoint_manager=ckpt_manager,
                on_event=on_event,
            )
            self._trainer = trainer

            def run() -> None:
                try:
                    state = trainer.train()
                    self.db.finish_run(run_id, state.status, _finite(state.best_val_loss))
                except Exception as exc:  # pragma: no cover
                    self.logger.exception("Trainingslauf %d fehlgeschlagen.", run_id)
                    self.db.finish_run(run_id, "error", None)
                    trainer.state.status = "error"
                    trainer.state.message = str(exc)

            thread = threading.Thread(target=run, name=f"train-{run_id}", daemon=True)
            self._train_thread = thread
            thread.start()

        return {**trainer.state.snapshot(), "run_id": run_id}

    @staticmethod
    def _split_dataset(dataset: PackedTextDataset, val_split: float):
        """Teilt den Datensatz deterministisch in Trainings- und Validierungsteil."""
        from torch.utils.data import Subset

        n = len(dataset)
        n_val = max(1, int(n * val_split)) if val_split > 0 and n > 1 else 0
        if n_val == 0:
            return dataset, None
        indices = list(range(n))
        # Letzte Blöcke als Validierung (zeitlich getrennt)
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    def training_status(self) -> dict:
        """Aktueller Trainingszustand für das Webinterface."""
        if self._trainer is None:
            return {"status": "idle", "step": 0, "max_steps": 0, "progress": 0.0, "message": ""}
        snapshot = self._trainer.state.snapshot()
        snapshot["run_id"] = self._active_run_id
        return snapshot

    def pause_training(self) -> dict:
        if self._trainer is None:
            raise RuntimeError("Es läuft kein Training.")
        self._trainer.pause()
        if self._active_run_id is not None:
            self.db.update_run_status(self._active_run_id, "paused")
        return self.training_status()

    def resume_training(self) -> dict:
        if self._trainer is None:
            raise RuntimeError("Es läuft kein Training.")
        self._trainer.resume()
        if self._active_run_id is not None:
            self.db.update_run_status(self._active_run_id, "running")
        return self.training_status()

    def stop_training(self) -> dict:
        if self._trainer is None:
            raise RuntimeError("Es läuft kein Training.")
        self._trainer.stop()
        return self.training_status()

    def get_run_metrics(self, run_id: int) -> list[dict]:
        """Liefert die gespeicherte Metrikreihe eines Laufs."""
        return self.db.get_metrics(run_id)

    def list_runs(self, model_name: str | None = None) -> list[dict]:
        model_id = None
        if model_name:
            entry = self.db.get_model(model_name)
            if entry is None:
                raise KeyError(f"Modell '{model_name}' nicht gefunden.")
            model_id = entry["id"]
        return self.db.list_runs(model_id)

    # ==================================================================
    # Checkpoints
    # ==================================================================
    def list_checkpoints(self, model_name: str) -> list[dict]:
        mgr = CheckpointManager(self.settings.root / self.settings.checkpoints_dir, model_name)
        return mgr.list_checkpoints()

    def _load_model_from_checkpoint(
        self, model_name: str, checkpoint: str, device: torch.device
    ) -> DecoderLM:
        """Instanziiert ein Modell und lädt einen Checkpoint (oder frische Gewichte)."""
        mgr = CheckpointManager(self.settings.root / self.settings.checkpoints_dir, model_name)
        try:
            path = mgr.path_for(checkpoint)
            data = CheckpointManager.load(path, map_location=str(device))
            config = ModelConfig.from_dict(data["model_config"])
            model = DecoderLM(config)
            model.load_state_dict(data["model_state"])
        except FileNotFoundError:
            # Noch kein Checkpoint -> untrainiertes Modell (für schnelle Tests)
            config = self.get_model_config(model_name)
            model = DecoderLM(config)
            self.logger.warning("Kein Checkpoint für '%s'; nutze untrainierte Gewichte.", model_name)
        return model.to(device)

    # ==================================================================
    # Inferenz
    # ==================================================================
    def build_generator(self, model_name: str, checkpoint: str = "best") -> TextGenerator:
        """Baut einen Textgenerator aus Modell-Checkpoint und Tokenizer."""
        device = resolve_device("auto")
        tokenizer = self.load_tokenizer(model_name)
        model = self._load_model_from_checkpoint(model_name, checkpoint, device)
        return TextGenerator(model, tokenizer, device)

    def generate_text(self, req: dict) -> dict:
        """Erzeugt (nicht-gestreamten) Text für einen Prompt."""
        generator = self.build_generator(req["model_name"], req.get("checkpoint", "best"))
        cfg = GenerationConfig(
            max_new_tokens=req.get("max_new_tokens", 256),
            temperature=req.get("temperature", 0.8),
            top_k=req.get("top_k", 40),
            top_p=req.get("top_p", 0.95),
            repetition_penalty=req.get("repetition_penalty", 1.1),
        )
        text = generator.generate(req["prompt"], cfg)
        num_tokens = len(generator.tokenizer.encode(text))
        return {"text": text, "prompt": req["prompt"], "num_tokens": num_tokens}

    # ==================================================================
    # Evaluation
    # ==================================================================
    def evaluate(self, model_name: str, checkpoint: str, dataset_name: str, max_batches: int) -> dict:
        """Bewertet ein Modell (Perplexität) auf einem Datensatz."""
        device = resolve_device("auto")
        meta = self.datasets.load_meta(dataset_name)
        model = self._load_model_from_checkpoint(model_name, checkpoint, device)
        dataset = PackedTextDataset(meta["path"], model.config.max_seq_len, dtype=meta["dtype"])
        result = evaluate_perplexity(model, dataset, device=device, max_batches=max_batches)
        return result.as_dict()

    # ==================================================================
    # Quantisierung
    # ==================================================================
    def quantize(self, model_name: str, checkpoint: str, bits: int, group_size: int) -> dict:
        """Quantisiert ein Modell und speichert es unter ``exports``."""
        device = torch.device("cpu")
        model = self._load_model_from_checkpoint(model_name, checkpoint, device)
        before = sum(p.numel() * p.element_size() for p in model.parameters())
        quantize_model(model, QuantConfig(bits=bits, group_size=group_size))

        out_dir = self.settings.root / self.settings.exports_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_int{bits}.pt"
        torch.save({"model_state": model.state_dict(), "bits": bits, "group_size": group_size}, out_path)

        after = sum(
            b.numel() * b.element_size() for b in model.buffers()
        )
        return {
            "path": str(out_path),
            "bits": bits,
            "size_before_mb": round(before / 1024 ** 2, 2),
            "size_after_mb": round(after / 1024 ** 2, 2),
        }

    # ==================================================================
    # Export
    # ==================================================================
    def export(self, model_name: str, checkpoint: str, fmt: str) -> dict:
        """Exportiert ein Modell nach ONNX oder GGUF."""
        device = torch.device("cpu")
        model = self._load_model_from_checkpoint(model_name, checkpoint, device)
        out_dir = self.settings.root / self.settings.exports_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if fmt == "onnx":
            path = export_onnx(model, out_dir / f"{model_name}.onnx")
        elif fmt == "gguf":
            path = export_gguf(model, out_dir / f"{model_name}.gguf")
        else:
            raise ValueError(f"Unbekanntes Exportformat: {fmt}")

        return {"path": str(path), "format": fmt}

    # ==================================================================
    # System
    # ==================================================================
    def system_info(self) -> dict:
        """Liefert Systeminformationen (Gerät, GPU-Auslastung)."""
        device = resolve_device("auto")
        return {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "gpus": gpu_stats(),
            "training_active": self.is_training(),
        }

    def shutdown(self) -> None:
        """Beendet laufende Ressourcen sauber."""
        if self._trainer is not None:
            self._trainer.stop()
        self.db.close()


def _finite(value: float) -> float | None:
    import math
    return None if (value is None or math.isinf(value) or math.isnan(value)) else value
