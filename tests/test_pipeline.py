"""Integrationstest: Dataset-Packing, kurzer Trainingslauf, Generierung und Export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.metrics import evaluate_perplexity
from inference.export import export_gguf, export_onnx
from inference.generator import GenerationConfig, TextGenerator
from models.config import ModelConfig
from models.transformer import DecoderLM
from tokenizer.bpe_tokenizer import BPETokenizer
from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.dataset import DatasetManager, PackedTextDataset
from training.trainer import Trainer

# Bewusst variierte Zeilen: der Standard-Cleaner entfernt exakte Duplikate,
# daher wären identische Zeilen für einen Trainingskorpus ungeeignet.
CORPUS = [
    f"das modell lernt sprache und wörter in beispiel nummer {i} sehr gut"
    for i in range(200)
]


def tiny_config() -> ModelConfig:
    return ModelConfig(
        name="pipe", vocab_size=512, hidden_size=32, num_layers=2,
        num_heads=4, num_kv_heads=2, intermediate_size=64, max_seq_len=32,
    )


def make_dataset(tmp: Path) -> tuple[BPETokenizer, PackedTextDataset]:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300, min_frequency=1)

    manager = DatasetManager(tmp)
    # Rohdatei schreiben
    (manager.raw_dir / "c.txt").write_text("\n".join(CORPUS), encoding="utf-8")
    meta = manager.tokenize_corpus(tok, output_name="train")
    ds = PackedTextDataset(meta["path"], seq_len=16, dtype=meta["dtype"])
    return tok, ds


def test_packed_dataset_shapes(tmp_path):
    _, ds = make_dataset(tmp_path)
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    # y ist x um eins verschoben
    assert torch.equal(x[1:], y[:-1])


def test_short_training_reduces_loss(tmp_path):
    tok, ds = make_dataset(tmp_path)
    cfg = tiny_config()
    cfg.vocab_size = tok.vocab_size
    model = DecoderLM(cfg)

    train_cfg = TrainingConfig(
        max_steps=30, warmup_steps=5, batch_size=4, grad_accum_steps=1,
        seq_len=16, eval_interval=1000, checkpoint_interval=1000,
        log_interval=5, device="cpu", mixed_precision=False, num_workers=0,
        val_split=0.0,
    )
    losses = []
    trainer = Trainer(
        model, cfg, train_cfg, ds, val_dataset=None,
        checkpoint_manager=CheckpointManager(tmp_path / "ckpt", cfg.name),
        on_event=lambda e: losses.append(e["train_loss"]) if e.get("type") == "metrics" else None,
    )
    state = trainer.train()
    assert state.status == "completed"
    # Der Verlust am Ende sollte niedriger sein als am Anfang
    valid = [l for l in losses if l is not None]
    assert len(valid) >= 2
    assert valid[-1] < valid[0]


def test_checkpoint_save_and_reload(tmp_path):
    tok, ds = make_dataset(tmp_path)
    cfg = tiny_config()
    cfg.vocab_size = tok.vocab_size
    model = DecoderLM(cfg)
    mgr = CheckpointManager(tmp_path / "ckpt", cfg.name)
    path = mgr.save(model=model, model_config=cfg, step=1)
    assert path.exists()

    data = CheckpointManager.load(path)
    restored = DecoderLM(ModelConfig.from_dict(data["model_config"]))
    restored.load_state_dict(data["model_state"])
    # Gewichte müssen identisch sein
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)


def test_generation_runs(tmp_path):
    tok, _ = make_dataset(tmp_path)
    cfg = tiny_config()
    cfg.vocab_size = tok.vocab_size
    model = DecoderLM(cfg)
    gen = TextGenerator(model, tok, device="cpu")
    out = gen.generate("das modell", GenerationConfig(max_new_tokens=10, temperature=0.8, seed=0))
    assert isinstance(out, str)


def test_evaluation_produces_perplexity(tmp_path):
    tok, ds = make_dataset(tmp_path)
    cfg = tiny_config()
    cfg.vocab_size = tok.vocab_size
    model = DecoderLM(cfg)
    result = evaluate_perplexity(model, ds, device="cpu", batch_size=4, max_batches=3)
    assert result.perplexity > 0
    assert result.num_batches > 0


def test_gguf_export(tmp_path):
    tok, _ = make_dataset(tmp_path)
    cfg = tiny_config()
    cfg.vocab_size = tok.vocab_size
    model = DecoderLM(cfg)
    path = export_gguf(model, tmp_path / "model.gguf")
    assert path.exists()
    # GGUF-Magic prüfen ("GGUF")
    with open(path, "rb") as fh:
        magic = fh.read(4)
    assert magic == b"GGUF"


def test_onnx_export(tmp_path):
    pytest.importorskip("onnx")
    tok, _ = make_dataset(tmp_path)
    cfg = tiny_config()
    cfg.vocab_size = tok.vocab_size
    model = DecoderLM(cfg)
    path = export_onnx(model, tmp_path / "model.onnx", seq_len=8)
    assert path.exists()
