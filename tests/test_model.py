"""Tests für Modellkonfiguration und Vorwärtsdurchlauf des Transformers."""

from __future__ import annotations

import pytest
import torch

from models.config import ModelConfig
from models.layers import KVCache, RMSNorm, RotaryEmbedding
from models.transformer import DecoderLM


def tiny_config(**overrides) -> ModelConfig:
    base = dict(
        name="test",
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=64,
    )
    base.update(overrides)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
def test_config_validation_head_divisibility():
    with pytest.raises(ValueError):
        tiny_config(hidden_size=65, num_heads=4)


def test_config_gqa_divisibility():
    with pytest.raises(ValueError):
        tiny_config(num_heads=4, num_kv_heads=3)


def test_config_odd_head_dim():
    # head_dim muss gerade sein (RoPE): hidden_size=6, num_heads=3 -> head_dim=2 ist gerade,
    # daher explizit eine ungerade Kopfdimension provozieren (head_dim=1).
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=3, num_heads=3, num_kv_heads=1,
                    intermediate_size=8, vocab_size=32, max_seq_len=8)


def test_config_roundtrip_json():
    cfg = tiny_config()
    restored = ModelConfig.from_json(cfg.to_json())
    assert restored.to_dict() == cfg.to_dict()


def test_num_parameters_matches_model():
    cfg = tiny_config()
    model = DecoderLM(cfg)
    # Bei gebundenen Embeddings zählt die Matrix einmal
    assert model.num_parameters() == cfg.num_parameters()


# ---------------------------------------------------------------------------
# Bausteine
# ---------------------------------------------------------------------------
def test_rmsnorm_shape_and_finite():
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_rope_preserves_shape():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    q2, k2 = rope(q, k)
    assert q2.shape == q.shape and k2.shape == k.shape


def test_kv_cache_accumulates():
    cache = KVCache()
    k1 = torch.randn(1, 2, 3, 8)
    cache.update(k1, k1)
    assert cache.seq_len == 3
    k2 = torch.randn(1, 2, 1, 8)
    cache.update(k2, k2)
    assert cache.seq_len == 4


# ---------------------------------------------------------------------------
# Vorwärtsdurchlauf
# ---------------------------------------------------------------------------
def test_forward_training_returns_loss():
    cfg = tiny_config()
    model = DecoderLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_forward_inference_returns_last_position():
    cfg = tiny_config()
    model = DecoderLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    logits, loss = model(x)
    # Inferenz gibt nur die letzte Position zurück
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert loss is None


def test_kv_cache_matches_full_forward():
    """Schrittweise Dekodierung mit Cache muss dem Volldurchlauf entsprechen."""
    torch.manual_seed(0)
    cfg = tiny_config()
    model = DecoderLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        full_logits, _ = model(x, targets=torch.zeros_like(x))
        last_full = full_logits[:, -1, :]

        caches = model.new_kv_caches()
        logits = None
        for i in range(x.shape[1]):
            logits, _ = model(x[:, i : i + 1], kv_caches=caches)
        last_cached = logits[:, -1, :]

    assert torch.allclose(last_full, last_cached, atol=1e-4)


def test_backward_updates_parameters():
    cfg = tiny_config()
    model = DecoderLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, targets=y)
    loss.backward()
    # Mindestens ein Parameter muss einen Gradienten besitzen
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)
