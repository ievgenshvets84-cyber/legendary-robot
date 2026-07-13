"""Tests für LoRA-Injektion und Quantisierung."""

from __future__ import annotations

import torch

from inference.quantization import QuantConfig, QuantizedLinear, quantize_model
from models.config import ModelConfig
from models.lora import LoRAConfig, LoRALinear, inject_lora, lora_state_dict, merge_lora
from models.transformer import DecoderLM


def tiny_model() -> DecoderLM:
    cfg = ModelConfig(
        name="t", vocab_size=128, hidden_size=32, num_layers=2,
        num_heads=4, num_kv_heads=2, intermediate_size=64, max_seq_len=32,
    )
    return DecoderLM(cfg)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
def test_lora_injection_replaces_targets():
    model = tiny_model()
    replaced = inject_lora(model, LoRAConfig(r=4, target_modules=["q_proj", "v_proj"]))
    # 2 Schichten × 2 Zielprojektionen
    assert replaced == 4
    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_layers) == 4


def test_lora_only_adapters_trainable():
    model = tiny_model()
    inject_lora(model, LoRAConfig(r=4))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora" in n for n in trainable)
    assert len(trainable) > 0


def test_lora_initially_identity():
    """LoRA-B ist mit Null initialisiert -> Ausgabe zunächst unverändert."""
    torch.manual_seed(0)
    model = tiny_model().eval()
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        before, _ = model(x, targets=torch.zeros_like(x))
    inject_lora(model, LoRAConfig(r=4))
    model.eval()
    with torch.no_grad():
        after, _ = model(x, targets=torch.zeros_like(x))
    assert torch.allclose(before, after, atol=1e-5)


def test_lora_state_dict_only_adapters():
    model = tiny_model()
    inject_lora(model, LoRAConfig(r=4))
    sd = lora_state_dict(model)
    assert len(sd) > 0
    assert all("lora_a" in k or "lora_b" in k for k in sd)


def test_lora_merge_restores_linear():
    model = tiny_model()
    inject_lora(model, LoRAConfig(r=4))
    merged = merge_lora(model)
    assert merged > 0
    assert not any(isinstance(m, LoRALinear) for m in model.modules())


# ---------------------------------------------------------------------------
# Quantisierung
# ---------------------------------------------------------------------------
def test_quantize_8bit_forward_close():
    torch.manual_seed(0)
    model = tiny_model().eval()
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        ref, _ = model(x, targets=torch.zeros_like(x))
    quantize_model(model, QuantConfig(bits=8, group_size=16))
    with torch.no_grad():
        out, _ = model(x, targets=torch.zeros_like(x))
    # 8-Bit-Quantisierung sollte nur geringe Abweichung erzeugen
    assert out.shape == ref.shape
    assert (out - ref).abs().mean() < 1.0


def test_quantize_replaces_linear():
    model = tiny_model()
    quantize_model(model, QuantConfig(bits=4, group_size=16))
    assert any(isinstance(m, QuantizedLinear) for m in model.modules())


def test_quantized_linear_4bit_packs():
    base = torch.nn.Linear(32, 16)
    q = QuantizedLinear(base, QuantConfig(bits=4, group_size=16))
    x = torch.randn(2, 32)
    out = q(x)
    assert out.shape == (2, 16)
    # 4-Bit gepackt: halb so viele Bytes wie Elemente
    assert q.quant.numel() * 2 >= 16 * 32 - 16
