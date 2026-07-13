"""Modellpaket von LLM-Forge.

Enthält die vollständige, eigenständige Decoder-Transformer-Architektur:
Konfiguration, Grundbausteine (RMSNorm, RoPE, GQA-Attention, GeGLU-FFN),
das Gesamtmodell sowie LoRA-Adapter für parametereffizientes Fine-Tuning.
"""

from models.config import ModelConfig
from models.transformer import DecoderLM
from models.lora import LoRAConfig, LoRALinear, inject_lora, merge_lora, lora_state_dict

__all__ = [
    "ModelConfig",
    "DecoderLM",
    "LoRAConfig",
    "LoRALinear",
    "inject_lora",
    "merge_lora",
    "lora_state_dict",
]
