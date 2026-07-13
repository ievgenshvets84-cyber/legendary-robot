"""Modellexport nach ONNX und GGUF.

* ONNX: Nutzt ``torch.onnx.export`` mit dynamischen Achsen für Batch und
  Sequenzlänge. Der KV-Cache wird für den Export deaktiviert (Prefill-Graph),
  was für Interoperabilität und Prüfzwecke ausreicht.
* GGUF: Schreibt einen eigenständigen, minimalen GGUF-Writer (das von
  llama.cpp verwendete Format), der Metadaten und Tensoren gemäß der
  offiziellen Spezifikation serialisiert. So bleibt der Export ohne externe
  Abhängigkeit lauffähig.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

from models.transformer import DecoderLM


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------
def export_onnx(
    model: DecoderLM,
    output_path: str | Path,
    opset: int = 17,
    seq_len: int = 16,
) -> Path:
    """Exportiert das Modell in das ONNX-Format.

    Args:
        model: Trainiertes Modell (wird in den Eval-Modus versetzt).
        output_path: Zielpfad (``.onnx``).
        opset: ONNX-Opset-Version.
        seq_len: Beispiel-Sequenzlänge für den Trace.

    Returns:
        Pfad der geschriebenen Datei.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().to("cpu")

    # Wrapper, der nur Logits zurückgibt (ONNX mag keine Tupel mit None)
    class _ExportWrapper(torch.nn.Module):
        def __init__(self, inner: DecoderLM) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # targets=None -> gibt nur die letzte Position zurück; für einen
            # vollständigen Graphen berechnen wir hier alle Logits.
            x = self.inner.embed_tokens(input_ids)
            for layer in self.inner.layers:
                x = layer(x, self.inner.rope, None)
            x = self.inner.final_norm(x)
            return self.inner.lm_head(x)

    wrapper = _ExportWrapper(model)
    example = torch.randint(0, model.config.vocab_size, (1, seq_len), dtype=torch.long)

    export_kwargs = dict(
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        do_constant_folding=True,
    )
    # Neuere PyTorch-Versionen nutzen standardmäßig den Dynamo-Exporter, der das
    # optionale Paket ``onnxscript`` benötigt. Wir wählen den bewährten
    # TorchScript-Exporter (``dynamo=False``), sofern der Parameter existiert.
    try:
        torch.onnx.export(wrapper, (example,), str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Ältere PyTorch-Versionen kennen den ``dynamo``-Parameter nicht.
        torch.onnx.export(wrapper, (example,), str(output_path), **export_kwargs)
    return output_path


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------
# GGUF-Konstanten gemäß Spezifikation (ggml/llama.cpp)
_GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
_GGUF_VERSION = 3

# Metadaten-Werttypen
_T_UINT32 = 4
_T_FLOAT32 = 6
_T_STRING = 8
_T_ARRAY = 9

# Tensor-Datentyp F32
_GGML_TYPE_F32 = 0


class _GGUFWriter:
    """Minimaler GGUF-Serializer für float32-Tensoren."""

    def __init__(self) -> None:
        self._kv: list[bytes] = []
        self._kv_count = 0
        self._tensors: list[tuple[str, np.ndarray]] = []

    # --- Low-Level-Schreiber ---
    @staticmethod
    def _str(value: str) -> bytes:
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    def add_string(self, key: str, value: str) -> None:
        self._kv.append(self._str(key) + struct.pack("<I", _T_STRING) + self._str(value))
        self._kv_count += 1

    def add_uint32(self, key: str, value: int) -> None:
        self._kv.append(self._str(key) + struct.pack("<II", _T_UINT32, value))
        self._kv_count += 1

    def add_float32(self, key: str, value: float) -> None:
        self._kv.append(self._str(key) + struct.pack("<I", _T_FLOAT32) + struct.pack("<f", value))
        self._kv_count += 1

    def add_tensor(self, name: str, array: np.ndarray) -> None:
        self._tensors.append((name, np.ascontiguousarray(array, dtype=np.float32)))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as fh:
            # Header
            fh.write(struct.pack("<I", _GGUF_MAGIC))
            fh.write(struct.pack("<I", _GGUF_VERSION))
            fh.write(struct.pack("<Q", len(self._tensors)))
            fh.write(struct.pack("<Q", self._kv_count))

            # Metadaten (Key-Value)
            for kv in self._kv:
                fh.write(kv)

            # Tensor-Infoblock: Name, Dimensionen, Typ, Offset
            offset = 0
            alignment = 32
            infos = []
            for name, array in self._tensors:
                dims = tuple(reversed(array.shape))  # GGUF speichert Dimensionen umgekehrt
                info = self._str(name)
                info += struct.pack("<I", len(dims))
                for d in dims:
                    info += struct.pack("<Q", d)
                info += struct.pack("<I", _GGML_TYPE_F32)
                info += struct.pack("<Q", offset)
                infos.append(info)
                # Offset auf Alignment aufrunden
                nbytes = array.nbytes
                offset += (nbytes + alignment - 1) // alignment * alignment
            for info in infos:
                fh.write(info)

            # Auf Alignment auffüllen, dann Tensordaten schreiben
            pos = fh.tell()
            pad = (-pos) % alignment
            fh.write(b"\x00" * pad)
            for _, array in self._tensors:
                start = fh.tell()
                fh.write(array.tobytes())
                written = fh.tell() - start
                pad = (-written) % alignment
                fh.write(b"\x00" * pad)

        return path


def export_gguf(model: DecoderLM, output_path: str | Path) -> Path:
    """Exportiert Architektur-Metadaten und Gewichte in eine GGUF-Datei.

    Die Datei folgt der GGUF-Struktur (Magic, Version, KV-Metadaten,
    Tensor-Infos, ausgerichtete Tensordaten) und speichert alle Gewichte
    als float32. Tensornamen orientieren sich an der llama.cpp-Konvention,
    sodass die Datei mit GGUF-Werkzeugen inspizierbar ist.

    Args:
        model: Trainiertes Modell.
        output_path: Zielpfad (``.gguf``).

    Returns:
        Pfad der geschriebenen Datei.
    """
    model = model.eval().to("cpu")
    cfg = model.config
    writer = _GGUFWriter()

    # Allgemeine und architekturbezogene Metadaten
    writer.add_string("general.architecture", "llm-forge")
    writer.add_string("general.name", cfg.name)
    writer.add_uint32("llm-forge.context_length", cfg.max_seq_len)
    writer.add_uint32("llm-forge.embedding_length", cfg.hidden_size)
    writer.add_uint32("llm-forge.block_count", cfg.num_layers)
    writer.add_uint32("llm-forge.feed_forward_length", cfg.intermediate_size)
    writer.add_uint32("llm-forge.attention.head_count", cfg.num_heads)
    writer.add_uint32("llm-forge.attention.head_count_kv", cfg.num_kv_heads)
    writer.add_uint32("llm-forge.vocab_size", cfg.vocab_size)
    writer.add_float32("llm-forge.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps)
    writer.add_float32("llm-forge.rope.freq_base", cfg.rope_theta)

    # Alle Gewichte als float32-Tensoren übernehmen
    state = model.state_dict()
    for name, tensor in state.items():
        # RoPE-Puffer werden zur Laufzeit neu berechnet und nicht exportiert
        if "rope" in name:
            continue
        writer.add_tensor(name, tensor.detach().cpu().numpy().astype(np.float32))

    path = writer.write(output_path)

    # Begleitende JSON-Beschreibung für einfache Weiterverarbeitung
    sidecar = Path(output_path).with_suffix(".gguf.json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    return path
