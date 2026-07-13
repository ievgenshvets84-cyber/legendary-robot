"""Allgemeine Hilfsfunktionen für LLM-Forge (Geräteauswahl, Logging, GPU-Telemetrie)."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Wählt das Rechengerät.

    Args:
        preference: ``"auto"`` (CUDA falls verfügbar, sonst CPU), ``"cuda"``
            oder ``"cpu"``.

    Returns:
        Das ausgewählte ``torch.device``.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA wurde angefordert, ist aber nicht verfügbar.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Setzt alle relevanten Zufallsgeneratoren für reproduzierbare Läufe."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(name: str, log_dir: str | Path = "logs", level: int = logging.INFO) -> logging.Logger:
    """Erzeugt einen Logger, der auf Konsole und Datei schreibt."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # bereits konfiguriert
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def gpu_stats() -> list[dict]:
    """Liefert GPU-Auslastung und Speicherbelegung, sofern verfügbar.

    Nutzt bevorzugt ``pynvml`` (nvidia-ml) für Auslastungswerte; fällt sonst
    auf die von PyTorch berichtete Speicherbelegung zurück. Auf Systemen
    ohne CUDA wird eine leere Liste zurückgegeben.
    """
    if not torch.cuda.is_available():
        return []

    # Versuch 1: pynvml für echte Auslastungsdaten
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        stats = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            stats.append(
                {
                    "index": i,
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "gpu_util_percent": util.gpu,
                    "mem_used_mb": round(mem.used / 1024 ** 2, 1),
                    "mem_total_mb": round(mem.total / 1024 ** 2, 1),
                    "mem_percent": round(100 * mem.used / mem.total, 1),
                }
            )
        pynvml.nvmlShutdown()
        return stats
    except Exception:
        # Versuch 2: reiner PyTorch-Fallback (nur Speicher)
        stats = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory
            used = torch.cuda.memory_allocated(i)
            stats.append(
                {
                    "index": i,
                    "name": props.name,
                    "gpu_util_percent": None,
                    "mem_used_mb": round(used / 1024 ** 2, 1),
                    "mem_total_mb": round(total / 1024 ** 2, 1),
                    "mem_percent": round(100 * used / total, 1) if total else 0.0,
                }
            )
        return stats


def autocast_context(device: torch.device, enabled: bool):
    """Liefert einen passenden ``torch.autocast``-Kontext für Mixed Precision.

    Auf CUDA wird bfloat16 bevorzugt (falls unterstützt), sonst float16.
    Auf CPU bleibt Mixed Precision deaktiviert.
    """
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
