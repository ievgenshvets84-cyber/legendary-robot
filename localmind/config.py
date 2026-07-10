"""Загрузка и хранение конфигурации LocalMind.

Конфигурация читается из config.yaml (если установлен PyYAML) либо из
встроенных значений по умолчанию. Никаких обязательных внешних зависимостей —
агент должен запускаться «из коробки».
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "llm": {
        # Хост локального Ollama. Модель качается командой: ollama pull <model>
        "host": "http://localhost:11434",
        "model": "qwen2.5:7b-instruct",
        "embed_model": "nomic-embed-text",
        "temperature": 0.3,
        "num_ctx": 8192,
        "timeout": 300,
    },
    "agent": {
        "max_steps": 12,        # предел итераций ReAct-цикла на одну задачу
        "verbose": True,
    },
    "memory": {
        "path": "state/memory.jsonl",
        "top_k": 5,             # сколько воспоминаний подмешивать в контекст
        "min_score": 0.15,      # порог релевантности при поиске
    },
    "autonomy": {
        # По умолчанию агент НЕ применяет самоизменения без подтверждения.
        # Это осознанный предохранитель, а не ограничение возможностей.
        "require_approval": True,
        "max_cycles": 25,       # предел циклов автономного режима
        "goal_file": "state/goals.md",
    },
    "safety": {
        # Все файловые операции и запись навыков ограничены этой папкой.
        "workspace": ".",
        "allow_shell": False,   # произвольный shell выключен по умолчанию
        "sandbox_timeout": 20,  # секунды на исполнение сгенерированного кода
    },
    "skills": {
        "dir": "localmind/skills",
    },
    "media": {
        # Локальная генерация медиа на открытых моделях.
        # image_host — API, совместимый с AUTOMATIC1111 / SD.Next (txt2img).
        "image_host": "http://localhost:7860",
        # video_host — локальный сервер видео (обёртка над ComfyUI / SVD и т.п.).
        "video_host": "",
        "save_dir": "state/media",
        "steps": 30,
        "width": 768,
        "height": 768,
        "cfg_scale": 7.0,
        "sampler": "DPM++ 2M Karras",
        "timeout": 600,
    },
}


@dataclass
class Config:
    llm: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["llm"]))
    agent: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["agent"]))
    memory: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["memory"]))
    autonomy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["autonomy"]))
    safety: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["safety"]))
    skills: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["skills"]))
    media: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS["media"]))
    root: Path = field(default_factory=lambda: Path.cwd())

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        """Читает config.yaml, накладывая его поверх значений по умолчанию."""
        cfg = cls()
        cfg.root = Path.cwd()
        candidate = Path(path) if path else cfg.root / "config.yaml"
        data: dict[str, Any] = {}
        if candidate.exists():
            data = _read_yaml(candidate)
        for section in ("llm", "agent", "memory", "autonomy", "safety", "skills", "media"):
            if section in data and isinstance(data[section], dict):
                getattr(cfg, section).update(data[section])
        # Переопределения через переменные окружения (удобно для контейнеров).
        if os.getenv("LOCALMIND_MODEL"):
            cfg.llm["model"] = os.environ["LOCALMIND_MODEL"]
        if os.getenv("LOCALMIND_HOST"):
            cfg.llm["host"] = os.environ["LOCALMIND_HOST"]
        return cfg

    def workspace(self) -> Path:
        return (self.root / str(self.safety["workspace"])).resolve()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["root"] = str(self.root)
        return d


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        # Минимальный парсер «ключ: значение» на случай отсутствия PyYAML.
        return _read_yaml_fallback(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _read_yaml_fallback(path: Path) -> dict[str, Any]:
    """Разбор плоского YAML с одноуровневой вложенностью без сторонних либ."""
    result: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1].strip()
            section = {}
            result[key] = section
        elif ":" in line and section is not None:
            k, _, v = line.strip().partition(":")
            section[k.strip()] = _coerce(v.strip())
    return result


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"').strip("'")
