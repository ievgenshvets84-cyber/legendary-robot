"""Laden und Bereitstellen der Serverkonfiguration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ServerSettings:
    """Laufzeiteinstellungen des FastAPI-Servers und der Projektpfade."""

    host: str = "0.0.0.0"
    port: int = 8000
    database: str = "config/llm_forge.db"
    datasets_dir: str = "datasets"
    checkpoints_dir: str = "checkpoints"
    exports_dir: str = "exports"
    logs_dir: str = "logs"
    tokenizer_dir: str = "tokenizer/trained"
    max_upload_mb: int = 2048

    # Nicht aus YAML: abgeleitetes Projektstammverzeichnis
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @classmethod
    def load(cls, path: str | Path = "config/server.yaml") -> "ServerSettings":
        """Lädt die Einstellungen aus YAML und überschreibt mit Umgebungsvariablen.

        Umgebungsvariablen (haben Vorrang):
            ``LLM_FORGE_HOST``, ``LLM_FORGE_PORT``, ``LLM_FORGE_DB``.
        """
        data: dict = {}
        cfg_path = Path(path)
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

        settings = cls(
            host=data.get("host", cls.host),
            port=int(data.get("port", cls.port)),
            database=data.get("database", cls.database),
            datasets_dir=data.get("datasets_dir", cls.datasets_dir),
            checkpoints_dir=data.get("checkpoints_dir", cls.checkpoints_dir),
            exports_dir=data.get("exports_dir", cls.exports_dir),
            logs_dir=data.get("logs_dir", cls.logs_dir),
            tokenizer_dir=data.get("tokenizer_dir", cls.tokenizer_dir),
            max_upload_mb=int(data.get("max_upload_mb", cls.max_upload_mb)),
        )

        # Umgebungsvariablen anwenden
        settings.host = os.environ.get("LLM_FORGE_HOST", settings.host)
        settings.port = int(os.environ.get("LLM_FORGE_PORT", settings.port))
        settings.database = os.environ.get("LLM_FORGE_DB", settings.database)
        return settings

    def ensure_dirs(self) -> None:
        """Legt alle benötigten Laufzeitverzeichnisse an."""
        for d in (self.datasets_dir, self.checkpoints_dir, self.exports_dir,
                  self.logs_dir, self.tokenizer_dir):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        (self.root / self.datasets_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.root / self.datasets_dir / "processed").mkdir(parents=True, exist_ok=True)
