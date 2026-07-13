"""SQLite-Datenbankschicht von LLM-Forge.

Speichert Modellkonfigurationen, Trainingsläufe und deren Metrikverläufe.
Bewusst schlank gehalten (nur die Standardbibliothek ``sqlite3``), mit
thread-sicherem Zugriff über eine Verbindungssperre, da FastAPI und der
Trainings-Thread parallel schreiben können.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Database:
    """Dünne Kapsel um eine SQLite-Datenbank mit den Projektschemata."""

    def __init__(self, path: str | Path = "config/llm_forge.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + Lock: Zugriff aus mehreren Threads erlauben
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    name          TEXT UNIQUE NOT NULL,
                    config_json   TEXT NOT NULL,
                    tokenizer_path TEXT,
                    created_at    REAL NOT NULL,
                    updated_at    REAL NOT NULL,
                    notes         TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id       INTEGER NOT NULL,
                    config_json    TEXT NOT NULL,
                    status         TEXT NOT NULL DEFAULT 'created',
                    dataset_name   TEXT,
                    started_at     REAL,
                    finished_at    REAL,
                    best_val_loss  REAL,
                    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id        INTEGER NOT NULL,
                    step          INTEGER NOT NULL,
                    train_loss    REAL,
                    val_loss      REAL,
                    learning_rate REAL,
                    tokens_per_sec REAL,
                    perplexity    REAL,
                    created_at    REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, step);
                """
            )

    # ------------------------------------------------------------------
    # Modelle
    # ------------------------------------------------------------------
    def create_model(self, name: str, config: dict, tokenizer_path: str | None = None) -> int:
        """Legt einen neuen Modelleintrag an und gibt dessen ID zurück."""
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO models (name, config_json, tokenizer_path, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, json.dumps(config), tokenizer_path, now, now),
            )
            return int(cur.lastrowid)

    def update_model(self, name: str, config: dict | None = None,
                     tokenizer_path: str | None = None) -> None:
        """Aktualisiert Konfiguration und/oder Tokenizer-Pfad eines Modells."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT config_json FROM models WHERE name = ?", (name,)).fetchone()
            if row is None:
                raise KeyError(f"Modell '{name}' existiert nicht.")
            new_config = json.dumps(config) if config is not None else row["config_json"]
            self._conn.execute(
                "UPDATE models SET config_json = ?, tokenizer_path = COALESCE(?, tokenizer_path), "
                "updated_at = ? WHERE name = ?",
                (new_config, tokenizer_path, time.time(), name),
            )

    def get_model(self, name: str) -> dict[str, Any] | None:
        """Liest einen Modelleintrag anhand des Namens."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
        return self._model_row(row) if row else None

    def list_models(self) -> list[dict[str, Any]]:
        """Listet alle Modelle (neueste zuerst)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        return [self._model_row(r) for r in rows]

    def delete_model(self, name: str) -> None:
        """Löscht ein Modell samt zugehöriger Läufe und Metriken."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM models WHERE name = ?", (name,))

    @staticmethod
    def _model_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "config": json.loads(row["config_json"]),
            "tokenizer_path": row["tokenizer_path"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "notes": row["notes"],
        }

    # ------------------------------------------------------------------
    # Trainingsläufe
    # ------------------------------------------------------------------
    def create_run(self, model_id: int, config: dict, dataset_name: str | None) -> int:
        """Erzeugt einen Trainingslauf-Eintrag."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO training_runs (model_id, config_json, status, dataset_name, started_at) "
                "VALUES (?, ?, 'running', ?, ?)",
                (model_id, json.dumps(config), dataset_name, time.time()),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, best_val_loss: float | None) -> None:
        """Markiert einen Lauf als beendet und speichert das beste Ergebnis."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE training_runs SET status = ?, finished_at = ?, best_val_loss = ? WHERE id = ?",
                (status, time.time(), best_val_loss, run_id),
            )

    def update_run_status(self, run_id: int, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE training_runs SET status = ? WHERE id = ?", (status, run_id))

    def list_runs(self, model_id: int | None = None) -> list[dict[str, Any]]:
        """Listet Trainingsläufe, optional gefiltert nach Modell."""
        query = "SELECT * FROM training_runs"
        params: tuple = ()
        if model_id is not None:
            query += " WHERE model_id = ?"
            params = (model_id,)
        query += " ORDER BY id DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Metriken
    # ------------------------------------------------------------------
    def add_metric(self, run_id: int, record: dict[str, Any]) -> None:
        """Speichert einen Metrikdatenpunkt eines Laufs."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO metrics (run_id, step, train_loss, val_loss, learning_rate, "
                "tokens_per_sec, perplexity, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    record.get("step", 0),
                    record.get("train_loss"),
                    record.get("val_loss"),
                    record.get("learning_rate"),
                    record.get("tokens_per_sec"),
                    record.get("perplexity"),
                    time.time(),
                ),
            )

    def get_metrics(self, run_id: int, limit: int = 5000) -> list[dict[str, Any]]:
        """Liest die Metrikreihe eines Laufs (aufsteigend nach Schritt)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT step, train_loss, val_loss, learning_rate, tokens_per_sec, perplexity "
                "FROM metrics WHERE run_id = ? ORDER BY step ASC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Schließt die Datenbankverbindung."""
        with self._lock:
            self._conn.close()
