"""End-to-End-Test der FastAPI-Schnittstelle über den Testclient."""

from __future__ import annotations

import importlib

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Startet die App mit isolierten Verzeichnissen in einem Temp-Ordner."""
    # Umgebungsvariablen auf das Temp-Verzeichnis lenken
    monkeypatch.setenv("LLM_FORGE_DB", str(tmp_path / "test.db"))

    # settings so patchen, dass alle Pfade unter tmp_path liegen
    from backend import settings as settings_mod

    orig_load = settings_mod.ServerSettings.load

    def patched_load(path="config/server.yaml"):
        s = orig_load(path)
        s.root = tmp_path
        s.database = "test.db"
        return s

    monkeypatch.setattr(settings_mod.ServerSettings, "load", staticmethod(patched_load))

    # main-Modul neu laden, damit es die gepatchten Settings verwendet
    import backend.main as main_mod
    importlib.reload(main_mod)

    with TestClient(main_mod.app) as c:
        yield c


def test_health(client):
    res = client.get("/api/system/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_system_info(client):
    res = client.get("/api/system/info")
    assert res.status_code == 200
    body = res.json()
    assert "device" in body
    assert "torch_version" in body


def test_create_and_list_model(client):
    arch = {
        "name": "apitest",
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 64,
        "max_seq_len": 32,
        "vocab_size": 256,
    }
    res = client.post("/api/models", json=arch)
    assert res.status_code == 200, res.text
    assert res.json()["name"] == "apitest"

    res = client.get("/api/models")
    assert any(m["name"] == "apitest" for m in res.json())


def test_duplicate_model_rejected(client):
    arch = {"name": "dup", "hidden_size": 32, "num_layers": 1, "num_heads": 4,
            "num_kv_heads": 2, "intermediate_size": 64, "max_seq_len": 32, "vocab_size": 256}
    assert client.post("/api/models", json=arch).status_code == 200
    assert client.post("/api/models", json=arch).status_code == 400


def test_invalid_architecture_rejected(client):
    # hidden_size nicht durch num_heads teilbar
    arch = {"name": "bad", "hidden_size": 33, "num_layers": 1, "num_heads": 4,
            "num_kv_heads": 2, "intermediate_size": 64, "max_seq_len": 32, "vocab_size": 256}
    assert client.post("/api/models", json=arch).status_code == 400


def test_full_flow_upload_train_generate(client):
    """Kompletter Ablauf: Modell -> Upload -> Tokenizer -> Daten -> Training -> Chat."""
    arch = {"name": "flow", "hidden_size": 32, "num_layers": 2, "num_heads": 4,
            "num_kv_heads": 2, "intermediate_size": 64, "max_seq_len": 32, "vocab_size": 300}
    assert client.post("/api/models", json=arch).status_code == 200

    corpus = "\n".join(
        f"das modell lernt sprache in beispiel nummer {i} sehr gut" for i in range(200)
    )
    res = client.post(
        "/api/datasets/upload",
        files={"file": ("c.txt", corpus, "text/plain")},
    )
    assert res.status_code == 200, res.text

    res = client.post("/api/tokenizer/train", json={"model_name": "flow", "vocab_size": 300, "min_frequency": 1})
    assert res.status_code == 200, res.text

    res = client.post("/api/datasets/prepare?model_name=flow")
    assert res.status_code == 200, res.text

    res = client.post("/api/training/start", json={
        "model_name": "flow", "dataset_name": "train",
        "params": {"max_steps": 5, "warmup_steps": 1, "batch_size": 2, "grad_accum_steps": 1,
                   "seq_len": 16, "device": "cpu", "mixed_precision": False},
    })
    assert res.status_code == 200, res.text

    # Auf Abschluss des Hintergrund-Trainings warten
    import time
    for _ in range(60):
        status = client.get("/api/training/status").json()
        if status["status"] in ("completed", "stopped", "error", "idle") and status["step"] >= 5:
            break
        if status["status"] in ("completed", "error"):
            break
        time.sleep(0.5)

    res = client.post("/api/inference/generate", json={
        "model_name": "flow", "prompt": "das modell", "checkpoint": "latest",
        "max_new_tokens": 5, "temperature": 0.8,
    })
    assert res.status_code == 200, res.text
    assert "text" in res.json()
