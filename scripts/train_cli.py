#!/usr/bin/env python3
"""Kommandozeilen-Werkzeug für einen kompletten LLM-Forge-Durchlauf ohne Webinterface.

Beispiel (kleines Modell auf den Beispieldaten trainieren)::

    python -m scripts.train_cli demo --steps 200

Der Befehl legt ein Modell an, trainiert einen Tokenizer, bereitet die Daten
auf, trainiert das Modell und erzeugt anschließend eine Beispielausgabe.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from backend.services.forge_service import ForgeService
from backend.settings import ServerSettings

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "datasets" / "samples" / "sample_corpus.txt"


def build_service() -> ForgeService:
    settings = ServerSettings.load(ROOT / "config" / "server.yaml")
    return ForgeService(settings)


def cmd_demo(args: argparse.Namespace) -> None:
    """End-to-End-Demo: Modell -> Tokenizer -> Daten -> Training -> Generierung."""
    service = build_service()

    # Beispielkorpus in das Rohdatenverzeichnis kopieren, falls nötig
    if SAMPLE.exists():
        dest = service.datasets.raw_dir / SAMPLE.name
        if not dest.exists():
            shutil.copyfile(SAMPLE, dest)
            print(f"[i] Beispielkorpus kopiert nach {dest}")

    name = args.name
    # Kleines Modell für schnelle CPU-Läufe
    arch = {
        "name": name,
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 384,
        "max_seq_len": 128,
        "vocab_size": 2048,
    }
    try:
        service.create_model(arch)
        print(f"[✓] Modell '{name}' angelegt.")
    except ValueError:
        print(f"[i] Modell '{name}' existiert bereits – wird wiederverwendet.")

    print("[…] Trainiere Tokenizer …")
    info = service.train_tokenizer(name, vocab_size=2048, min_frequency=1)
    print(f"[✓] Tokenizer: Vokabular {info['vocab_size']}")

    print("[…] Bereite Datensatz auf …")
    meta = service.prepare_dataset(name)
    print(f"[✓] {meta['num_tokens']} Token vorbereitet.")

    print(f"[…] Starte Training über {args.steps} Schritte …")
    service.start_training(
        model_name=name,
        dataset_name="train",
        params={
            "max_steps": args.steps,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "seq_len": 64,
            "eval_interval": max(20, args.steps // 4),
            "checkpoint_interval": max(50, args.steps // 2),
            "warmup_steps": min(20, args.steps // 5),
            "device": args.device,
        },
    )
    # Warten, bis der Hintergrund-Thread fertig ist
    while service.is_training():
        import time
        time.sleep(1)
    status = service.training_status()
    print(f"[✓] Training beendet: Status={status['status']} Loss={status.get('train_loss')}")

    print("[…] Beispielgenerierung …")
    out = service.generate_text({
        "model_name": name,
        "prompt": args.prompt,
        "checkpoint": "latest",
        "max_new_tokens": 60,
        "temperature": 0.8,
    })
    print("\n--- Generierter Text ---")
    print(out["text"])
    print("------------------------")
    service.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-Forge CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="End-to-End-Demo ausführen")
    demo.add_argument("name", nargs="?", default="demo", help="Modellname")
    demo.add_argument("--steps", type=int, default=200, help="Anzahl Trainingsschritte")
    demo.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    demo.add_argument("--prompt", default="Das Sprachmodell", help="Prompt für die Generierung")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
