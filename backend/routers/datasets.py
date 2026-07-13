"""Router für Datensatzverwaltung (Upload, Auflistung, Aufbereitung)."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.dependencies import get_service
from backend.schemas import MessageResponse
from backend.services.forge_service import ForgeService

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Erlaubte Dateiendungen für Rohdaten
_ALLOWED_SUFFIXES = {".txt", ".jsonl"}


@router.get("", response_model=dict)
def list_datasets(service: ForgeService = Depends(get_service)) -> dict:
    """Listet Roh- und verarbeitete Datensätze."""
    return service.list_datasets()


@router.post("/upload", response_model=MessageResponse)
async def upload_dataset(
    file: UploadFile, service: ForgeService = Depends(get_service)
) -> MessageResponse:
    """Lädt eine Rohdatendatei (``.txt``/``.jsonl``) in das Datensatzverzeichnis."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Nur {sorted(_ALLOWED_SUFFIXES)} werden unterstützt.",
        )

    # Dateinamen bereinigen (kein Pfad-Traversal)
    safe_name = Path(file.filename or "upload").name
    dest = service.datasets.raw_dir / safe_name

    max_bytes = service.settings.max_upload_mb * 1024 * 1024
    written = 0
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                await out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Datei überschreitet das Uploadlimit.")
            await out.write(chunk)

    return MessageResponse(
        ok=True,
        message=f"Datei '{safe_name}' hochgeladen.",
        data={"name": safe_name, "size_bytes": written},
    )


@router.post("/prepare", response_model=MessageResponse)
def prepare_dataset(
    model_name: str,
    output_name: str = "train",
    service: ForgeService = Depends(get_service),
) -> MessageResponse:
    """Bereinigt und tokenisiert den Rohkorpus für ein Modell."""
    try:
        meta = service.prepare_dataset(model_name, output_name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return MessageResponse(ok=True, message="Datensatz vorbereitet.", data=meta)


@router.delete("/{name}", response_model=MessageResponse)
def delete_raw(name: str, service: ForgeService = Depends(get_service)) -> MessageResponse:
    """Löscht eine Rohdatendatei."""
    safe_name = Path(name).name
    path = service.datasets.raw_dir / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Datei '{safe_name}' nicht gefunden.")
    path.unlink()
    return MessageResponse(ok=True, message=f"Datei '{safe_name}' gelöscht.")
