"""Router für Modell- und Architekturverwaltung."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_service
from backend.schemas import (
    MessageResponse,
    ModelArchitecture,
    UpdateArchitecture,
)
from backend.services.forge_service import ForgeService

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("", response_model=dict)
def create_model(arch: ModelArchitecture, service: ForgeService = Depends(get_service)) -> dict:
    """Legt ein neues Modell mit der angegebenen Architektur an."""
    try:
        return service.create_model(arch.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("", response_model=list)
def list_models(service: ForgeService = Depends(get_service)) -> list:
    """Listet alle registrierten Modelle."""
    return service.list_models()


@router.get("/{name}", response_model=dict)
def get_model(name: str, service: ForgeService = Depends(get_service)) -> dict:
    """Liefert Details und geschätzte Parameterzahl eines Modells."""
    try:
        config = service.get_model_config(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "name": name,
        "config": config.to_dict(),
        "num_parameters": config.num_parameters(),
        "estimated_size_mb": round(config.num_parameters() * 4 / 1024 ** 2, 2),
    }


@router.patch("/{name}", response_model=dict)
def update_model(
    name: str, updates: UpdateArchitecture, service: ForgeService = Depends(get_service)
) -> dict:
    """Aktualisiert Architekturparameter eines Modells."""
    try:
        return service.update_architecture(name, updates.model_dump(exclude_none=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{name}", response_model=MessageResponse)
def delete_model(name: str, service: ForgeService = Depends(get_service)) -> MessageResponse:
    """Löscht ein Modell."""
    try:
        service.delete_model(name)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return MessageResponse(ok=True, message=f"Modell '{name}' gelöscht.")
