"""Router für Checkpoints, Quantisierung, Export und Systeminfo."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_service
from backend.schemas import ExportRequest, MessageResponse, QuantizeRequest
from backend.services.forge_service import ForgeService

router = APIRouter(prefix="/api", tags=["management"])


@router.get("/checkpoints/{model_name}", response_model=list)
def list_checkpoints(model_name: str, service: ForgeService = Depends(get_service)) -> list:
    """Listet vorhandene Checkpoints eines Modells."""
    return service.list_checkpoints(model_name)


@router.post("/quantize", response_model=MessageResponse)
def quantize(req: QuantizeRequest, service: ForgeService = Depends(get_service)) -> MessageResponse:
    """Quantisiert ein Modell auf 4 oder 8 Bit."""
    try:
        data = service.quantize(req.model_name, req.checkpoint, req.bits, req.group_size)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return MessageResponse(ok=True, message=f"Modell auf {req.bits} Bit quantisiert.", data=data)


@router.post("/export", response_model=MessageResponse)
def export(req: ExportRequest, service: ForgeService = Depends(get_service)) -> MessageResponse:
    """Exportiert ein Modell nach ONNX oder GGUF."""
    try:
        data = service.export(req.model_name, req.checkpoint, req.format)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return MessageResponse(ok=True, message=f"Export nach {req.format} abgeschlossen.", data=data)


@router.get("/system/info", response_model=dict)
def system_info(service: ForgeService = Depends(get_service)) -> dict:
    """Systeminformationen: Gerät, CUDA, GPU-Auslastung, Trainingsstatus."""
    return service.system_info()


@router.get("/system/gpu", response_model=list)
def gpu_info(service: ForgeService = Depends(get_service)) -> list:
    """Aktuelle GPU-Auslastung (für das Live-Dashboard)."""
    return service.system_info()["gpus"]


@router.get("/system/health", response_model=dict)
def health() -> dict:
    """Einfacher Healthcheck (für Docker/Monitoring)."""
    return {"status": "ok"}
