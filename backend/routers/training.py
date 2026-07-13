"""Router zur Steuerung des Trainings (Start/Pause/Fortsetzen/Stopp/Metriken)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_service
from backend.schemas import MessageResponse, StartTrainingRequest, TrainingStatus
from backend.services.forge_service import ForgeService

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", response_model=TrainingStatus)
def start_training(
    req: StartTrainingRequest, service: ForgeService = Depends(get_service)
) -> TrainingStatus:
    """Startet einen Trainings- oder Fine-Tuning-Lauf."""
    try:
        status = service.start_training(
            model_name=req.model_name,
            dataset_name=req.dataset_name,
            params=req.params.model_dump(),
            resume_from=req.resume_from,
            use_lora=req.use_lora,
            lora_rank=req.lora_rank,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TrainingStatus(**status)


@router.get("/status", response_model=TrainingStatus)
def training_status(service: ForgeService = Depends(get_service)) -> TrainingStatus:
    """Aktueller Trainingsstatus (für Live-Anzeige im Webinterface)."""
    return TrainingStatus(**service.training_status())


@router.post("/pause", response_model=TrainingStatus)
def pause_training(service: ForgeService = Depends(get_service)) -> TrainingStatus:
    """Pausiert das laufende Training."""
    try:
        return TrainingStatus(**service.pause_training())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/resume", response_model=TrainingStatus)
def resume_training(service: ForgeService = Depends(get_service)) -> TrainingStatus:
    """Setzt ein pausiertes Training fort."""
    try:
        return TrainingStatus(**service.resume_training())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/stop", response_model=TrainingStatus)
def stop_training(service: ForgeService = Depends(get_service)) -> TrainingStatus:
    """Stoppt das laufende Training sauber."""
    try:
        return TrainingStatus(**service.stop_training())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/runs", response_model=list)
def list_runs(model_name: str | None = None, service: ForgeService = Depends(get_service)) -> list:
    """Listet vergangene und laufende Trainingsläufe."""
    try:
        return service.list_runs(model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/runs/{run_id}/metrics", response_model=list)
def run_metrics(run_id: int, service: ForgeService = Depends(get_service)) -> list:
    """Liefert die Metrikreihe (Loss, Perplexität, LR) eines Laufs für Diagramme."""
    return service.get_run_metrics(run_id)
