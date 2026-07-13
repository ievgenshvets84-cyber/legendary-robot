"""Router für Inferenz: Chat/Generierung (mit Streaming) und Evaluation."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.dependencies import get_service
from backend.schemas import EvaluateRequest, GenerateRequest, GenerateResponse
from backend.services.forge_service import ForgeService
from inference.generator import GenerationConfig

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, service: ForgeService = Depends(get_service)) -> GenerateResponse:
    """Erzeugt eine vollständige Antwort (nicht gestreamt)."""
    try:
        result = service.generate_text(req.model_dump())
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return GenerateResponse(**result)


@router.post("/stream")
def generate_stream(req: GenerateRequest, service: ForgeService = Depends(get_service)) -> StreamingResponse:
    """Streamt die Antwort tokenweise als Server-Sent-Events (SSE).

    Jedes Ereignis hat die Form ``data: {"token": "..."}``; das Ende wird mit
    ``data: {"done": true}`` signalisiert. Das Chatfenster im Frontend liest
    diesen Strom und hängt die Fragmente fortlaufend an.
    """
    try:
        generator = service.build_generator(req.model_name, req.checkpoint)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    cfg = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )

    def event_stream():
        try:
            for fragment in generator.stream(req.prompt, cfg):
                yield f"data: {json.dumps({'token': fragment}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:  # pragma: no cover
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/evaluate", response_model=dict)
def evaluate(req: EvaluateRequest, service: ForgeService = Depends(get_service)) -> dict:
    """Bewertet ein Modell (Verlust und Perplexität) auf einem Datensatz."""
    try:
        return service.evaluate(req.model_name, req.checkpoint, req.dataset_name, req.max_batches)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
