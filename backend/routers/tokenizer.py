"""Router für das Training des eigenen Tokenizers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.dependencies import get_service
from backend.schemas import TokenizerInfo, TokenizerTrainRequest
from backend.services.forge_service import ForgeService

router = APIRouter(prefix="/api/tokenizer", tags=["tokenizer"])


@router.post("/train", response_model=TokenizerInfo)
def train_tokenizer(
    req: TokenizerTrainRequest, service: ForgeService = Depends(get_service)
) -> TokenizerInfo:
    """Trainiert einen BPE-Tokenizer auf dem hochgeladenen Korpus."""
    try:
        info = service.train_tokenizer(
            req.model_name, req.vocab_size, req.min_frequency, req.dataset_files
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TokenizerInfo(**info)


@router.get("/{model_name}", response_model=TokenizerInfo)
def tokenizer_info(model_name: str, service: ForgeService = Depends(get_service)) -> TokenizerInfo:
    """Liefert Informationen zum trainierten Tokenizer eines Modells."""
    try:
        tokenizer = service.load_tokenizer(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return TokenizerInfo(
        vocab_size=tokenizer.vocab_size,
        path=str(service._tokenizer_path(model_name)),
    )
