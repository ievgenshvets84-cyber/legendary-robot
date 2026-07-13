"""FastAPI-Anwendung von LLM-Forge.

Bindet alle Router ein, stellt das statische Webinterface bereit und
initialisiert den zentralen :class:`ForgeService`. Start des Servers::

    python -m backend.main
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.dependencies import get_service, init_service
from backend.routers import datasets, inference, management, models, tokenizer, training
from backend.settings import ServerSettings
from backend.utils import setup_logger

logger = setup_logger("server")
settings = ServerSettings.load()

FRONTEND_DIR = settings.root / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisiert und beendet den Service über den App-Lebenszyklus."""
    init_service(settings)
    logger.info("LLM-Forge gestartet – Webinterface unter http://%s:%d", settings.host, settings.port)
    yield
    # Sauberes Herunterfahren
    try:
        get_service().shutdown()
    except Exception:  # pragma: no cover
        logger.warning("Fehler beim Herunterfahren des Service.", exc_info=True)


app = FastAPI(
    title="LLM-Forge",
    description="Framework zum Entwickeln, Trainieren und Betreiben eines eigenen LLM von Grund auf.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS erlauben (Frontend kann von beliebigem Ursprung zugreifen)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API-Router registrieren
app.include_router(models.router)
app.include_router(tokenizer.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(management.router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Liefert die Einstiegsseite des Webinterfaces."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Statische Dateien (CSS/JS) bereitstellen
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def run() -> None:
    """Startet den Uvicorn-Server (Einstiegspunkt ``llm-forge``)."""
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
