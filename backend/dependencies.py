"""Gemeinsame Abhängigkeiten (Dependency Injection) für die Router.

Der :class:`ForgeService` wird als Singleton gehalten und den Endpunkten
über :func:`get_service` bereitgestellt.
"""

from __future__ import annotations

from backend.services.forge_service import ForgeService
from backend.settings import ServerSettings

_service: ForgeService | None = None


def init_service(settings: ServerSettings) -> ForgeService:
    """Initialisiert das globale Service-Singleton."""
    global _service
    _service = ForgeService(settings)
    return _service


def get_service() -> ForgeService:
    """FastAPI-Dependency: liefert das Service-Singleton."""
    if _service is None:
        raise RuntimeError("ForgeService wurde nicht initialisiert.")
    return _service
