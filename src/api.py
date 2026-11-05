"""Módulo de API y endpoints para EcoClasifica.

Re-exporta las funciones de rutas y provee la interfaz para llamadas de API y servidor FastAPI.
"""

from __future__ import annotations

from .routes import classify_image_api, health_check
from .model import classify_waste
from .fastapi_app import app

__all__ = [
    "classify_image_api",
    "health_check",
    "classify_waste",
    "app",
]
