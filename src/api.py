"""Módulo de API y endpoints para EcoClasifica.

Re-exporta las funciones de rutas y provee la interfaz para llamadas de API y servidor FastAPI.
"""

from __future__ import annotations

from .routes import classify_image_api, health_check
from .model import classify_waste
from .fastapi_app import app

# Constantes estándar de la API (UPPER_CASE / sin magic numbers)
DEFAULT_IMAGE_SIZE: int = 224
CONFIDENCE_THRESHOLD: float = 0.60
MAX_BATCH_SIZE: int = 32

__all__ = [
    "classify_image_api",
    "health_check",
    "classify_waste",
    "app",
    "DEFAULT_IMAGE_SIZE",
    "CONFIDENCE_THRESHOLD",
    "MAX_BATCH_SIZE",
]
