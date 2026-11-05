"""Módulo de rutas API para EcoClasifica.

Proporciona funciones wrapper que pueden integrarse con un framework
web como FastAPI o Flask para exponer la clasificación como servicio.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Any

from .postprocess import create_results_summary

logger = logging.getLogger(__name__)

FINETUNED_MODEL_PATH = "./models/ecoclasifica-resnet50"
_APP_START_TIME = time.time()


def health_check() -> dict:
    """Endpoint de salud del servicio.

    Returns
    -------
    dict
        Estado del servicio con información del modelo.
    """
    model_available = Path(FINETUNED_MODEL_PATH).exists()
    uptime_seconds = round(time.time() - _APP_START_TIME, 2)

    return {
        "status": "ok",
        "servicio": "EcoClasifica API",
        "version": "1.0.0",
        "modelo_afinado_disponible": model_available,
        "uptime_seconds": uptime_seconds,
    }


def classify_image_api(
    image_input: Any,
    model_path: Optional[str] = None,
) -> dict:
    """Clasifica una imagen de residuo y devuelve un resultado JSON.

    Parameters
    ----------
    image_input : Any
        Ruta al archivo, objeto PIL.Image o arreglo NumPy.
    model_path : str | None
        Ruta opcional al modelo afinado. Si no se proporciona, se
        busca en la ruta por defecto.

    Returns
    -------
    dict
        Resultado de la clasificación con predicción, recomendación
        y metadatos.
    """
    try:
        if image_input is None:
            return {
                "error": True,
                "mensaje": "No se proporcionó ninguna imagen de entrada.",
            }

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return {
                    "error": True,
                    "mensaje": f"Archivo no encontrado: '{image_input}'",
                }

        _model_path = model_path or FINETUNED_MODEL_PATH
        from . import api
        prediction = api.classify_waste(image_input, model_path=_model_path)
        summary = create_results_summary(prediction)

        return {
            "error": False,
            "categoria": summary["categoria"],
            "confianza": summary["confianza"],
            "emoji": summary["emoji"],
            "recomendacion": summary["recomendacion"],
            "todas_las_predicciones": prediction["todas_las_predicciones"],
            "modelo_afinado": Path(_model_path).exists(),
        }

    except Exception as exc:
        logger.error("Error en classify_image_api: %s", exc)
        return {
            "error": True,
            "mensaje": f"Error al clasificar la imagen: {exc}",
        }

# Constantes UPPER_CASE (Refactor sin magic numbers)
DEFAULT_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60
MAX_BATCH_SIZE = 32
