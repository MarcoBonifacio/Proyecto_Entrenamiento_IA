"""API REST de EcoClasifica con FastAPI.

Expone el modelo ResNet-50 para ser consumido por el frontend en React.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import EcoClasificaModel  # noqa: E402
from src.postprocess import create_results_summary  # noqa: E402

logger = logging.getLogger(__name__)

FINETUNED_MODEL_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "ecoclasifica-resnet50",
)

app = FastAPI(title="EcoClasifica API", version="2.0.0")

# Permitir peticiones desde el servidor de desarrollo de Vite (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: EcoClasificaModel | None = None


def _get_model() -> EcoClasificaModel:
    """Obtiene o inicializa el modelo singleton."""
    global _model
    if _model is None:
        if Path(FINETUNED_MODEL_PATH).exists():
            logger.info("Cargando modelo afinado desde '%s'", FINETUNED_MODEL_PATH)
            _model = EcoClasificaModel(model_path=FINETUNED_MODEL_PATH)
        else:
            logger.warning("Modelo afinado no encontrado. Usando ResNet-50 base.")
            _model = EcoClasificaModel(model_path=None)
    return _model


@app.on_event("startup")
def startup_event() -> None:
    """Pre-carga el modelo al iniciar el servidor."""
    _get_model()


@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)) -> dict:
    """Recibe una imagen y devuelve la categoría, confianza y recomendación."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model = _get_model()
        prediction = model.predict(image)
        summary = create_results_summary(prediction)

        return {
            "prediction": summary["categoria"],
            "confidence": float(summary["confianza"]),
            "disposal_recommendation": summary["recomendacion"],
        }
    except Exception as exc:
        logger.error("Error al clasificar la imagen: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error interno: {exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
    