"""Servidor REST API con FastAPI para EcoClasifica.

Expone endpoints para la clasificación de residuos sólidos mediante Deep Learning (ResNet-50),
health-checks y documentación interactiva OpenAPI/Swagger en /docs.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from .routes import classify_image_api, health_check
from .model import EcoClasificaModel, classify_waste

logger = logging.getLogger(__name__)

# Metadatos de la aplicación
app = FastAPI(
    title="EcoClasifica API",
    description=(
        "API REST para la clasificación inteligente de residuos sólidos usando Deep Learning (ResNet-50).\n\n"
        "Desarrollado para el proyecto EcoClasifica - Universidad Continental."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Habilitar CORS para permitir consumo desde el Frontend (React, Vue, Gradio, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Schemas Pydantic (Modelos de Datos)
# ==========================================
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    servicio: str = Field(..., example="EcoClasifica API")
    version: str = Field(..., example="1.0.0")
    modelo_afinado_disponible: bool = Field(..., example=False)
    uptime_seconds: float = Field(..., example=12.34)


class ClassItem(BaseModel):
    id: int = Field(..., example=0)
    categoria: str = Field(..., example="Cartón")
    emoji: str = Field(..., example="📦")
    consejo: str = Field(..., example="Dobla las cajas y mantenlas secas para reciclar.")


class ClassesListResponse(BaseModel):
    total_clases: int = Field(..., example=6)
    clases: List[ClassItem]


class ClassificationResult(BaseModel):
    error: bool = Field(..., example=False)
    categoria: Optional[str] = Field(None, example="Metal")
    confianza: Optional[float] = Field(None, example=0.92)
    emoji: Optional[str] = Field(None, example="🥫")
    recomendacion: Optional[str] = Field(None, example="Enjuagar las latas antes de reciclarlas.")
    todas_las_predicciones: Optional[Dict[str, float]] = Field(None, example={
        "Cartón": 0.02,
        "Vidrio": 0.03,
        "Metal": 0.92,
        "Papel": 0.01,
        "Plástico": 0.01,
        "Basura general": 0.01,
    })
    modelo_afinado: Optional[bool] = Field(None, example=False)
    mensaje: Optional[str] = Field(None, example=None)


class Base64ClassificationRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Cadena de la imagen codificada en base64 (con o sin prefijo data:image/...)",
        example="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


# ==========================================
# Endpoints
# ==========================================
@app.get(
    "/",
    summary="Raíz del servicio",
    tags=["General"],
)
def root() -> Dict[str, Any]:
    """Retorna información general de la API y enlaces útiles a la documentación."""
    return {
        "mensaje": "Bienvenido a la API REST de EcoClasifica ♻️",
        "documentacion_swagger": "/docs",
        "documentacion_redoc": "/redoc",
        "health_check": "/health",
        "clases_disponibles": "/classes",
        "clasificar_imagen": "/classify (POST multipart/form-data o /classify/base64)",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificación del estado del servicio",
    tags=["Salud"],
)
def get_health() -> Dict[str, Any]:
    """Endpoint de salud que indica si el servicio y el modelo afinado están operativos."""
    return health_check()


@app.get(
    "/classes",
    response_model=ClassesListResponse,
    summary="Listado de categorías de residuos",
    tags=["Información"],
)
def get_classes() -> Dict[str, Any]:
    """Retorna el listado de las 6 categorías soportadas por el modelo EcoClasifica con sus emojis."""
    class_details = [
        {"id": 0, "categoria": "Cartón", "emoji": "📦", "consejo": "Dobla y aplana las cajas; deposítalas en el contenedor azul sin restos de grasa."},
        {"id": 1, "categoria": "Vidrio", "emoji": "🫙", "consejo": "Retira tapas o corchos; deposítalo en el contenedor verde sin romperlo."},
        {"id": 2, "categoria": "Metal", "emoji": "🥫", "consejo": "Enjuaga y escurre latas o envases metálicos antes de depositarlos en el contenedor correspondiente."},
        {"id": 3, "categoria": "Papel", "emoji": "📄", "consejo": "Evita arrugarlo excesivamente y mantenlo seco; recíclalo en el contenedor azul."},
        {"id": 4, "categoria": "Plástico", "emoji": "♻️", "consejo": "Enjuaga, aplana para reducir volumen y coloca la tapa en el contenedor amarillo."},
        {"id": 5, "categoria": "Basura general", "emoji": "🗑️", "consejo": "Residuos no reciclables o contaminados; deposítalos en el contenedor de desechos generales."},
    ]
    return {
        "total_clases": len(class_details),
        "clases": class_details,
    }


@app.post(
    "/classify",
    response_model=ClassificationResult,
    summary="Clasificar residuo mediante subida de archivo de imagen",
    tags=["Clasificación"],
)
async def classify_image_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Recibe una imagen (JPEG, PNG, WEBP, etc.) mediante multipart/form-data,

    ejecuta el modelo de Deep Learning y devuelve la categoría predicha, confianza y recomendaciones.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de archivo no permitido: '{file.content_type}'. Debe ser una imagen (image/*).",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo enviado está vacío.",
            )

        image = Image.open(io.BytesIO(content)).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error al procesar la imagen subida: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No se pudo decodificar el archivo como imagen: {exc}",
        )

    result = classify_image_api(image)
    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("mensaje", "Error interno al clasificar la imagen."),
        )

    return result


@app.post(
    "/predict",
    response_model=ClassificationResult,
    summary="Alias de /classify para compatibilidad",
    tags=["Clasificación"],
)
async def predict_alias(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Alias idéntico a /classify para permitir flexibilidad a clientes REST."""
    return await classify_image_file(file)


@app.post(
    "/classify/base64",
    response_model=ClassificationResult,
    summary="Clasificar residuo a partir de una cadena base64",
    tags=["Clasificación"],
)
def classify_base64_image(payload: Base64ClassificationRequest) -> Dict[str, Any]:
    """Permite enviar una imagen codificada en base64 en un cuerpo JSON.

    Útil para aplicaciones móviles o frontend sin FormData.
    """
    raw_b64 = payload.image_base64.strip()
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.error("Error al decodificar base64: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"La cadena base64 proporcionada es inválida: {exc}",
        )

    result = classify_image_api(image)
    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("mensaje", "Error interno al clasificar la imagen."),
        )

    return result
