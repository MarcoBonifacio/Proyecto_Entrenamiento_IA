"""Servidor REST API con FastAPI para EcoClasifica.

Expone endpoints para la clasificación de residuos sólidos mediante Deep Learning (ResNet-50),
validaciones estrictas de tipos y formatos de archivo, health-checks y documentación OpenAPI en /docs.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from .routes import classify_image_api, health_check

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuración y Constantes de Validación de Archivos
# ==============================================================================
# Tipos MIME permitidos para las imágenes de entrada
MIME_TYPES_PERMITIDOS: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

# Extensiones de archivo permitidas (en minúsculas)
EXTENSIONES_PERMITIDAS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

# Formatos internos validados por Pillow mediante Magic Bytes
FORMATOS_PIL_PERMITIDOS: Set[str] = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
}

# Límites de tamaño y dimensiones para seguridad y rendimiento
TAMANO_MAXIMO_BYTES: int = 10 * 1024 * 1024          # 10 Megabytes
TAMANO_MAXIMO_BASE64_CHARS: int = 15 * 1024 * 1024  # ~15 MB de cadena base64
DIMENSION_MINIMA: Tuple[int, int] = (10, 10)         # 10x10 píxeles
DIMENSION_MAXIMA: Tuple[int, int] = (8192, 8192)     # Protección contra bombas de descompresión


# ==============================================================================
# Metadatos de la Aplicación FastAPI
# ==============================================================================
app = FastAPI(
    title="EcoClasifica API",
    description=(
        "API REST para la clasificación inteligente de residuos sólidos usando Deep Learning (ResNet-50).\n\n"
        "Incluye validación estricta de tipos de archivo (MIME, extensión, magic bytes y dimensiones).\n\n"
        "Desarrollado para el proyecto EcoClasifica — Universidad Continental."
    ),
    version="1.1.0",
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


# ==============================================================================
# Schemas Pydantic (Modelos de Datos)
# ==============================================================================
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    servicio: str = Field(..., example="EcoClasifica API")
    version: str = Field(..., example="1.1.0")
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
        description="Cadena de imagen codificada en base64 (con o sin encabezado data:image/...;base64,)",
        example="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


# ==============================================================================
# Funciones de Validación Estricta de Archivos
# ==============================================================================
def validar_tipo_mime(content_type: Optional[str]) -> None:
    """Valida el encabezado Content-Type contra la lista blanca permitida.

    Parameters
    ----------
    content_type : Optional[str]
        Tipo MIME reportado en la petición HTTP.

    Raises
    ------
    HTTPException
        Si el tipo MIME está ausente o no está permitido (Código 400).
    """
    if not content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Encabezado Content-Type no proporcionado. Debe ser un tipo de imagen válido.",
        )

    mime_limpio = content_type.lower().split(";")[0].strip()
    if mime_limpio not in MIME_TYPES_PERMITIDOS:
        mimes_str = ", ".join(sorted(MIME_TYPES_PERMITIDOS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Tipo de archivo no permitido: '{content_type}'. "
                f"Formatos aceptados: {mimes_str}."
            ),
        )


def validar_extension_archivo(nombre_archivo: Optional[str]) -> None:
    """Valida que la extensión del archivo pertenezca a la lista blanca permitida.

    Parameters
    ----------
    nombre_archivo : Optional[str]
        Nombre original del archivo proporcionado por el cliente.

    Raises
    ------
    HTTPException
        Si la extensión no está permitida (Código 400).
    """
    if not nombre_archivo:
        return

    extension = Path(nombre_archivo).suffix.lower()
    if extension and extension not in EXTENSIONES_PERMITIDAS:
        formatos_str = ", ".join(sorted(EXTENSIONES_PERMITIDAS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Extensión de archivo no permitida: '{extension}'. "
                f"Las extensiones admitidas son: {formatos_str}."
            ),
        )


def validar_tamano_bytes(contenido: bytes) -> None:
    """Valida que el archivo no esté vacío y no supere el límite de memoria configurado.

    Parameters
    ----------
    contenido : bytes
        Bytes en memoria del archivo subido.

    Raises
    ------
    HTTPException
        Si el archivo está vacío (400) o si supera el límite de 10 MB (413).
    """
    if not contenido:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo enviado está vacío.",
        )

    tamano = len(contenido)
    if tamano > TAMANO_MAXIMO_BYTES:
        mb_recibido = round(tamano / (1024 * 1024), 2)
        mb_maximo = round(TAMANO_MAXIMO_BYTES / (1024 * 1024), 2)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"El archivo supera el tamaño máximo permitido de {mb_maximo} MB "
                f"(recibido: {mb_recibido} MB)."
            ),
        )


def validar_y_procesar_imagen(
    contenido: bytes,
    nombre_archivo: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Image.Image:
    """Aplica el pipeline completo de validación e inspección profunda de bytes.

    Inspecciona tipo MIME, extensión, tamaño, magic bytes reales y dimensiones de resolución.

    Parameters
    ----------
    contenido : bytes
        Bytes crudos de la imagen.
    nombre_archivo : Optional[str]
        Nombre original opcional del archivo.
    content_type : Optional[str]
        Tipo MIME opcional.

    Returns
    -------
    Image.Image
        Objeto PIL Image validado y convertido a RGB.

    Raises
    ------
    HTTPException
        Si alguna de las validaciones de seguridad o formato falla.
    """
    # 1. Validación de tamaño (vacío o límite máximo de bytes)
    validar_tamano_bytes(contenido)

    # 2. Validación de tipo MIME reportado
    if content_type:
        validar_tipo_mime(content_type)

    # 3. Validación de extensión de archivo
    if nombre_archivo:
        validar_extension_archivo(nombre_archivo)

    # 4. Inspección profunda de Magic Bytes e integridad estructural con Pillow
    try:
        # Primer paso: verificar formato real y magic bytes
        flujo_inicial = io.BytesIO(contenido)
        with Image.open(flujo_inicial) as img_temp:
            formato_real = img_temp.format
            if not formato_real or formato_real.upper() not in FORMATOS_PIL_PERMITIDOS:
                formatos_str = ", ".join(sorted(FORMATOS_PIL_PERMITIDOS))
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"El formato interno de la imagen ('{formato_real}') no está permitido "
                        f"o ha sido suplantado. Formatos válidos: {formatos_str}."
                    ),
                )
            img_temp.verify()

        # Segundo paso: decodificar la imagen y verificar resolución
        imagen = Image.open(io.BytesIO(contenido))
        ancho, alto = imagen.size

        if ancho < DIMENSION_MINIMA[0] or alto < DIMENSION_MINIMA[1]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Las dimensiones de la imagen ({ancho}x{alto}) son inferiores al mínimo "
                    f"requerido de {DIMENSION_MINIMA[0]}x{DIMENSION_MINIMA[1]} píxeles."
                ),
            )

        if ancho > DIMENSION_MAXIMA[0] or alto > DIMENSION_MAXIMA[1]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Las dimensiones de la imagen ({ancho}x{alto}) superan el límite máximo "
                    f"permitido de {DIMENSION_MAXIMA[0]}x{DIMENSION_MAXIMA[1]} píxeles."
                ),
            )

        # Conversión garantizada a RGB para el modelo ResNet-50
        return imagen.convert("RGB")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error al validar/decodificar imagen: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No se pudo decodificar el archivo como imagen válida: {exc}",
        )


# ==============================================================================
# Endpoints de la API
# ==============================================================================
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
        "formatos_permitidos": sorted(list(MIME_TYPES_PERMITIDOS)),
        "tamano_maximo_mb": TAMANO_MAXIMO_BYTES // (1024 * 1024),
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
    detalles_clases = [
        {
            "id": 0,
            "categoria": "Cartón",
            "emoji": "📦",
            "consejo": "Dobla y aplana las cajas; deposítalas en el contenedor azul sin restos de grasa.",
        },
        {
            "id": 1,
            "categoria": "Vidrio",
            "emoji": "🫙",
            "consejo": "Retira tapas o corchos; deposítalo en el contenedor verde sin romperlo.",
        },
        {
            "id": 2,
            "categoria": "Metal",
            "emoji": "🥫",
            "consejo": "Enjuaga y escurre latas o envases metálicos antes de depositarlos en su contenedor.",
        },
        {
            "id": 3,
            "categoria": "Papel",
            "emoji": "📄",
            "consejo": "Evita arrugarlo excesivamente y mantenlo seco; recíclalo en el contenedor azul.",
        },
        {
            "id": 4,
            "categoria": "Plástico",
            "emoji": "♻️",
            "consejo": "Enjuaga, aplana para reducir volumen y coloca la tapa en el contenedor amarillo.",
        },
        {
            "id": 5,
            "categoria": "Basura general",
            "emoji": "🗑️",
            "consejo": "Residuos no reciclables o contaminados; deposítalos en el contenedor de desechos generales.",
        },
    ]
    return {
        "total_clases": len(detalles_clases),
        "clases": detalles_clases,
    }


@app.post(
    "/classify",
    response_model=ClassificationResult,
    summary="Clasificar residuo mediante subida de archivo de imagen",
    tags=["Clasificación"],
)
async def classify_image_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Recibe una imagen (JPEG, PNG, WEBP, BMP) mediante multipart/form-data.

    Aplica validaciones estrictas de tamaño, tipo MIME, extensión y magic bytes.
    Ejecuta el modelo de Deep Learning y devuelve la categoría, confianza y recomendación.
    """
    contenido = await file.read()
    imagen = validar_y_procesar_imagen(
        contenido=contenido,
        nombre_archivo=file.filename,
        content_type=file.content_type,
    )

    resultado = classify_image_api(imagen)
    if resultado.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=resultado.get("mensaje", "Error interno al clasificar la imagen."),
        )

    return resultado


@app.post(
    "/predict",
    response_model=ClassificationResult,
    summary="Alias de /classify para compatibilidad",
    tags=["Clasificación"],
)
async def predict_alias(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Alias idéntico a /classify para permitir máxima compatibilidad a clientes REST."""
    return await classify_image_file(file)


@app.post(
    "/classify/base64",
    response_model=ClassificationResult,
    summary="Clasificar residuo a partir de una cadena base64",
    tags=["Clasificación"],
)
def classify_base64_image(payload: Base64ClassificationRequest) -> Dict[str, Any]:
    """Permite enviar una imagen codificada en base64 en un cuerpo JSON.

    Valida tamaño de la cadena, decodificación segura, magic bytes y dimensiones.
    """
    cadena_b64 = payload.image_base64.strip()
    if not cadena_b64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La cadena base64 proporcionada está vacía.",
        )

    if len(cadena_b64) > TAMANO_MAXIMO_BASE64_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="La cadena base64 supera el tamaño máximo permitido de 15 MB.",
        )

    if "," in cadena_b64:
        cadena_b64 = cadena_b64.split(",", 1)[1]

    try:
        bytes_imagen = base64.b64decode(cadena_b64, validate=True)
    except Exception as exc:
        logger.error("Error al decodificar base64: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"La cadena base64 proporcionada es inválida o está corrupta: {exc}",
        )

    # Validar que los bytes correspondan a una imagen auténtica
    imagen = validar_y_procesar_imagen(
        contenido=bytes_imagen,
        nombre_archivo=None,
        content_type=None,
    )

    resultado = classify_image_api(imagen)
    if resultado.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=resultado.get("mensaje", "Error interno al clasificar la imagen."),
        )

    return resultado

# Constantes UPPER_CASE (Refactor sin magic numbers)
DEFAULT_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60
MAX_BATCH_SIZE = 32
