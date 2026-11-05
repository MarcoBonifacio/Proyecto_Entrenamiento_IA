"""Módulo de preprocesamiento de imágenes para EcoClasifica.

Proporciona funciones para validar, cargar y preprocesar imágenes
antes de pasarlas al modelo de clasificación de residuos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor

# Formatos de imagen aceptados
VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE: tuple[int, int] = (224, 224)
MODEL_NAME: str = "microsoft/resnet-50"

logger = logging.getLogger(__name__)

# Tipo de entrada flexible para imágenes
ImageInput = Union[str, Path, Image.Image, np.ndarray]


def _load_image_from_input(image_input: ImageInput) -> Image.Image:
    """Convierte distintos tipos de entrada en un objeto PIL Image.

    Parameters
    ----------
    image_input : ImageInput
        Ruta de archivo (``str`` o ``Path``), ``PIL.Image.Image`` o
        ``numpy.ndarray`` con datos de imagen.

    Returns
    -------
    Image.Image
        Imagen en modo RGB lista para redimensionar.

    Raises
    ------
    TypeError
        Si el tipo de entrada no es soportado.
    FileNotFoundError
        Si la ruta proporcionada no existe.
    ValueError
        Si el formato del archivo no es válido.
    """
    if isinstance(image_input, np.ndarray):
        try:
            img = Image.fromarray(image_input)
            return img.convert("RGB")
        except Exception as exc:
            raise ValueError(
                "Error al convertir el arreglo NumPy a imagen. "
                "Asegúrese de que el arreglo tenga forma (alto, ancho, canales)."
            ) from exc

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de imagen: '{path}'"
            )
        suffix = path.suffix.lower()
        if suffix not in VALID_EXTENSIONS:
            raise ValueError(
                f"Formato de imagen no soportado: '{suffix}'. "
                f"Formatos válidos: {', '.join(sorted(VALID_EXTENSIONS))}"
            )
        try:
            img = Image.open(path)
            img.load()  # Fuerza la carga completa del archivo
            return img.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"El archivo '{path.name}' no es una imagen válida o está corrupto."
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"Error al abrir la imagen '{path.name}': {exc}"
            ) from exc

    raise TypeError(
        f"Tipo de entrada no soportado: {type(image_input).__name__}. "
        "Se acepta str, Path, PIL.Image.Image o numpy.ndarray."
    )


def validate_image(image_input: ImageInput) -> bool:
    """Valida si la entrada proporcionada es una imagen procesable.

    Parameters
    ----------
    image_input : ImageInput
        Entrada a validar (ruta, PIL Image o arreglo NumPy).

    Returns
    -------
    bool
        ``True`` si la imagen es válida, ``False`` en caso contrario.
    """
    try:
        if image_input is None:
            return False
        img = _load_image_from_input(image_input)
        width, height = img.size
        return width >= 1 and height >= 1
    except (TypeError, FileNotFoundError, ValueError):
        return False
    except Exception:  # pragma: no cover — errores inesperados
        logger.exception("Error inesperado durante la validación de la imagen.")
        return False


def load_and_preprocess_image(image_input: ImageInput) -> dict:
    """Carga, valida y preprocesa una imagen para el modelo ResNet-50.

    La imagen se redimensiona a 224×224 y se transforma mediante el
    ``AutoImageProcessor`` de HuggingFace para ResNet-50.

    Parameters
    ----------
    image_input : ImageInput
        Ruta de archivo, ``PIL.Image.Image`` o ``numpy.ndarray``.

    Returns
    -------
    dict
        Diccionario con tensores listos para el modelo (clave
        ``"pixel_values"``), tal como lo devuelve el procesador de
        imágenes de HuggingFace.

    Raises
    ------
    TypeError
        Si el tipo de entrada no es soportado.
    FileNotFoundError
        Si la ruta no existe.
    ValueError
        Si la imagen no es válida o tiene un formato no soportado.
    RuntimeError
        Si ocurre un error durante el preprocesamiento con el modelo.
    """
    # 1. Cargar y convertir a PIL Image RGB
    img = _load_image_from_input(image_input)

    # 2. Redimensionar a 224×224
    img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    # 3. Procesar con AutoImageProcessor de ResNet-50
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        inputs = processor(images=img, return_tensors="pt")
        logger.info(
            "Imagen preprocesada exitosamente — forma del tensor: %s",
            inputs["pixel_values"].shape,
        )
        return inputs
    except Exception as exc:
        raise RuntimeError(
            "Error al preprocesar la imagen con el procesador del modelo. "
            f"Detalle: {exc}"
        ) from exc


# Constantes UPPER_CASE (Refactor sin magic numbers)
DEFAULT_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60
MAX_BATCH_SIZE = 32
