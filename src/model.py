"""Módulo del modelo de clasificación de residuos EcoClasifica.

Encapsula la carga del modelo ResNet-50 (base o afinado), la inferencia
con Test-Time Augmentation (TTA) y la generación de predicciones por categoría.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ResNetForImageClassification,
)

from .preprocess import ImageInput, load_and_preprocess_image

logger = logging.getLogger(__name__)

# Constantes del modelo de clasificación (sin magic numbers)
BASE_MODEL_NAME: str = "microsoft/resnet-50"
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_FINETUNED_PATH: str = os.environ.get(
    "ECOCLASIFICA_MODEL_PATH",
    str(_PROJECT_ROOT / "models" / "ecoclasifica-resnet50"),
)
NUM_CLASSES: int = 6
ROUND_DECIMALS: int = 4
DEFAULT_IMAGE_SIZE: int = 224
CONFIDENCE_THRESHOLD: float = 0.60
MAX_BATCH_SIZE: int = 32


class EcoClasificaModel:
    """Modelo de clasificación de residuos basado en ResNet-50."""

    LABELS: dict[int, str] = {
        0: "Cartón",
        1: "Vidrio",
        2: "Metal",
        3: "Papel",
        4: "Plástico",
        5: "Basura general",
    }

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_finetuned: bool = False
        self.model: AutoModelForImageClassification
        self.processor: AutoImageProcessor

        target_path = model_path if model_path is not None else DEFAULT_FINETUNED_PATH

        if target_path is not None and Path(target_path).exists():
            self._load_finetuned(target_path)
        else:
            if model_path is not None:
                logger.warning(
                    "No se encontró el modelo afinado en '%s'. "
                    "Se cargará el modelo base ResNet-50.",
                    model_path,
                )
            self._load_base_model()

    # ------------------------------------------------------------------
    # Métodos de carga
    # ------------------------------------------------------------------

    def _load_base_model(self) -> None:
        """Carga el modelo base ResNet-50 y reconfigura la cabeza para 6 clases."""
        logger.info("Cargando modelo base ResNet-50 desde '%s'…", BASE_MODEL_NAME)
        try:
            self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL_NAME)
            self.model = ResNetForImageClassification.from_pretrained(
                BASE_MODEL_NAME,
                num_labels=NUM_CLASSES,
                ignore_mismatched_sizes=True,
            )
            self.model.config.id2label = {i: lbl for i, lbl in self.LABELS.items()}
            self.model.config.label2id = {lbl: i for i, lbl in self.LABELS.items()}
            self.model.to(self.device)
            self.model.eval()
            self.is_finetuned = False
            logger.info(
                "Modelo base cargado exitosamente en %s (6 clases configuradas).",
                self.device,
            )
        except Exception as exc:
            logger.error("Error al cargar el modelo base: %s", exc)
            raise RuntimeError(
                f"No se pudo cargar el modelo base ResNet-50: {exc}"
            ) from exc

    def _load_finetuned(self, model_path: str) -> None:
        """Carga un modelo afinado desde disco."""
        logger.info("Cargando modelo afinado desde '%s'…", model_path)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.is_finetuned = True
            logger.info(
                "Modelo afinado cargado exitosamente desde '%s' en %s.",
                model_path,
                self.device,
            )
        except Exception as exc:
            logger.warning(
                "Error al cargar el modelo afinado ('%s'): %s. "
                "Se usará el modelo base como respaldo.",
                model_path,
                exc,
            )
            self._load_base_model()

    def load_model(self, model_path: str) -> None:
        """Carga (o recarga) un checkpoint afinado."""
        self._load_finetuned(model_path)

    # ------------------------------------------------------------------
    # Generación de vistas para Test-Time Augmentation (TTA)
    # ------------------------------------------------------------------

    def _build_tta_images(self, pil_img: Image.Image) -> list[Image.Image]:
        """Genera 3 vistas complementarias para estabilizar la predicción de cámara."""
        img_rgb = pil_img.convert("RGB")
        img_flipped = ImageOps.mirror(img_rgb)

        # Vista 3: realce suave de nitidez y contraste para bordes de plástico transparente/blanco
        img_enhanced = ImageEnhance.Sharpness(img_rgb).enhance(1.25)
        img_enhanced = ImageEnhance.Contrast(img_enhanced).enhance(1.10)

        return [img_rgb, img_flipped, img_enhanced]

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image_input: ImageInput) -> dict:
        """Clasifica una imagen de residuo usando Test-Time Augmentation (TTA)."""
        try:
            # Si recibimos un PIL.Image directamente (ej. desde FastAPI), aplicamos TTA
            if isinstance(image_input, Image.Image):
                vistas = self._build_tta_images(image_input)
                inputs = self.processor(images=vistas, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                # Promediar probabilidades de las 3 vistas (batch_size=3 -> 1)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).mean(dim=0)
                probs_np: np.ndarray = probabilities.cpu().numpy()
            else:
                # Respaldo estándar para rutas o arreglos numpy
                inputs = load_and_preprocess_image(image_input)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs_np = probabilities.cpu().numpy()[0]

            # Construir lista de predicciones ordenadas
            all_predictions: list[tuple[str, float]] = []
            for idx in range(len(probs_np)):
                label = self.LABELS.get(idx, f"Clase {idx}")
                all_predictions.append((label, round(float(probs_np[idx]), ROUND_DECIMALS)))

            all_predictions.sort(key=lambda x: x[1], reverse=True)

            top_label, top_confidence = all_predictions[0]

            return {
                "categoria": top_label,
                "confianza": round(top_confidence, ROUND_DECIMALS),
                "todas_las_predicciones": all_predictions,
            }

        except (TypeError, FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            logger.error("Error durante la predicción: %s", exc)
            raise RuntimeError(
                f"Error al realizar la clasificación de la imagen: {exc}"
            ) from exc


# ----------------------------------------------------------------------
# Función de conveniencia
# ----------------------------------------------------------------------

_MODEL_CACHE: dict[str, "EcoClasificaModel"] = {}


def classify_waste(
    image_input: ImageInput,
    model_path: Optional[str] = None,
) -> dict:
    """Clasifica una imagen de residuo de forma directa."""
    resolved_path = model_path if model_path is not None else DEFAULT_FINETUNED_PATH
    key = str(resolved_path)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = EcoClasificaModel(model_path=resolved_path)
        _MODEL_CACHE[key] = model
    return model.predict(image_input)
