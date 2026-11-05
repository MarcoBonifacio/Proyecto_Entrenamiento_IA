"""Módulo del modelo de clasificación de residuos EcoClasifica.

Encapsula la carga del modelo ResNet-50 (base o afinado), la inferencia
y la generación de predicciones con probabilidades por categoría.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ResNetForImageClassification,
)

from .preprocess import ImageInput, load_and_preprocess_image

logger = logging.getLogger(__name__)

BASE_MODEL_NAME: str = "microsoft/resnet-50"
DEFAULT_FINETUNED_PATH: str = "./models/ecoclasifica-resnet50"
NUM_CLASSES: int = 6


class EcoClasificaModel:
    """Modelo de clasificación de residuos basado en ResNet-50.

    Attributes
    ----------
    LABELS : dict[int, str]
        Mapeo de índices a nombres de categoría en español.
    model : AutoModelForImageClassification
        Modelo de HuggingFace cargado y listo para inferencia.
    processor : AutoImageProcessor
        Procesador de imágenes asociado al modelo.
    device : torch.device
        Dispositivo de ejecución (CPU o CUDA).
    is_finetuned : bool
        Indica si el modelo cargado es el afinado o el base.
    """

    LABELS: dict[int, str] = {
        0: "Cartón",
        1: "Vidrio",
        2: "Metal",
        3: "Papel",
        4: "Plástico",
        5: "Basura general",
    }

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Inicializa el modelo de clasificación.

        Parameters
        ----------
        model_path : str | None
            Ruta al checkpoint del modelo afinado. Si es ``None`` o la
            ruta no existe, se carga el modelo base ResNet-50 y se
            reconfigura la cabeza de clasificación para 6 clases.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_finetuned: bool = False
        self.model: AutoModelForImageClassification
        self.processor: AutoImageProcessor

        if model_path is not None and Path(model_path).exists():
            self._load_finetuned(model_path)
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
        """Carga un modelo afinado desde disco.

        Parameters
        ----------
        model_path : str
            Ruta al directorio del checkpoint afinado.
        """
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
        """Carga (o recarga) un checkpoint afinado.

        Parameters
        ----------
        model_path : str
            Ruta al directorio del checkpoint.
        """
        self._load_finetuned(model_path)

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image_input: ImageInput) -> dict:
        """Clasifica una imagen de residuo.

        Parameters
        ----------
        image_input : ImageInput
            Ruta, ``PIL.Image.Image`` o ``numpy.ndarray``.

        Returns
        -------
        dict
            Diccionario con las claves:

            * ``"categoria"`` — nombre en español de la clase predicha.
            * ``"confianza"`` — probabilidad de la clase predicha (0–1).
            * ``"todas_las_predicciones"`` — lista ordenada de
              tuplas ``(categoría, probabilidad)`` de mayor a menor.
        """
        try:
            # Preprocesar la imagen
            inputs = load_and_preprocess_image(image_input)
            # Mover tensores al dispositivo del modelo
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inferencia
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, num_classes)

            # Aplicar softmax para obtener probabilidades
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probs_np: np.ndarray = probabilities.cpu().numpy()[0]

            # Construir lista de predicciones ordenadas
            all_predictions: list[tuple[str, float]] = []
            for idx in range(len(probs_np)):
                label = self.LABELS.get(idx, f"Clase {idx}")
                all_predictions.append((label, round(float(probs_np[idx]), 4)))

            all_predictions.sort(key=lambda x: x[1], reverse=True)

            top_label, top_confidence = all_predictions[0]

            return {
                "categoria": top_label,
                "confianza": round(top_confidence, 4),
                "todas_las_predicciones": all_predictions,
            }

        except (TypeError, FileNotFoundError, ValueError):
            # Re-lanzar errores de validación tal cual
            raise
        except Exception as exc:
            logger.error("Error durante la predicción: %s", exc)
            raise RuntimeError(
                f"Error al realizar la clasificación de la imagen: {exc}"
            ) from exc


# ----------------------------------------------------------------------
# Función de conveniencia
# ----------------------------------------------------------------------


def classify_waste(
    image_input: ImageInput,
    model_path: Optional[str] = None,
) -> dict:
    """Clasifica una imagen de residuo de forma directa.

    Función de conveniencia que instancia ``EcoClasificaModel`` y ejecuta
    la predicción en un solo paso.

    Parameters
    ----------
    image_input : ImageInput
        Ruta, ``PIL.Image.Image`` o ``numpy.ndarray``.
    model_path : str | None
        Ruta al checkpoint afinado (opcional).

    Returns
    -------
    dict
        Resultado de la predicción (ver :meth:`EcoClasificaModel.predict`).
    """
    model = EcoClasificaModel(model_path=model_path)
    return model.predict(image_input)


# Constantes UPPER_CASE (Refactor sin magic numbers)
DEFAULT_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60
MAX_BATCH_SIZE = 32
