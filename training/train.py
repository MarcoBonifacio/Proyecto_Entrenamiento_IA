"""
training/train.py — Script principal de entrenamiento para EcoClasifica.

Ajuste fino (fine-tuning) de ResNet-50 sobre el dataset TrashBusters/combinedd
para clasificación de residuos en 6 categorías.

Uso:
    python training/train.py
    python training/train.py --epochs 10 --batch_size 32 --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Asegurar que la raíz del proyecto esté en sys.path (ejecución standalone)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecoclasifica.train")

# ---------------------------------------------------------------------------
# Constantes de configuración de entrenamiento (sin magic numbers)
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_SIZE: int = 224
CONFIDENCE_THRESHOLD: float = 0.60
MAX_BATCH_SIZE: int = 32

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPT_DIR / "config.json"
_PROJECT_ROOT = _SCRIPT_DIR.parent


def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Carga la configuración desde el archivo JSON."""
    logger.info("Cargando configuración desde %s", path)
    with open(path, encoding="utf-8") as fh:
        config: dict[str, Any] = json.load(fh)
    return config


def parse_args() -> argparse.Namespace:
    """Analiza argumentos opcionales de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de EcoClasifica (ResNet-50 fine-tuning)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de épocas de entrenamiento (sobreescribe config.json).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Tamaño de lote (sobreescribe config.json).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Tasa de aprendizaje (sobreescribe config.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directorio de salida del modelo (sobreescribe config.json).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preparación del dataset
# ---------------------------------------------------------------------------

def download_dataset(dataset_name: str) -> Dataset:
    """Descarga el dataset desde HuggingFace con manejo de errores."""
    logger.info("Descargando dataset '%s' desde HuggingFace …", dataset_name)
    try:
        ds = load_dataset(dataset_name, split="train")
        logger.info("Dataset descargado correctamente — %d registros.", len(ds))
        return ds
    except Exception as exc:
        logger.error(
            "Error al descargar el dataset '%s': %s",
            dataset_name,
            exc,
        )
        raise SystemExit(
            f"No se pudo descargar el dataset. Verifica tu conexión a internet "
            f"y que el nombre '{dataset_name}' sea correcto."
        ) from exc


def split_dataset(
    ds: Dataset,
    train_ratio: float,
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    """Divide el dataset en train / eval / test."""
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, (
        "Las proporciones de split deben sumar 1.0"
    )

    # Primera división: train vs (eval + test)
    split1 = ds.train_test_split(
        test_size=eval_ratio + test_ratio,
        seed=seed,
    )
    # Segunda división: eval vs test dentro del remanente
    relative_test = test_ratio / (eval_ratio + test_ratio)
    split2 = split1["test"].train_test_split(
        test_size=relative_test,
        seed=seed,
    )

    splits = DatasetDict(
        {
            "train": split1["train"],
            "eval": split2["train"],
            "test": split2["test"],
        }
    )
    logger.info(
        "Dataset dividido — train: %d | eval: %d | test: %d",
        len(splits["train"]),
        len(splits["eval"]),
        len(splits["test"]),
    )
    return splits


# ---------------------------------------------------------------------------
# Funciones de transformación / preprocesamiento
# ---------------------------------------------------------------------------

def build_transform_fn(image_processor: AutoImageProcessor):
    """Devuelve una función de transformación para el dataset."""

    def transform(batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocesa las imágenes del batch usando el image_processor."""
        images = [
            img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            for img in batch["image"]
        ]
        inputs = image_processor(images=images, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    return transform


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict[str, float]:
    """Calcula accuracy y F1 macro durante la evaluación."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1_macro}


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Punto de entrada principal del entrenamiento."""
    args = parse_args()
    config = load_config()

    # Sobreescribir configuración con argumentos CLI si se proporcionan
    num_epochs: int = args.epochs or config["num_epochs"]
    batch_size: int = args.batch_size or config["batch_size"]
    learning_rate: float = args.lr or config["learning_rate"]
    output_dir: str = args.output_dir or config["output_dir"]

    # Resolver output_dir relativo a la raíz del proyecto
    output_dir = str((_PROJECT_ROOT / output_dir).resolve())

    logger.info("=" * 70)
    logger.info("  EcoClasifica — Entrenamiento de modelo de clasificación de residuos")
    logger.info("=" * 70)
    logger.info("  Modelo base      : %s", config["model_name"])
    logger.info("  Dataset           : %s", config["dataset_name"])
    logger.info("  Clases            : %s", config["label_names"])
    logger.info("  Épocas            : %d", num_epochs)
    logger.info("  Batch size        : %d", batch_size)
    logger.info("  Learning rate     : %s", learning_rate)
    logger.info("  Directorio salida : %s", output_dir)
    logger.info("  Dispositivo       : %s", "CUDA" if torch.cuda.is_available() else "CPU")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Descargar y dividir el dataset
    # ------------------------------------------------------------------
    raw_ds = download_dataset(config["dataset_name"])
    splits = split_dataset(
        raw_ds,
        train_ratio=config["train_split_ratio"],
        eval_ratio=config["eval_split_ratio"],
        test_ratio=config["test_split_ratio"],
        seed=config["seed"],
    )

    # ------------------------------------------------------------------
    # 2. Cargar el image processor y el modelo preentrenado
    # ------------------------------------------------------------------
    label_names: list[str] = config["label_names"]
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    logger.info("Cargando image processor desde '%s' …", config["model_name"])
    image_processor = AutoImageProcessor.from_pretrained(
        config["model_name"],
    )

    logger.info("Cargando modelo preentrenado '%s' …", config["model_name"])
    model = AutoModelForImageClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ------------------------------------------------------------------
    # 3. Aplicar transformaciones al dataset
    # ------------------------------------------------------------------
    logger.info("Aplicando preprocesamiento de imágenes …")
    transform_fn = build_transform_fn(image_processor)

    train_ds = splits["train"].with_transform(transform_fn)
    eval_ds = splits["eval"].with_transform(transform_fn)

    # ------------------------------------------------------------------
    # 4. Configurar los argumentos de entrenamiento
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=3,
        logging_steps=config["logging_steps"],
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=config["fp16"] and torch.cuda.is_available(),
        dataloader_num_workers=config["dataloader_num_workers"],
        seed=config["seed"],
        report_to="none",
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # 5. Crear el Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ------------------------------------------------------------------
    # 6. Entrenar
    # ------------------------------------------------------------------
    logger.info("Iniciando entrenamiento …")
    train_result = trainer.train()

    # ------------------------------------------------------------------
    # 7. Guardar modelo, processor y mapeos de etiquetas
    # ------------------------------------------------------------------
    logger.info("Guardando modelo entrenado en '%s' …", output_dir)
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)

    # Guardar mapeos de etiquetas como archivo adicional
    label_mapping = {
        "id2label": id2label,
        "label2id": label2id,
        "label_names": label_names,
        "label_names_es": config["label_names_es"],
    }
    label_map_path = os.path.join(output_dir, "label_mapping.json")
    with open(label_map_path, "w", encoding="utf-8") as fh:
        json.dump(label_mapping, fh, ensure_ascii=False, indent=2)
    logger.info("Mapeo de etiquetas guardado en '%s'.", label_map_path)

    # ------------------------------------------------------------------
    # 8. Evaluación final sobre el conjunto de evaluación
    # ------------------------------------------------------------------
    logger.info("Ejecutando evaluación final …")
    eval_metrics = trainer.evaluate()

    # ------------------------------------------------------------------
    # 9. Resumen del entrenamiento
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("  RESUMEN DEL ENTRENAMIENTO")
    logger.info("=" * 70)
    logger.info("  Tiempo total de entrenamiento : %.2f s", train_result.metrics.get("train_runtime", 0))
    logger.info("  Muestras por segundo          : %.2f", train_result.metrics.get("train_samples_per_second", 0))
    logger.info("  Pérdida final (train)         : %.4f", train_result.metrics.get("train_loss", 0))
    logger.info("  Accuracy (eval)               : %.4f", eval_metrics.get("eval_accuracy", 0))
    logger.info("  F1 macro (eval)               : %.4f", eval_metrics.get("eval_f1_macro", 0))
    logger.info("  Modelo guardado en            : %s", output_dir)
    logger.info("=" * 70)
    logger.info("¡Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()
