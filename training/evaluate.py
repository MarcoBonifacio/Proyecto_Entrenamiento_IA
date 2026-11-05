"""
training/evaluate.py — Script de evaluación para EcoClasifica.

Evalúa el modelo fine-tuned sobre el split de test del dataset
TrashBusters/combinedd y genera métricas detalladas, reporte de
clasificación y matriz de confusión.

Uso:
    python training/evaluate.py
    python training/evaluate.py --image ruta/a/imagen.jpg
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

import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica (compatible con servidores)

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecoclasifica.evaluate")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SCRIPT_DIR / "config.json"
_PROJECT_ROOT = _SCRIPT_DIR.parent


def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Carga la configuración desde el archivo JSON."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_args() -> argparse.Namespace:
    """Analiza los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evaluación de EcoClasifica — métricas y matriz de confusión",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directorio del modelo entrenado (sobreescribe config.json).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a una imagen individual para clasificar.",
    )
    parser.add_argument(
        "--output_cm",
        type=str,
        default=None,
        help="Ruta para guardar la imagen de la matriz de confusión.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inferencia sobre una sola imagen
# ---------------------------------------------------------------------------

def predict_single_image(
    image_path: str,
    model: AutoModelForImageClassification,
    image_processor: AutoImageProcessor,
    id2label: dict[int, str],
    label_names_es: list[str],
    device: torch.device,
) -> None:
    """Clasifica una única imagen y muestra el resultado."""
    logger.info("Clasificando imagen: %s", image_path)

    if not os.path.isfile(image_path):
        logger.error("El archivo '%s' no existe.", image_path)
        raise SystemExit(1)

    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    predicted_label = id2label[predicted_id]
    predicted_label_es = label_names_es[predicted_id]
    confidence = float(probabilities[predicted_id].item())

    print()
    print("=" * 60)
    print("  RESULTADO DE CLASIFICACIÓN")
    print("=" * 60)
    print(f"  Imagen              : {image_path}")
    print(f"  Categoría (EN)      : {predicted_label}")
    print(f"  Categoría (ES)      : {predicted_label_es}")
    print(f"  Confianza           : {confidence:.2%}")
    print()
    print("  Probabilidades por clase:")
    print("  " + "-" * 40)
    for idx in range(len(id2label)):
        label = id2label[idx]
        label_es = label_names_es[idx]
        prob = float(probabilities[idx].item())
        bar = "█" * int(prob * 30)
        marker = " ◄" if idx == predicted_id else ""
        print(f"  {label_es:<18s} ({label:<10s}): {prob:6.2%} {bar}{marker}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Evaluación completa sobre el split de test
# ---------------------------------------------------------------------------

def evaluate_test_split(
    model: AutoModelForImageClassification,
    image_processor: AutoImageProcessor,
    config: dict[str, Any],
    model_dir: str,
    cm_output_path: str,
    device: torch.device,
) -> None:
    """Ejecuta la evaluación completa sobre el conjunto de test."""

    label_names: list[str] = config["label_names"]
    label_names_es: list[str] = config["label_names_es"]

    # ------------------------------------------------------------------
    # 1. Cargar y dividir el dataset (misma semilla que el entrenamiento)
    # ------------------------------------------------------------------
    logger.info("Cargando dataset '%s' …", config["dataset_name"])
    try:
        ds = load_dataset(config["dataset_name"], split="train")
    except Exception as exc:
        logger.error("Error al cargar el dataset: %s", exc)
        raise SystemExit(1) from exc

    # Reproducir exactamente las mismas divisiones que train.py
    split1 = ds.train_test_split(
        test_size=config["eval_split_ratio"] + config["test_split_ratio"],
        seed=config["seed"],
    )
    relative_test = config["test_split_ratio"] / (
        config["eval_split_ratio"] + config["test_split_ratio"]
    )
    split2 = split1["test"].train_test_split(
        test_size=relative_test,
        seed=config["seed"],
    )
    test_ds = split2["test"]
    logger.info("Conjunto de test: %d muestras.", len(test_ds))

    # ------------------------------------------------------------------
    # 2. Realizar predicciones
    # ------------------------------------------------------------------
    logger.info("Realizando predicciones sobre el conjunto de test …")
    all_preds: list[int] = []
    all_labels: list[int] = []
    batch_size = config["batch_size"]
    total = len(test_ds)

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = test_ds[start_idx:end_idx]

            images = [
                img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
                for img in batch["image"]
            ]
            inputs = image_processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(batch["label"])

            if (start_idx // batch_size) % 10 == 0:
                logger.info(
                    "  Progreso: %d / %d (%.1f%%)",
                    end_idx,
                    total,
                    end_idx / total * 100,
                )

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    # ------------------------------------------------------------------
    # 3. Calcular métricas
    # ------------------------------------------------------------------
    acc = accuracy_score(all_labels_np, all_preds_np)
    precision_macro = precision_score(
        all_labels_np, all_preds_np, average="macro", zero_division=0,
    )
    recall_macro = recall_score(
        all_labels_np, all_preds_np, average="macro", zero_division=0,
    )
    f1_macro = f1_score(
        all_labels_np, all_preds_np, average="macro", zero_division=0,
    )

    # ------------------------------------------------------------------
    # 4. Reporte de clasificación
    # ------------------------------------------------------------------
    report = classification_report(
        all_labels_np,
        all_preds_np,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("  RESULTADOS DE EVALUACIÓN — EcoClasifica")
    print("=" * 70)
    print()
    print(f"  Total de muestras de test : {total}")
    print(f"  Accuracy                  : {acc:.4f}")
    print(f"  Precision (macro)         : {precision_macro:.4f}")
    print(f"  Recall (macro)            : {recall_macro:.4f}")
    print(f"  F1-score (macro)          : {f1_macro:.4f}")
    print()
    print("-" * 70)
    print("  REPORTE DE CLASIFICACIÓN POR CLASE")
    print("-" * 70)
    print()
    print(report)
    print()

    # Tabla resumen en español
    print("-" * 70)
    print("  RESUMEN POR CATEGORÍA (ES)")
    print("-" * 70)
    per_class_f1 = f1_score(
        all_labels_np, all_preds_np, average=None, zero_division=0,
    )
    per_class_precision = precision_score(
        all_labels_np, all_preds_np, average=None, zero_division=0,
    )
    per_class_recall = recall_score(
        all_labels_np, all_preds_np, average=None, zero_division=0,
    )

    header = f"  {'Categoría':<20s} {'Precisión':>10s} {'Recall':>10s} {'F1':>10s}"
    print(header)
    print("  " + "-" * 52)
    for idx, (name_en, name_es) in enumerate(zip(label_names, label_names_es)):
        print(
            f"  {name_es:<20s} {per_class_precision[idx]:>10.4f} "
            f"{per_class_recall[idx]:>10.4f} {per_class_f1[idx]:>10.4f}"
        )
    print("  " + "-" * 52)
    print(
        f"  {'Macro promedio':<20s} {precision_macro:>10.4f} "
        f"{recall_macro:>10.4f} {f1_macro:>10.4f}"
    )
    print("=" * 70)

    # ------------------------------------------------------------------
    # 5. Matriz de confusión
    # ------------------------------------------------------------------
    cm = confusion_matrix(all_labels_np, all_preds_np)
    logger.info("Generando matriz de confusión …")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=label_names_es,
        yticklabels=label_names_es,
        title="Matriz de Confusión — EcoClasifica",
        ylabel="Etiqueta real",
        xlabel="Etiqueta predicha",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Anotar cada celda con el conteo
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=11,
            )

    fig.tight_layout()

    # Guardar la imagen
    cm_dir = Path(cm_output_path).parent
    cm_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Matriz de confusión guardada en '%s'.", cm_output_path)

    # Guardar las métricas como JSON
    metrics_path = os.path.join(model_dir, "eval_metrics.json")
    metrics_dict = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "per_class": {
            label_names[i]: {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            }
            for i in range(len(label_names))
        },
        "confusion_matrix": cm.tolist(),
        "num_test_samples": total,
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_dict, fh, ensure_ascii=False, indent=2)
    logger.info("Métricas de evaluación guardadas en '%s'.", metrics_path)

    print()
    logger.info("¡Evaluación completada exitosamente!")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    """Punto de entrada principal de la evaluación."""
    args = parse_args()
    config = load_config()

    # Resolver directorio del modelo
    model_dir = args.model_dir or config["output_dir"]
    model_dir = str((_PROJECT_ROOT / model_dir).resolve())

    # Ruta de la matriz de confusión
    cm_output_path = args.output_cm or str(
        (_PROJECT_ROOT / "static" / "confusion_matrix.png").resolve()
    )

    # ------------------------------------------------------------------
    # Verificar que el modelo exista
    # ------------------------------------------------------------------
    if not os.path.isdir(model_dir):
        logger.error(
            "No se encontró el directorio del modelo: '%s'. "
            "Ejecuta primero 'python training/train.py'.",
            model_dir,
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Cargar modelo y procesador
    # ------------------------------------------------------------------
    logger.info("Cargando modelo desde '%s' …", model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)

    # Cargar mapeo de etiquetas
    label_map_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.isfile(label_map_path):
        with open(label_map_path, encoding="utf-8") as fh:
            label_mapping = json.load(fh)
        id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
        label_names_es = label_mapping["label_names_es"]
    else:
        id2label = {i: name for i, name in enumerate(config["label_names"])}
        label_names_es = config["label_names_es"]

    logger.info("Modelo cargado correctamente. Dispositivo: %s", device)

    # ------------------------------------------------------------------
    # Modo: imagen individual o evaluación completa
    # ------------------------------------------------------------------
    if args.image:
        predict_single_image(
            image_path=args.image,
            model=model,
            image_processor=image_processor,
            id2label=id2label,
            label_names_es=label_names_es,
            device=device,
        )
    else:
        evaluate_test_split(
            model=model,
            image_processor=image_processor,
            config=config,
            model_dir=model_dir,
            cm_output_path=cm_output_path,
            device=device,
        )


if __name__ == "__main__":
    main()

# Constantes UPPER_CASE (Refactor sin magic numbers)
DEFAULT_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60
MAX_BATCH_SIZE = 32
