"""Funciones utilitarias para clasificar residuos según su descripción."""

from __future__ import annotations

CATEGORY_CONFIDENCE = {
    "plastic": 0.92,
    "paper": 0.88,
    "glass": 0.9,
    "metal": 0.91,
    "other": 0.5,
}

ALIASES = {
    "glass": {"vidrio", "botella de vidrio", "cristal", "glass"},
    "metal": {"metal", "lata", "aluminio", "hierro", "metalico"},
    "paper": {"papel", "carton", "cartón", "caja", "paper"},
    "plastic": {"plastico", "plástico", "botella", "envase", "plastic"},
}

LOW_RESOLUTION_THRESHOLD = 128
CONFIDENCE_ADJUSTMENT = 0.8
CATEGORY_PRIORITY = ("glass", "metal", "paper", "plastic")


def normalize_material_name(description: str) -> str:
    """Normaliza una descripción a una categoría reconocida."""
    if description is None:
        return "other"

    cleaned = description.lower().strip()
    for category in CATEGORY_PRIORITY:
        for alias in sorted(ALIASES[category], key=len, reverse=True):
            if cleaned == alias or alias in cleaned:
                return category
    return "other"


def calculate_confidence(category: str) -> float:
    """Devuelve la confianza base asociada a una categoría."""
    return CATEGORY_CONFIDENCE.get(category, CATEGORY_CONFIDENCE["other"])


def classify_residue(description: str, image_size: tuple[int, int] | None = None) -> dict:
    """Clasifica un residuo y ajusta la confianza si la imagen es pequeña."""
    category = normalize_material_name(description)
    confidence = calculate_confidence(category)

    if image_size is not None:
        width, height = image_size
        max_dimension = max(width, height)
        if max_dimension < LOW_RESOLUTION_THRESHOLD:
            confidence *= CONFIDENCE_ADJUSTMENT

    return {
        "category": category,
        "confidence": round(confidence, 2),
    }
