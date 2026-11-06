"""Funciones utilitarias para clasificar residuos según su descripción."""

from __future__ import annotations

# Constantes de configuración general y categorías
DEFAULT_CATEGORY: str = "other"
CATEGORY_GLASS: str = "glass"
CATEGORY_METAL: str = "metal"
CATEGORY_PAPER: str = "paper"
CATEGORY_PLASTIC: str = "plastic"
CATEGORY_OTHER: str = DEFAULT_CATEGORY

# Constantes numéricas de confianza por categoría (sin magic numbers)
CONFIDENCE_PLASTIC: float = 0.92
CONFIDENCE_PAPER: float = 0.88
CONFIDENCE_GLASS: float = 0.90
CONFIDENCE_METAL: float = 0.91
CONFIDENCE_OTHER: float = 0.50

# Constantes de resolución y ajuste de confianza
LOW_RESOLUTION_THRESHOLD: int = 128
CONFIDENCE_ADJUSTMENT: float = 0.8
ROUND_DECIMALS: int = 2

# Parámetros estándar de procesamiento de imágenes
DEFAULT_IMAGE_SIZE: int = 224
CONFIDENCE_THRESHOLD: float = 0.60
MAX_BATCH_SIZE: int = 32

# Mapeo de niveles de confianza según categoría
CATEGORY_CONFIDENCE: dict[str, float] = {
    CATEGORY_PLASTIC: CONFIDENCE_PLASTIC,
    CATEGORY_PAPER: CONFIDENCE_PAPER,
    CATEGORY_GLASS: CONFIDENCE_GLASS,
    CATEGORY_METAL: CONFIDENCE_METAL,
    CATEGORY_OTHER: CONFIDENCE_OTHER,
}

# Diccionario de alias por categoría
ALIASES: dict[str, set[str]] = {
    CATEGORY_GLASS: {"vidrio", "botella de vidrio", "cristal", "glass"},
    CATEGORY_METAL: {"metal", "lata", "aluminio", "hierro", "metalico"},
    CATEGORY_PAPER: {"papel", "carton", "cartón", "caja", "paper"},
    CATEGORY_PLASTIC: {"plastico", "plástico", "botella", "envase", "plastic"},
}

CATEGORY_PRIORITY: tuple[str, ...] = (
    CATEGORY_GLASS,
    CATEGORY_METAL,
    CATEGORY_PAPER,
    CATEGORY_PLASTIC,
)


def normalize_material_name(description: str | None) -> str:
    """Normaliza una descripción a una categoría reconocida."""
    if description is None:
        return DEFAULT_CATEGORY

    cleaned = description.lower().strip()
    for category in CATEGORY_PRIORITY:
        for alias in sorted(ALIASES[category], key=len, reverse=True):
            if cleaned == alias or alias in cleaned:
                return category
    return DEFAULT_CATEGORY


def calculate_confidence(category: str) -> float:
    """Devuelve la confianza base asociada a una categoría."""
    return CATEGORY_CONFIDENCE.get(category, CATEGORY_CONFIDENCE[DEFAULT_CATEGORY])


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
        "confidence": round(confidence, ROUND_DECIMALS),
    }
