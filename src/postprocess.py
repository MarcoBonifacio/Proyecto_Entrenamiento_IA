"""Módulo de post-procesamiento de resultados para EcoClasifica.

Transforma las predicciones del modelo en información amigable
para el usuario, incluyendo recomendaciones de disposición ecológica.
"""

from __future__ import annotations

DEFAULT_CONFIDENCE = 0.0
BAR_LENGTH = 20
LABEL_PADDING = 18
PERCENTAGE_MULTIPLIER = 100.0

CATEGORY_EMOJIS = {
    "Cartón": "",
    "Vidrio": "",
    "Metal": "",
    "Papel": "",
    "Plástico": "",
    "Basura general": "",
}

DISPOSAL_RECOMMENDATIONS = {
    "Cartón": (
        "RECICLABLE — Aplana las cajas y retira cintas adhesivas. "
        "Deposita en el contenedor azul de papel y cartón. "
        "Evita mojar el cartón; si está sucio con comida, va a orgánicos."
    ),
    "Vidrio": (
        "RECICLABLE — Enjuaga el envase y retira tapas metálicas. "
        "Deposita en el contenedor verde de vidrio. "
        "¡El vidrio se recicla infinitamente sin perder calidad!"
    ),
    "Metal": (
        "RECICLABLE — Enjuaga latas y enlatados antes de reciclar. "
        "Deposita en el contenedor amarillo. "
        "El aluminio reciclado ahorra 95% de energía frente a producir nuevo."
    ),
    "Papel": (
        "RECICLABLE — Asegúrate de que esté limpio y seco. "
        "Deposita en el contenedor azul. "
        "No mezcles papel con servilletas o pañuelos usados."
    ),
    "Plástico": (
        "RECICLABLE (según tipo) — Revisa el número de reciclaje (1-7). "
        "Enjuaga el envase y deposita en el contenedor amarillo. "
        "Reduce el uso de plásticos de un solo uso siempre que sea posible."
    ),
    "Basura general": (
        "NO RECICLABLE — Este residuo no corresponde a las categorías "
        "reciclables principales. Deposita en el contenedor gris o de basura "
        "general. Considera si puede reutilizarse antes de desecharlo."
    ),
}


def format_prediction(prediction_dict: dict) -> str:
    """Formatea una predicción en una cadena legible para el usuario."""
    categoria = prediction_dict.get("categoria", "Desconocido")
    confianza = prediction_dict.get("confianza", DEFAULT_CONFIDENCE)

    header = f"Categoría detectada: {categoria}"
    conf_line = f"   Confianza: {confianza:.1%}"

    lines = [header, conf_line, "", "Distribución de probabilidades:"]

    todas = prediction_dict.get("todas_las_predicciones", [])
    if isinstance(todas, dict):
        todas_items = list(todas.items())
    elif isinstance(todas, list):
        todas_items = todas
    else:
        todas_items = []

    for item in todas_items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, prob = item[0], item[1]
        else:
            continue
        bar_len = int(prob * BAR_LENGTH)
        bar = "█" * bar_len + "░" * (BAR_LENGTH - bar_len)
        lines.append(f"   {label:{LABEL_PADDING}s} [{bar}] {prob:.1%}")

    return "\n".join(lines)


def get_disposal_recommendation(category: str) -> str:
    """Devuelve una recomendación de disposición ecológica."""
    return DISPOSAL_RECOMMENDATIONS.get(
        category,
        "No se encontró una recomendación específica para esta categoría. "
        "Consulta con tu municipio sobre la disposición adecuada de este residuo.",
    )


def create_results_summary(prediction_dict: dict) -> dict:
    """Crea un resumen completo de la predicción con recomendación."""
    categoria = prediction_dict.get("categoria", "Desconocido")

    return {
        "prediccion": format_prediction(prediction_dict),
        "recomendacion": get_disposal_recommendation(categoria),
        "emoji": CATEGORY_EMOJIS.get(categoria, ""),
        "categoria": categoria,
        "confianza": prediction_dict.get("confianza", DEFAULT_CONFIDENCE),
    }
