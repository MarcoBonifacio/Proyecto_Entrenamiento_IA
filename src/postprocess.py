"""Módulo de post-procesamiento de resultados para EcoClasifica.

Transforma las predicciones del modelo en información amigable
para el usuario, incluyendo recomendaciones de disposición ecológica.
"""

from __future__ import annotations

from typing import Optional


# Emojis asociados a cada categoría
CATEGORY_EMOJIS: dict[str, str] = {
    "Cartón": "📦",
    "Vidrio": "🫙",
    "Metal": "🥫",
    "Papel": "📄",
    "Plástico": "♻️",
    "Basura general": "🗑️",
}

# Recomendaciones de disposición ecológica por categoría
DISPOSAL_RECOMMENDATIONS: dict[str, str] = {
    "Cartón": (
        "♻️ RECICLABLE — Aplana las cajas y retira cintas adhesivas. "
        "Deposita en el contenedor azul de papel y cartón. "
        "Evita mojar el cartón; si está sucio con comida, va a orgánicos."
    ),
    "Vidrio": (
        "♻️ RECICLABLE — Enjuaga el envase y retira tapas metálicas. "
        "Deposita en el contenedor verde de vidrio. "
        "¡El vidrio se recicla infinitamente sin perder calidad!"
    ),
    "Metal": (
        "♻️ RECICLABLE — Enjuaga latas y enlatados antes de reciclar. "
        "Deposita en el contenedor amarillo. "
        "El aluminio reciclado ahorra 95% de energía frente a producir nuevo."
    ),
    "Papel": (
        "♻️ RECICLABLE — Asegúrate de que esté limpio y seco. "
        "Deposita en el contenedor azul. "
        "No mezcles papel con servilletas o pañuelos usados."
    ),
    "Plástico": (
        "♻️ RECICLABLE (según tipo) — Revisa el número de reciclaje (1-7). "
        "Enjuaga el envase y deposita en el contenedor amarillo. "
        "Reduce el uso de plásticos de un solo uso siempre que sea posible."
    ),
    "Basura general": (
        "🗑️ NO RECICLABLE — Este residuo no corresponde a las categorías "
        "reciclables principales. Deposita en el contenedor gris o de basura "
        "general. Considera si puede reutilizarse antes de desecharlo."
    ),
}


def format_prediction(prediction_dict: dict) -> str:
    """Formatea una predicción en una cadena legible para el usuario.

    Parameters
    ----------
    prediction_dict : dict
        Resultado de ``EcoClasificaModel.predict()`` con las claves
        ``"categoria"``, ``"confianza"`` y ``"todas_las_predicciones"``.

    Returns
    -------
    str
        Texto formateado con la categoría, confianza y desglose.
    """
    categoria = prediction_dict.get("categoria", "Desconocido")
    confianza = prediction_dict.get("confianza", 0.0)
    emoji = CATEGORY_EMOJIS.get(categoria, "❓")

    header = f"{emoji} Categoría detectada: {categoria}"
    conf_line = f"   Confianza: {confianza:.1%}"

    lines = [header, conf_line, "", "📊 Distribución de probabilidades:"]

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
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        cat_emoji = CATEGORY_EMOJIS.get(label, "")
        lines.append(f"   {cat_emoji} {label:18s} [{bar}] {prob:.1%}")

    return "\n".join(lines)


def get_disposal_recommendation(category: str) -> str:
    """Devuelve una recomendación de disposición ecológica.

    Parameters
    ----------
    category : str
        Nombre de la categoría en español (e.g. ``"Plástico"``).

    Returns
    -------
    str
        Recomendación de disposición ecológica para la categoría.
    """
    return DISPOSAL_RECOMMENDATIONS.get(
        category,
        "ℹ️ No se encontró una recomendación específica para esta categoría. "
        "Consulta con tu municipio sobre la disposición adecuada de este residuo.",
    )


def create_results_summary(prediction_dict: dict) -> dict:
    """Crea un resumen completo de la predicción con recomendación.

    Parameters
    ----------
    prediction_dict : dict
        Resultado de ``EcoClasificaModel.predict()``.

    Returns
    -------
    dict
        Diccionario con las claves:

        * ``"prediccion"`` — texto formateado de la predicción.
        * ``"recomendacion"`` — consejo de disposición ecológica.
        * ``"emoji"`` — emoji representativo de la categoría.
        * ``"categoria"`` — nombre de la categoría detectada.
        * ``"confianza"`` — valor de confianza (0–1).
    """
    categoria = prediction_dict.get("categoria", "Desconocido")

    return {
        "prediccion": format_prediction(prediction_dict),
        "recomendacion": get_disposal_recommendation(categoria),
        "emoji": CATEGORY_EMOJIS.get(categoria, "❓"),
        "categoria": categoria,
        "confianza": prediction_dict.get("confianza", 0.0),
    }
