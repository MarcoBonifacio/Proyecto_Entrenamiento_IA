"""Interfaz web de EcoClasifica con Gradio.

Proporciona una aplicación interactiva para clasificar imágenes
de residuos y obtener recomendaciones de reciclaje.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import EcoClasificaModel  # noqa: E402
from src.postprocess import (  # noqa: E402
    create_results_summary,
    CATEGORY_EMOJIS,
)

logger = logging.getLogger(__name__)

# Ruta al modelo afinado
FINETUNED_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "ecoclasifica-resnet50",
)

# Variable global para el modelo (se carga una sola vez)
_model: EcoClasificaModel | None = None


def _get_model() -> EcoClasificaModel:
    """Obtiene o inicializa el modelo singleton."""
    global _model
    if _model is None:
        if Path(FINETUNED_MODEL_PATH).exists():
            logger.info("Cargando modelo afinado desde '%s'", FINETUNED_MODEL_PATH)
            _model = EcoClasificaModel(model_path=FINETUNED_MODEL_PATH)
        else:
            logger.warning(
                "Modelo afinado no encontrado en '%s'. "
                "Usando modelo base ResNet-50 (las predicciones pueden no ser precisas). "
                "Ejecuta 'python training/train.py' para entrenar el modelo.",
                FINETUNED_MODEL_PATH,
            )
            _model = EcoClasificaModel(model_path=None)
    return _model


def predict_waste(image: np.ndarray | Image.Image | None) -> tuple[dict, str]:
    """Clasifica una imagen de residuo y devuelve resultados.

    Parameters
    ----------
    image : np.ndarray | Image.Image | None
        Imagen capturada o subida por el usuario desde Gradio.

    Returns
    -------
    tuple[dict, str]
        Tupla con (diccionario de probabilidades para gr.Label, texto de recomendación).
    """
    if image is None:
        return {}, "⚠️ Por favor sube o captura una imagen de un residuo."

    try:
        model = _get_model()
        prediction = model.predict(image)

        # Preparar el diccionario para el componente Label de Gradio
        label_dict: dict[str, float] = {}
        for label, prob in prediction.get("todas_las_predicciones", []):
            emoji = CATEGORY_EMOJIS.get(label, "")
            label_dict[f"{emoji} {label}"] = prob

        # Crear resumen con recomendación
        summary = create_results_summary(prediction)
        recommendation = (
            f"## {summary['emoji']} {summary['categoria']} "
            f"(Confianza: {summary['confianza']:.1%})\n\n"
            f"{summary['recomendacion']}"
        )

        if not model.is_finetuned:
            recommendation += (
                "\n\n---\n"
                "⚠️ **Nota:** Se está usando el modelo base sin afinar. "
                "Las predicciones pueden no ser precisas. "
                "Ejecuta el entrenamiento para mejorar los resultados."
            )

        return label_dict, recommendation

    except Exception as exc:
        logger.error("Error al clasificar la imagen: %s", exc)
        return {}, f"❌ Error al procesar la imagen: {exc}"


def create_interface() -> gr.Blocks:
    """Crea y configura la interfaz Gradio."""
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="EcoClasifica — Clasificación de Residuos con IA",
        css="""
        .footer { text-align: center; margin-top: 20px; padding: 15px;
                  color: #666; font-size: 0.85em; }
        .header-img { max-height: 80px; }
        """,
    ) as app:
        gr.Markdown(
            """
            # 🌿 EcoClasifica — Clasificación Inteligente de Residuos

            Sube o captura una fotografía de un residuo y nuestro modelo de IA
            basado en **ResNet-50** identificará su categoría y te dará
            recomendaciones para su disposición ecológica.

            **Categorías soportadas:** 📦 Cartón · 🫙 Vidrio · 🥫 Metal
            · 📄 Papel · ♻️ Plástico · 🗑️ Basura general
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 Sube o captura una imagen del residuo",
                    type="pil",
                    height=350,
                    sources=["upload", "webcam"],
                )
                classify_btn = gr.Button(
                    "🔍 Clasificar Residuo",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                label_output = gr.Label(
                    label="📊 Resultados de la clasificación",
                    num_top_classes=6,
                )
                recommendation_output = gr.Markdown(
                    label="♻️ Recomendación",
                    value="*Los resultados aparecerán aquí después de clasificar una imagen.*",
                )

        # Vincular el botón con la función de predicción
        classify_btn.click(
            fn=predict_waste,
            inputs=[image_input],
            outputs=[label_output, recommendation_output],
        )

        # También clasificar al soltar una imagen
        image_input.change(
            fn=predict_waste,
            inputs=[image_input],
            outputs=[label_output, recommendation_output],
        )

        gr.Markdown(
            """
            ---
            <div class="footer">
                <strong>EcoClasifica v1.0</strong> · Equipo DEV<br>
                Proyecto Final — Construcción de Software —
                <em>Universidad Continental</em><br>
                Modelo: ResNet-50 afinado con el dataset TrashBusters/combinedd
                (29,849 imágenes)
            </div>
            """
        )

    return app


def main() -> None:
    """Punto de entrada principal de la aplicación."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Iniciando EcoClasifica…")

    # Pre-cargar el modelo al iniciar
    _get_model()

    app = create_interface()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
