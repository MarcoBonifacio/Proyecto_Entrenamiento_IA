"""
Tests unitarios para el módulo de postprocesamiento.
Verifica el formateo de predicciones y las recomendaciones de disposición.
"""

import unittest

from src.postprocess import (
    format_prediction,
    get_disposal_recommendation,
    create_results_summary,
)


class TestFormatPrediction(unittest.TestCase):
    """Pruebas para la función format_prediction."""

    def test_format_prediction_returns_string(self) -> None:
        """Verifica que format_prediction retorna una cadena de texto."""
        prediction = {
            "categoria": "Cartón",
            "confianza": 0.95,
            "todas_las_predicciones": {
                "Cartón": 0.95,
                "Vidrio": 0.02,
                "Metal": 0.01,
                "Papel": 0.01,
                "Plástico": 0.005,
                "Basura general": 0.005,
            },
        }
        result = format_prediction(prediction)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestGetDisposalRecommendation(unittest.TestCase):
    """Pruebas para la función get_disposal_recommendation."""

    def test_get_disposal_recommendation_known_categories(self) -> None:
        """Verifica que cada categoría conocida retorna una recomendación no vacía."""
        categories = [
            "Cartón",
            "Vidrio",
            "Metal",
            "Papel",
            "Plástico",
            "Basura general",
        ]
        for category in categories:
            with self.subTest(category=category):
                result = get_disposal_recommendation(category)
                self.assertIsInstance(result, str)
                self.assertTrue(
                    len(result) > 0,
                    f"La recomendación para '{category}' no debe estar vacía",
                )

    def test_get_disposal_recommendation_unknown_category(self) -> None:
        """Verifica que una categoría desconocida retorna un mensaje por defecto."""
        result = get_disposal_recommendation("categoría_inexistente")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestCreateResultsSummary(unittest.TestCase):
    """Pruebas para la función create_results_summary."""

    def test_create_results_summary_has_required_keys(self) -> None:
        """Verifica que el resumen contiene las claves requeridas."""
        prediction = {
            "categoria": "Vidrio",
            "confianza": 0.88,
            "todas_las_predicciones": {
                "Cartón": 0.05,
                "Vidrio": 0.88,
                "Metal": 0.03,
                "Papel": 0.02,
                "Plástico": 0.01,
                "Basura general": 0.01,
            },
        }
        result = create_results_summary(prediction)
        self.assertIsInstance(result, dict)
        self.assertIn("prediccion", result)
        self.assertIn("recomendacion", result)
        self.assertIn("emoji", result)


if __name__ == "__main__":
    unittest.main()
