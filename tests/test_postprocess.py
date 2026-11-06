"""
Tests unitarios para el módulo de postprocesamiento.
Verifica el formateo de predicciones y las recomendaciones de disposición sin magic numbers.
"""

from __future__ import annotations

import unittest

from src.postprocess import (
    BAR_LENGTH,
    DEFAULT_CONFIDENCE,
    LABEL_PADDING,
    PERCENTAGE_MULTIPLIER,
    create_results_summary,
    format_prediction,
    get_disposal_recommendation,
)

# Constantes de prueba TDD (UPPER_CASE / sin magic numbers)
TEST_HIGH_CONFIDENCE: float = 0.95
TEST_MEDIUM_CONFIDENCE: float = 0.88
TEST_CATEGORY_CARDBOARD: str = "Cartón"
TEST_CATEGORY_GLASS: str = "Vidrio"
TEST_UNKNOWN_CATEGORY: str = "categoría_inexistente"


class TestFormatPrediction(unittest.TestCase):
    """Pruebas para la función format_prediction."""

    def test_format_prediction_returns_string(self) -> None:
        """Verifica que format_prediction retorna una cadena de texto."""
        prediction = {
            "categoria": TEST_CATEGORY_CARDBOARD,
            "confianza": TEST_HIGH_CONFIDENCE,
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

    def test_format_prediction_with_list_input(self) -> None:
        """Verifica que format_prediction maneje lista de tuplas e ignore items no válidos."""
        prediction = {
            "categoria": TEST_CATEGORY_GLASS,
            "confianza": TEST_MEDIUM_CONFIDENCE,
            "todas_las_predicciones": [
                ("Vidrio", 0.88),
                ("Metal", 0.08),
                ("Papel", 0.04),
                "item_invalido",
            ],
        }
        result = format_prediction(prediction)
        self.assertIsInstance(result, str)
        self.assertIn("Vidrio", result)

    def test_format_prediction_with_non_iterable_predictions(self) -> None:
        """Verifica que format_prediction funcione con predicciones None o vacías."""
        prediction = {
            "categoria": TEST_CATEGORY_GLASS,
            "confianza": TEST_MEDIUM_CONFIDENCE,
            "todas_las_predicciones": None,
        }
        result = format_prediction(prediction)
        self.assertIsInstance(result, str)
        self.assertIn("Vidrio", result)


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
        result = get_disposal_recommendation(TEST_UNKNOWN_CATEGORY)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestCreateResultsSummary(unittest.TestCase):
    """Pruebas para la función create_results_summary."""

    def test_create_results_summary_has_required_keys(self) -> None:
        """Verifica que el resumen contiene las claves requeridas."""
        prediction = {
            "categoria": TEST_CATEGORY_GLASS,
            "confianza": TEST_MEDIUM_CONFIDENCE,
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

    def test_constants_defined_without_magic_numbers(self) -> None:
        """Verifica que las constantes del módulo postprocess mantengan valores correctos."""
        self.assertEqual(DEFAULT_CONFIDENCE, 0.0)
        self.assertEqual(BAR_LENGTH, 20)
        self.assertEqual(LABEL_PADDING, 18)
        self.assertEqual(PERCENTAGE_MULTIPLIER, 100.0)


if __name__ == "__main__":
    unittest.main()
