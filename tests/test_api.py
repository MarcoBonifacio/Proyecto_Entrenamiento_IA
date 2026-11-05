"""
Tests unitarios para el módulo de API / rutas de la aplicación.
Verifica los endpoints de salud y clasificación de imágenes.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.api import health_check, classify_image_api


class TestHealthCheck(unittest.TestCase):
    """Pruebas para el endpoint de verificación de salud."""

    def test_health_check_returns_status(self) -> None:
        """Verifica que health_check retorna un dict con la clave 'status'."""
        result = health_check()
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)


class TestClassifyImageApi(unittest.TestCase):
    """Pruebas para el endpoint de clasificación de imágenes."""

    @patch("src.api.classify_waste")
    def test_classify_image_api_returns_dict(
        self, mock_classify: MagicMock
    ) -> None:
        """Verifica que classify_image_api retorna un diccionario."""
        mock_classify.return_value = {
            "categoria": "Metal",
            "confianza": 0.92,
            "todas_las_predicciones": {
                "Cartón": 0.02,
                "Vidrio": 0.03,
                "Metal": 0.92,
                "Papel": 0.01,
                "Plástico": 0.01,
                "Basura general": 0.01,
            },
        }

        from PIL import Image

        image = Image.new("RGB", (224, 224), color=(100, 100, 100))
        result = classify_image_api(image)

        self.assertIsInstance(result, dict)

    @patch("src.api.classify_waste")
    def test_classify_image_api_handles_invalid_path(
        self, mock_classify: MagicMock
    ) -> None:
        """Verifica que se manejan correctamente las entradas inválidas."""
        mock_classify.side_effect = Exception(
            "Error al procesar la imagen"
        )

        result = classify_image_api(None)

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
