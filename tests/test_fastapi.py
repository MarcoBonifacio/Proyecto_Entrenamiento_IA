"""Tests unitarios para la API REST con FastAPI (src/fastapi_app.py).

Verifica los endpoints /, /health, /classes, /classify, /predict y /classify/base64.
"""

from __future__ import annotations

import base64
import io
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from src.fastapi_app import app


class TestFastAPIEndpoints(unittest.TestCase):
    """Suite de pruebas para los endpoints de la API FastAPI."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def _create_sample_image_bytes(self, format: str = "PNG") -> bytes:
        img = Image.new("RGB", (224, 224), color=(120, 180, 240))
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()

    def test_root_endpoint(self) -> None:
        """Verifica que el endpoint raíz devuelve 200 y enlaces a la documentación."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("documentacion_swagger", data)
        self.assertEqual(data["documentacion_swagger"], "/docs")
        self.assertIn("health_check", data)

    def test_health_endpoint(self) -> None:
        """Verifica que el endpoint /health devuelve status ok y metadatos del servicio."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["servicio"], "EcoClasifica API")
        self.assertIn("uptime_seconds", data)

    def test_classes_endpoint(self) -> None:
        """Verifica que /classes devuelve exactamente 6 categorías con sus emojis."""
        response = self.client.get("/classes")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_clases"], 6)
        nombres = [c["categoria"] for c in data["clases"]]
        self.assertIn("Cartón", nombres)
        self.assertIn("Vidrio", nombres)
        self.assertIn("Metal", nombres)
        self.assertIn("Papel", nombres)
        self.assertIn("Plástico", nombres)
        self.assertIn("Basura general", nombres)

    @patch("src.fastapi_app.classify_image_api")
    def test_classify_file_success(self, mock_api: MagicMock) -> None:
        """Verifica la clasificación exitosa de un archivo de imagen vía multipart/form-data."""
        mock_api.return_value = {
            "error": False,
            "categoria": "Plástico",
            "confianza": 0.94,
            "emoji": "♻️",
            "recomendacion": "Enjuaga y aplana el plástico.",
            "todas_las_predicciones": {
                "Plástico": 0.94,
                "Cartón": 0.02,
                "Vidrio": 0.01,
                "Metal": 0.01,
                "Papel": 0.01,
                "Basura general": 0.01,
            },
            "modelo_afinado": False,
        }

        image_bytes = self._create_sample_image_bytes("PNG")
        response = self.client.post(
            "/classify",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["error"])
        self.assertEqual(data["categoria"], "Plástico")
        self.assertEqual(data["emoji"], "♻️")
        self.assertAlmostEqual(data["confianza"], 0.94)

    @patch("src.fastapi_app.classify_image_api")
    def test_predict_alias_success(self, mock_api: MagicMock) -> None:
        """Verifica que /predict funciona como alias idéntico de /classify."""
        mock_api.return_value = {
            "error": False,
            "categoria": "Metal",
            "confianza": 0.91,
            "emoji": "🥫",
            "recomendacion": "Enjuagar latas.",
            "todas_las_predicciones": {"Metal": 0.91},
            "modelo_afinado": False,
        }

        image_bytes = self._create_sample_image_bytes("JPEG")
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["categoria"], "Metal")

    def test_classify_rejects_non_image_content_type(self) -> None:
        """Verifica que se rechacen archivos que no sean imágenes con 400 Bad Request."""
        response = self.client.post(
            "/classify",
            files={"file": ("notas.txt", b"este es un archivo de texto", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Tipo de archivo no permitido", response.json()["detail"])

    @patch("src.fastapi_app.classify_image_api")
    def test_classify_base64_success(self, mock_api: MagicMock) -> None:
        """Verifica la clasificación a partir de una cadena codificada en base64."""
        mock_api.return_value = {
            "error": False,
            "categoria": "Vidrio",
            "confianza": 0.89,
            "emoji": "🫙",
            "recomendacion": "Depositar en contenedor verde.",
            "todas_las_predicciones": {"Vidrio": 0.89},
            "modelo_afinado": False,
        }

        image_bytes = self._create_sample_image_bytes("PNG")
        b64_str = base64.b64encode(image_bytes).decode("utf-8")
        response = self.client.post(
            "/classify/base64",
            json={"image_base64": f"data:image/png;base64,{b64_str}"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["categoria"], "Vidrio")
        self.assertEqual(data["emoji"], "🫙")

    def test_classify_base64_invalid_string(self) -> None:
        """Verifica que una cadena base64 corrompida devuelve 400 Bad Request."""
        response = self.client.post(
            "/classify/base64",
            json={"image_base64": "!!!no-es-base64-valido!!!"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
