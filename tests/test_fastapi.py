"""Tests unitarios para la API REST con FastAPI (src/fastapi_app.py).

Verifica los endpoints /, /health, /classes, /classify, /predict y /classify/base64,
con especial énfasis en las validaciones de tipos de archivo, extensiones,
magic bytes, límites de tamaño y resolución.
"""

from __future__ import annotations

import base64
import io
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from src.fastapi_app import (
    app,
    TAMANO_MAXIMO_BYTES,
)


class TestFastAPIEndpoints(unittest.TestCase):
    """Suite de pruebas para los endpoints y validaciones de la API FastAPI."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def _create_sample_image_bytes(
        self,
        format: str = "PNG",
        size: tuple[int, int] = (224, 224),
    ) -> bytes:
        """Crea bytes de una imagen en memoria para pruebas."""
        img = Image.new("RGB", size, color=(120, 180, 240))
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()

    # ==========================================================================
    # Pruebas de Endpoints Generales y de Salud
    # ==========================================================================
    def test_root_endpoint(self) -> None:
        """Verifica que el endpoint raíz devuelve 200 y enlaces a la documentación."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("documentacion_swagger", data)
        self.assertEqual(data["documentacion_swagger"], "/docs")
        self.assertIn("health_check", data)
        self.assertIn("formatos_permitidos", data)
        self.assertIn("tamano_maximo_mb", data)

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
        for categoria in ["Cartón", "Vidrio", "Metal", "Papel", "Plástico", "Basura general"]:
            self.assertIn(categoria, nombres)

    # ==========================================================================
    # Pruebas de Clasificación Exitosa (Formatos Permitidos)
    # ==========================================================================
    @patch("src.fastapi_app.classify_image_api")
    def test_classify_png_file_success(self, mock_api: MagicMock) -> None:
        """Verifica la clasificación exitosa de una imagen PNG vía multipart/form-data."""
        mock_api.return_value = {
            "error": False,
            "categoria": "Plástico",
            "confianza": 0.94,
            "emoji": "♻️",
            "recomendacion": "Enjuaga y aplana el plástico.",
            "todas_las_predicciones": {"Plástico": 0.94, "Cartón": 0.06},
            "modelo_afinado": False,
        }

        image_bytes = self._create_sample_image_bytes("PNG")
        response = self.client.post(
            "/classify",
            files={"file": ("residuo.png", image_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["error"])
        self.assertEqual(data["categoria"], "Plástico")
        self.assertEqual(data["emoji"], "♻️")
        self.assertAlmostEqual(data["confianza"], 0.94)

    @patch("src.fastapi_app.classify_image_api")
    def test_classify_jpeg_file_success(self, mock_api: MagicMock) -> None:
        """Verifica la clasificación exitosa de una imagen JPEG vía /classify."""
        mock_api.return_value = {
            "error": False,
            "categoria": "Cartón",
            "confianza": 0.95,
            "emoji": "📦",
            "recomendacion": "Dobla y aplana la caja de cartón.",
            "todas_las_predicciones": {"Cartón": 0.95},
            "modelo_afinado": False,
        }

        image_bytes = self._create_sample_image_bytes("JPEG")
        response = self.client.post(
            "/classify",
            files={"file": ("caja.jpg", image_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["categoria"], "Cartón")

    @patch("src.fastapi_app.classify_image_api")
    def test_classify_webp_file_success(self, mock_api: MagicMock) -> None:
        """Verifica la clasificación exitosa de una imagen WEBP moderna."""
        mock_api.return_value = {
            "error": False,
            "categoria": "Vidrio",
            "confianza": 0.90,
            "emoji": "🫙",
            "recomendacion": "Depositar en el contenedor verde.",
            "todas_las_predicciones": {"Vidrio": 0.90},
            "modelo_afinado": False,
        }

        image_bytes = self._create_sample_image_bytes("WEBP")
        response = self.client.post(
            "/classify",
            files={"file": ("botella.webp", image_bytes, "image/webp")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["categoria"], "Vidrio")

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
            files={"file": ("lata.jpg", image_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["categoria"], "Metal")

    # ==========================================================================
    # Pruebas de Validaciones de Tipos de Archivo y Seguridad
    # ==========================================================================
    def test_classify_rejects_non_image_content_type(self) -> None:
        """Verifica que se rechacen archivos que no sean imágenes con 400 Bad Request."""
        response = self.client.post(
            "/classify",
            files={"file": ("notas.txt", b"este es un archivo de texto largo para pruebas", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Tipo de archivo no permitido", response.json()["detail"])

    def test_classify_rejects_invalid_extension(self) -> None:
        """Verifica que se rechacen archivos cuya extensión no esté permitida (ej. .pdf)."""
        image_bytes = self._create_sample_image_bytes("PNG")
        response = self.client.post(
            "/classify",
            files={"file": ("documento.pdf", image_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Extensión de archivo no permitida", response.json()["detail"])

    def test_classify_rejects_spoofed_file_content(self) -> None:
        """Verifica que un archivo de texto con extensión y tipo MIME falsificados sea detectado."""
        archivo_falso = b"Esto no es una imagen real, es codigo malicioso simulado" * 10
        response = self.client.post(
            "/classify",
            files={"file": ("malicioso.png", archivo_falso, "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("No se pudo decodificar el archivo como imagen válida", response.json()["detail"])

    def test_classify_rejects_empty_file(self) -> None:
        """Verifica que un archivo de 0 bytes o menor al tamaño mínimo sea rechazado."""
        response = self.client.post(
            "/classify",
            files={"file": ("vacio.png", b"", "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("vacío", response.json()["detail"])

    def test_classify_rejects_oversized_file(self) -> None:
        """Verifica que un archivo que supere el límite de 10 MB devuelva 413 Payload Too Large."""
        contenido_pesado = b"X" * (TAMANO_MAXIMO_BYTES + 1024)
        response = self.client.post(
            "/classify",
            files={"file": ("pesado.png", contenido_pesado, "image/png")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("supera el tamaño máximo permitido", response.json()["detail"])

    def test_classify_rejects_sub_minimum_dimensions(self) -> None:
        """Verifica que imágenes con dimensiones menores a 10x10 píxeles sean rechazadas."""
        imagen_diminuta = self._create_sample_image_bytes("PNG", size=(5, 5))
        response = self.client.post(
            "/classify",
            files={"file": ("diminuto.png", imagen_diminuta, "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("inferiores al mínimo requerido", response.json()["detail"])

    # ==========================================================================
    # Pruebas del Endpoint Base64 (/classify/base64)
    # ==========================================================================
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

    def test_classify_base64_empty_string(self) -> None:
        """Verifica que una cadena base64 vacía retorne 400 Bad Request."""
        response = self.client.post(
            "/classify/base64",
            json={"image_base64": "   "},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("cadena base64 proporcionada está vacía", response.json()["detail"])

    def test_classify_base64_invalid_string(self) -> None:
        """Verifica que una cadena base64 corrompida devuelva 400 Bad Request."""
        response = self.client.post(
            "/classify/base64",
            json={"image_base64": "!!!no-es-base64-valido!!!"},
        )
        self.assertEqual(response.status_code, 400)

    def test_classify_base64_non_image_payload(self) -> None:
        """Verifica que una cadena base64 que no contenga una imagen real devuelva 400 Bad Request."""
        texto_bytes = b"Este es un texto arbitrario codificado en base64" * 5
        b64_texto = base64.b64encode(texto_bytes).decode("utf-8")
        response = self.client.post(
            "/classify/base64",
            json={"image_base64": b64_texto},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("No se pudo decodificar el archivo como imagen válida", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
