"""
Tests unitarios para el módulo ecoclasifica.
Verifica la normalización de materiales, cálculo de confianza y clasificación sin magic numbers.
"""

from __future__ import annotations

import unittest

from src.ecoclasifica import (
    CATEGORY_GLASS,
    CATEGORY_METAL,
    CATEGORY_OTHER,
    CATEGORY_PAPER,
    CATEGORY_PLASTIC,
    CONFIDENCE_ADJUSTMENT,
    CONFIDENCE_GLASS,
    CONFIDENCE_METAL,
    CONFIDENCE_OTHER,
    CONFIDENCE_PAPER,
    CONFIDENCE_PLASTIC,
    DEFAULT_CATEGORY,
    DEFAULT_IMAGE_SIZE,
    LOW_RESOLUTION_THRESHOLD,
    ROUND_DECIMALS,
    calculate_confidence,
    classify_residue,
    normalize_material_name,
)

# Constantes de prueba TDD (UPPER_CASE / sin magic numbers)
TEST_SMALL_IMAGE_SIZE: tuple[int, int] = (64, 64)
TEST_THRESHOLD_IMAGE_SIZE: tuple[int, int] = (LOW_RESOLUTION_THRESHOLD, LOW_RESOLUTION_THRESHOLD)
TEST_HIGH_RES_IMAGE_SIZE: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
TEST_GLASS_DESCRIPTION: str = "Botella de vidrio"
TEST_PAPER_DESCRIPTION: str = "Caja de cartón"
TEST_METAL_DESCRIPTION: str = "Lata de aluminio"
TEST_PLASTIC_DESCRIPTION: str = "Envase de plastico"
TEST_UNKNOWN_DESCRIPTION: str = "objeto raro"


class EcoClasificaTests(unittest.TestCase):
    """Batería de pruebas unitarias para el motor de reglas de EcoClasifica."""

    def test_normalize_material_name_accepts_aliases(self) -> None:
        """Verifica que descripciones con alias se mapeen a categorías reconocidas."""
        self.assertEqual(normalize_material_name(TEST_GLASS_DESCRIPTION), CATEGORY_GLASS)
        self.assertEqual(normalize_material_name(TEST_PAPER_DESCRIPTION), CATEGORY_PAPER)
        self.assertEqual(normalize_material_name(TEST_METAL_DESCRIPTION), CATEGORY_METAL)

    def test_normalize_material_name_handles_none_and_unknown(self) -> None:
        """Verifica que None o entradas no reconocidas retornen DEFAULT_CATEGORY."""
        self.assertEqual(normalize_material_name(None), DEFAULT_CATEGORY)
        self.assertEqual(normalize_material_name(TEST_UNKNOWN_DESCRIPTION), DEFAULT_CATEGORY)

    def test_calculate_confidence_returns_known_values(self) -> None:
        """Verifica que los niveles de confianza coincidan con las constantes definidas."""
        self.assertEqual(calculate_confidence(CATEGORY_PLASTIC), CONFIDENCE_PLASTIC)
        self.assertEqual(calculate_confidence(CATEGORY_METAL), CONFIDENCE_METAL)
        self.assertEqual(calculate_confidence(CATEGORY_PAPER), CONFIDENCE_PAPER)
        self.assertEqual(calculate_confidence(CATEGORY_GLASS), CONFIDENCE_GLASS)
        self.assertEqual(calculate_confidence(CATEGORY_OTHER), CONFIDENCE_OTHER)

    def test_classify_residue_handles_unknown_items(self) -> None:
        """Verifica que un residuo desconocido se clasifique con categoría y confianza base."""
        result = classify_residue(TEST_UNKNOWN_DESCRIPTION)
        self.assertEqual(result["category"], DEFAULT_CATEGORY)
        self.assertEqual(result["confidence"], CONFIDENCE_OTHER)

    def test_classify_residue_reduces_confidence_on_small_image(self) -> None:
        """Verifica que una imagen pequeña reduzca proporcionalmente la confianza estimada."""
        result = classify_residue(TEST_PLASTIC_DESCRIPTION, image_size=TEST_SMALL_IMAGE_SIZE)
        self.assertEqual(result["category"], CATEGORY_PLASTIC)
        self.assertLess(result["confidence"], CONFIDENCE_PLASTIC)
        expected_reduced = round(CONFIDENCE_PLASTIC * CONFIDENCE_ADJUSTMENT, ROUND_DECIMALS)
        self.assertEqual(result["confidence"], expected_reduced)

    def test_classify_residue_resolution_boundary(self) -> None:
        """Verifica el umbral límite de LOW_RESOLUTION_THRESHOLD para el ajuste de confianza."""
        # Por debajo del umbral -> aplica ajuste reductor
        below_result = classify_residue(
            TEST_PLASTIC_DESCRIPTION,
            image_size=(LOW_RESOLUTION_THRESHOLD - 1, LOW_RESOLUTION_THRESHOLD - 1),
        )
        self.assertLess(below_result["confidence"], CONFIDENCE_PLASTIC)

        # En el umbral exacto -> mantiene confianza completa
        boundary_result = classify_residue(
            TEST_PLASTIC_DESCRIPTION,
            image_size=TEST_THRESHOLD_IMAGE_SIZE,
        )
        self.assertEqual(boundary_result["confidence"], CONFIDENCE_PLASTIC)

    def test_constants_values_and_types(self) -> None:
        """Verifica que todas las constantes UPPER_CASE tengan tipos y valores correctos."""
        self.assertIsInstance(DEFAULT_CATEGORY, str)
        self.assertIsInstance(CONFIDENCE_PLASTIC, float)
        self.assertIsInstance(LOW_RESOLUTION_THRESHOLD, int)
        self.assertIsInstance(ROUND_DECIMALS, int)
        self.assertEqual(ROUND_DECIMALS, 2)
        self.assertEqual(LOW_RESOLUTION_THRESHOLD, 128)


if __name__ == "__main__":
    unittest.main()
