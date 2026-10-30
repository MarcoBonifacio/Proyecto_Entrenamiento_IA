import unittest

from src.ecoclasifica import (
    calculate_confidence,
    classify_residue,
    normalize_material_name,
)


class EcoClasificaTests(unittest.TestCase):
    def test_normalize_material_name_accepts_aliases(self):
        self.assertEqual(normalize_material_name("Botella de vidrio"), "glass")
        self.assertEqual(normalize_material_name("Caja de cartón"), "paper")
        self.assertEqual(normalize_material_name("Lata de aluminio"), "metal")

    def test_calculate_confidence_returns_known_values(self):
        self.assertEqual(calculate_confidence("plastic"), 0.92)
        self.assertEqual(calculate_confidence("metal"), 0.91)

    def test_classify_residue_handles_unknown_items(self):
        result = classify_residue("objeto raro")
        self.assertEqual(result["category"], "other")
        self.assertEqual(result["confidence"], 0.5)

    def test_classify_residue_reduces_confidence_on_small_image(self):
        result = classify_residue("Envase de plastico", image_size=(64, 64))
        self.assertEqual(result["category"], "plastic")
        self.assertLess(result["confidence"], 0.92)


if __name__ == "__main__":
    unittest.main()
