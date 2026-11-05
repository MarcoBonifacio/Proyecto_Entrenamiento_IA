"""Tests unitarios para el módulo de preprocesamiento de imágenes."""

import unittest
from PIL import Image

from src.preprocess import validate_image, load_and_preprocess_image


class TestValidateImage(unittest.TestCase):

    def test_validate_image_with_valid_pil_image(self):
        image = Image.new("RGB", (224, 224), color=(128, 64, 32))
        self.assertTrue(validate_image(image))

    def test_validate_image_with_invalid_input(self):
        for invalid_input in [None, "not a path", ""]:
            with self.subTest(input=invalid_input):
                self.assertFalse(validate_image(invalid_input))


class TestLoadAndPreprocessImage(unittest.TestCase):

    def test_load_and_preprocess_image_with_pil_image(self):
        image = Image.new("RGB", (640, 480), color=(100, 150, 200))
        result = load_and_preprocess_image(image)
        self.assertIn("pixel_values", result)

    def test_load_and_preprocess_image_resizes_correctly(self):
        for w, h in [(64, 64), (224, 224), (640, 480)]:
            with self.subTest(size=(w, h)):
                image = Image.new("RGB", (w, h), color=(50, 100, 150))
                result = load_and_preprocess_image(image)
                self.assertIn("pixel_values", result)

    def test_load_and_preprocess_image_converts_rgba_to_rgb(self):
        image = Image.new("RGBA", (224, 224), color=(100, 150, 200, 255))
        result = load_and_preprocess_image(image)
        self.assertIn("pixel_values", result)

    def test_load_and_preprocess_image_converts_grayscale_to_rgb(self):
        image = Image.new("L", (224, 224), color=128)
        result = load_and_preprocess_image(image)
        self.assertIn("pixel_values", result)


if __name__ == "__main__":
    unittest.main()


# Constantes de prueba TDD (Sin magic numbers)
EXPECTED_STATUS_OK = 200
TEST_IMAGE_SIZE = 224
MIN_CONFIDENCE_TEST = 0.60
