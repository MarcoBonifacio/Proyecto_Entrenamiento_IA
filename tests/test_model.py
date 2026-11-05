"""Tests unitarios para el módulo del modelo EcoClasifica."""

import unittest
from unittest.mock import patch, MagicMock

import torch

from src.model import EcoClasificaModel, classify_waste


class TestModelLabels(unittest.TestCase):

    def test_model_labels_count(self):
        self.assertEqual(len(EcoClasificaModel.LABELS), 6)

    def test_model_labels_content(self):
        expected = {"Cartón", "Vidrio", "Metal", "Papel", "Plástico", "Basura general"}
        actual = set(EcoClasificaModel.LABELS.values())
        self.assertEqual(actual, expected)


class TestModelInitialization(unittest.TestCase):

    @patch("src.model.ResNetForImageClassification.from_pretrained")
    @patch("src.model.AutoImageProcessor.from_pretrained")
    def test_model_initialization(self, mock_proc, mock_model):
        mock_proc.return_value = MagicMock()
        m = MagicMock()
        m.eval.return_value = m
        mock_model.return_value = m
        model = EcoClasificaModel()
        self.assertIsNotNone(model)


class TestModelPredict(unittest.TestCase):

    def setUp(self):
        self.p1 = patch("src.model.ResNetForImageClassification.from_pretrained")
        self.p2 = patch("src.model.AutoImageProcessor.from_pretrained")
        self.p3 = patch("src.model.load_and_preprocess_image")
        self.mock_model_cls = self.p1.start()
        self.mock_proc_cls = self.p2.start()
        self.mock_prep = self.p3.start()

        self.mock_proc_cls.return_value = MagicMock()
        self.mock_prep.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        mock_out = MagicMock()
        mock_out.logits = torch.tensor([[2.0, 0.5, 0.1, 0.3, 0.8, 0.2]])
        m = MagicMock()
        m.return_value = mock_out
        m.eval.return_value = m
        self.mock_model_cls.return_value = m

        self.model = EcoClasificaModel()

    def tearDown(self):
        self.p1.stop()
        self.p2.stop()
        self.p3.stop()

    def test_predict_returns_correct_structure(self):
        from PIL import Image
        result = self.model.predict(Image.new("RGB", (224, 224)))
        self.assertIn("categoria", result)
        self.assertIn("confianza", result)
        self.assertIn("todas_las_predicciones", result)

    def test_predict_confidence_between_0_and_1(self):
        from PIL import Image
        result = self.model.predict(Image.new("RGB", (224, 224)))
        self.assertGreaterEqual(result["confianza"], 0.0)
        self.assertLessEqual(result["confianza"], 1.0)


class TestClassifyWaste(unittest.TestCase):

    @patch("src.model.load_and_preprocess_image")
    @patch("src.model.ResNetForImageClassification.from_pretrained")
    @patch("src.model.AutoImageProcessor.from_pretrained")
    def test_classify_waste_function(self, mock_proc, mock_model_cls, mock_prep):
        from PIL import Image
        mock_proc.return_value = MagicMock()
        mock_prep.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mock_out = MagicMock()
        mock_out.logits = torch.tensor([[1.0, 0.5, 0.2, 0.3, 0.8, 0.1]])
        m = MagicMock()
        m.return_value = mock_out
        m.eval.return_value = m
        mock_model_cls.return_value = m

        result = classify_waste(Image.new("RGB", (224, 224)))
        self.assertIn("categoria", result)
        self.assertIn("confianza", result)


if __name__ == "__main__":
    unittest.main()
