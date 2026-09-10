"""
Automated Test Suite for VERBA:
1. Normalization & Contraction Expansion
2. Phrase Resolution & Synonym Handling
3. Semantic Parsing (Intents, Negation, WH-Questions, Commands)
4. ISL Intermediate Representation (SOV order, Temporal-first, Negation)
5. Sign Asset Registry & Security (Path Traversal Rejection, Invalid IDs)
6. Missing-Sign UX & Translation Status (SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED)
7. REST API Endpoints (POST /api/english-to-sign, GET /api/signs, GET /health)
8. End-to-End Model Inference on Amrita trained model
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "ml_pipeline"))

from nlp_pipeline import (
    NormalizationLayer,
    SemanticParser,
    IslTransformationLayer,
    SignAssetRegistry,
    EnglishToSignPipeline,
)
from fastapi.testclient import TestClient
import main


class TestNormalization(unittest.TestCase):
    def test_uppercase_and_whitespace(self):
        self.assertEqual(NormalizationLayer.normalize("  HELLO!  "), "hello")
        self.assertEqual(NormalizationLayer.normalize("HELLO WORLD"), "hello world")

    def test_contractions(self):
        self.assertEqual(NormalizationLayer.normalize("don't eat this"), "do not eat this")
        self.assertEqual(NormalizationLayer.normalize("can't help"), "cannot help")
        self.assertEqual(NormalizationLayer.normalize("i'm ready"), "i am ready")

    def test_synonyms_and_lemmas(self):
        norm = NormalizationLayer.normalize("my stomach is hurting")
        self.assertIn("pain", norm)
        norm_med = NormalizationLayer.normalize("take these tablets")
        self.assertIn("medicine", norm_med)

    def test_phrase_canonicalization(self):
        norm = NormalizationLayer.normalize("good morning doctor")
        self.assertIn("good_morning", norm)
        norm_ty = NormalizationLayer.normalize("thank you very much")
        self.assertIn("thank_you", norm_ty)


class TestSemanticParser(unittest.TestCase):
    def test_request_and_action(self):
        parsed = SemanticParser.parse("hello can you help me?")
        self.assertEqual(parsed["action"], "help")
        self.assertEqual(parsed["intent"], "greeting")

    def test_negation_preservation(self):
        parsed = SemanticParser.parse("do not eat this")
        self.assertTrue(parsed["negation"])
        self.assertEqual(parsed["action"], "eat")

    def test_wh_question(self):
        parsed = SemanticParser.parse("where is the doctor?")
        self.assertEqual(parsed["intent"], "question")
        self.assertEqual(parsed["question_type"], "wh")
        self.assertTrue(parsed["location_query"])

    def test_command(self):
        parsed = SemanticParser.parse("take your medicine")
        self.assertEqual(parsed["intent"], "command")
        self.assertEqual(parsed["action"], "take")
        self.assertEqual(parsed["object"], "medicine")


class TestIslTransformation(unittest.TestCase):
    def test_demo_1_isl_signs(self):
        parsed = SemanticParser.parse("hello can you help me?")
        isl = IslTransformationLayer.transform(parsed, "hello can you help me?")
        glosses = [s["gloss"] for s in isl["signs"]]
        self.assertIn("HELLO", glosses)
        self.assertIn("HELP", glosses)
        # Verify SOV / greeting order: HELLO comes before HELP
        self.assertLess(glosses.index("HELLO"), glosses.index("HELP"))

    def test_sov_ordering(self):
        parsed = SemanticParser.parse("take your medicine")
        isl = IslTransformationLayer.transform(parsed, "take your medicine")
        glosses = [s["gloss"] for s in isl["signs"]]
        # In ISL, Object (MEDICINE) precedes Action (TAKE)
        self.assertIn("MEDICINE", glosses)
        self.assertIn("TAKE", glosses)
        self.assertLess(glosses.index("MEDICINE"), glosses.index("TAKE"))

    def test_negation_at_end(self):
        parsed = SemanticParser.parse("do not eat")
        isl = IslTransformationLayer.transform(parsed, "do not eat")
        glosses = [s["gloss"] for s in isl["signs"]]
        self.assertIn("NO", glosses)
        # Negation comes after action
        self.assertLess(glosses.index("EAT"), glosses.index("NO"))


class TestAssetRegistryAndSecurity(unittest.TestCase):
    def setUp(self):
        self.registry = SignAssetRegistry()

    def test_approved_sign_resolution(self):
        hello = self.registry.get("hello")
        self.assertIsNotNone(hello)
        self.assertEqual(hello["gloss"], "HELLO")
        self.assertTrue(hello["verified"])
        self.assertTrue(hello["url"].startswith("/media/isl/clips/"))

    def test_reject_path_traversal(self):
        self.assertIsNone(self.registry.get("../../../etc/passwd"))
        self.assertIsNone(self.registry.get("..\\..\\windows\\win.ini"))
        self.assertIsNone(self.registry.get("hello/world"))
        self.assertIsNone(self.registry.get("hello\x00world"))

    def test_reject_invalid_sign_id(self):
        self.assertIsNone(self.registry.get("invalid sign id with spaces"))
        self.assertIsNone(self.registry.get("!@#$%^&*()"))


class TestMissingSignUX(unittest.TestCase):
    def setUp(self):
        self.pipe = EnglishToSignPipeline()

    def test_fully_supported(self):
        res = self.pipe.translate("Hello, can you help me?")
        self.assertEqual(res["translation_status"], "supported")
        self.assertEqual(len(res["missing_concepts"]), 0)
        self.assertIn("HELLO", res["available_signs"])
        self.assertIn("HELP", res["available_signs"])
        self.assertTrue(len(res["realization"]["assets"]) >= 2)

    def test_partially_supported(self):
        res = self.pipe.translate("Please take your medicine after lunch.")
        self.assertEqual(res["translation_status"], "partially_supported")
        self.assertIn("AFTER_LUNCH", res["missing_concepts"])
        self.assertIn("TAKE", res["available_signs"])
        self.assertIn("MEDICINE", res["available_signs"])
        self.assertIn("Partially translated", res["fallback_text"])

    def test_unsupported_concepts(self):
        res = self.pipe.translate("Supercalifragilisticexpialidocious quantum physics")
        self.assertEqual(res["translation_status"], "unsupported")
        self.assertTrue(len(res["missing_concepts"]) > 0)
        self.assertEqual(len(res["realization"]["assets"]), 0)


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["model"])
        self.assertGreater(data["registry_signs"], 0)

    def test_signs_list(self):
        r = self.client.get("/api/signs")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertGreater(data["count"], 20)
        self.assertIn("hello", data["signs"])
        self.assertIn("help", data["signs"])

    def test_english_to_sign_demo_1(self):
        r = self.client.post("/api/english-to-sign", json={"text": "Hello, can you help me?"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["translation_status"], "supported")
        self.assertIn("HELLO", data["available_signs"])
        self.assertIn("HELP", data["available_signs"])
        self.assertEqual(data["missing_concepts"], [])
        self.assertEqual(data["realization"]["renderer"], "video")
        self.assertTrue(len(data["realization"]["assets"]) >= 2)

    def test_english_to_sign_demo_2(self):
        r = self.client.post("/api/english-to-sign", json={"text": "Please take your medicine after lunch."})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["translation_status"], "partially_supported")
        self.assertIn("AFTER_LUNCH", data["missing_concepts"])
        self.assertIn("TAKE", data["available_signs"])
        self.assertIn("MEDICINE", data["available_signs"])

    def test_english_to_sign_empty_text(self):
        r = self.client.post("/api/english-to-sign", json={"text": "   "})
        self.assertEqual(r.status_code, 400)

    def test_english_to_sign_oversized(self):
        r = self.client.post("/api/english-to-sign", json={"text": "a" * 1500})
        self.assertEqual(r.status_code, 400)


class TestModelInference(unittest.TestCase):
    def test_model_loading_and_prediction(self):
        import onnxruntime as ort
        import numpy as np

        onnx_path = os.path.join(REPO_ROOT, "backend", "models", "model.onnx")
        self.assertTrue(os.path.exists(onnx_path))
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.assertEqual(sess.get_inputs()[0].name, "x")
        self.assertEqual(sess.get_outputs()[0].name, "logits")

        # Test inference with dummy normalized window
        dummy = np.zeros((1, 1322), dtype=np.float32)
        logits = sess.run(None, {"x": dummy})[0]
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 37)


if __name__ == "__main__":
    unittest.main(verbosity=2)
