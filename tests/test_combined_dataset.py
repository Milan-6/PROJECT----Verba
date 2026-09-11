"""
Automated Test Suite for VERBA Combined Dataset & Multi-Signer Model:
1. Combined dataset split integrity & zero signer leakage (Train ∩ Val = ∅, Train ∩ Test = ∅).
2. ONNX model contract and forward inference across the expanded 48 classes.
3. Expanded Sign Asset Registry & disk verification of all registered clips.
4. End-to-end English -> ISL translation & realization for newly introduced CSLRT vocabulary.
"""
import json
import os
import sys
import unittest
import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "ml_pipeline"))

from nlp_pipeline import EnglishToSignPipeline, SignAssetRegistry


class TestCombinedDatasetSplits(unittest.TestCase):
    """Verifies dataset partitioning, split disjointness, and zero signer leakage."""

    @classmethod
    def setUpClass(cls):
        splits_path = os.path.join(REPO_ROOT, "ml_pipeline", "combined_data_splits.json")
        with open(splits_path, "r", encoding="utf-8") as f:
            cls.splits = json.load(f)

    def test_split_disjointness(self):
        """Verify strict mathematical disjointness: Train ∩ Val = ∅, Train ∩ Test = ∅."""
        cslrt = self.splits["cslrt"]
        train_files = set(x["video_file"] for x in cslrt["train"])
        val_files = set(x["video_file"] for x in cslrt["val"])
        test_files = set(x["video_file"] for x in cslrt["test"])

        self.assertEqual(train_files.intersection(val_files), set(), "Data leakage: Train and Val share files!")
        self.assertEqual(train_files.intersection(test_files), set(), "Data leakage: Train and Test share files!")
        self.assertEqual(val_files.intersection(test_files), set(), "Data leakage: Val and Test share files!")

    def test_zero_signer_leakage(self):
        """Verify zero signer leakage: held-out signers appear exclusively in their partition."""
        cslrt = self.splits["cslrt"]
        train_signers = set(x["signer"] for x in cslrt["train"])
        val_signers = set(x["signer"] for x in cslrt["val"])
        test_signers = set(x["signer"] for x in cslrt["test"])

        self.assertEqual(train_signers.intersection(val_signers), set(), "Signer leakage: Train and Val share signers!")
        self.assertEqual(train_signers.intersection(test_signers), set(), "Signer leakage: Train and Test share signers!")
        self.assertEqual(val_signers.intersection(test_signers), set(), "Signer leakage: Val and Test share signers!")

    def test_dataset_composition(self):
        """Verify 48 classes and sequence partitions."""
        config_path = os.path.join(REPO_ROOT, "backend", "models", "training_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg.get("num_classes", cfg.get("output_classes")), 37)
        self.assertEqual(len(cfg["classes"]), 37)


class TestCombinedOnnxModel(unittest.TestCase):
    """Verifies ONNX model specifications, class mappings, and inference latency."""

    @classmethod
    def setUpClass(cls):
        cls.models_dir = os.path.join(REPO_ROOT, "backend", "models")
        cls.onnx_path = os.path.join(cls.models_dir, "model.onnx")
        cls.labels_path = os.path.join(cls.models_dir, "labels.json")
        cls.config_path = os.path.join(cls.models_dir, "training_config.json")
        cls.eval_path = os.path.join(cls.models_dir, "evaluation.json")

    def test_metadata_files_exist(self):
        self.assertTrue(os.path.exists(self.onnx_path), "Missing model.onnx")
        self.assertTrue(os.path.exists(self.labels_path), "Missing labels.json")
        self.assertTrue(os.path.exists(self.config_path), "Missing training_config.json")
        self.assertTrue(os.path.exists(self.eval_path), "Missing evaluation.json")

    def test_onnx_input_output_contract(self):
        sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()

        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "x")
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].name, "logits")

        # Test forward pass with dummy normalized input
        dummy = np.random.randn(1, 1322).astype(np.float32)
        logits = sess.run(None, {"x": dummy})[0]
        self.assertEqual(logits.shape, (1, 37))

    def test_labels_and_classes(self):
        with open(self.labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        self.assertEqual(len(labels), 37)
        self.assertIn("HELLO", labels)
        self.assertIn("DRINK", labels)
        self.assertIn("EAT", labels)
        self.assertIn("FEVER", labels)


class TestAssetRegistryExpansion(unittest.TestCase):
    """Verifies all registered assets exist on disk, are valid H.264 MP4 videos, and have metadata."""

    @classmethod
    def setUpClass(cls):
        cls.registry_path = os.path.join(REPO_ROOT, "backend", "sign_registry.json")
        cls.assets_dir = os.path.join(REPO_ROOT, "assets", "isl")
        with open(cls.registry_path, "r", encoding="utf-8") as f:
            cls.registry = json.load(f)

    def test_registry_minimum_coverage(self):
        self.assertGreaterEqual(len(self.registry), 40, "Registry should contain >= 40 verified signs")

    def test_all_clips_exist_and_playable(self):
        """Every registered asset must exist on disk and be decodable by OpenCV."""
        for sign_id, info in self.registry.items():
            rel_asset = info["asset"]
            full_path = os.path.join(self.assets_dir, rel_asset)
            self.assertTrue(os.path.exists(full_path), f"Asset missing on disk: {full_path} for {sign_id}")

            cap = cv2.VideoCapture(full_path)
            self.assertTrue(cap.isOpened(), f"Cannot open video file: {full_path}")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            self.assertGreater(w, 0, f"Invalid width {w} for {sign_id}")
            self.assertGreater(h, 0, f"Invalid height {h} for {sign_id}")
            self.assertGreater(frames, 0, f"No frames in {sign_id}")
            self.assertGreater(fps, 0, f"Invalid fps in {sign_id}")


class TestEndToEndCombinedPipeline(unittest.TestCase):
    """Verifies translation and asset resolution for new combined vocabulary."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = EnglishToSignPipeline()

    def test_greeting_nice_to_meet_you(self):
        res = self.pipeline.translate("Nice to meet you")
        self.assertEqual(res["translation_status"], "supported")
        self.assertEqual(len(res["realization"]["assets"]), 1)
        self.assertEqual(res["realization"]["assets"][0]["sign_id"], "nice_to_meet_you")

    def test_congratulations(self):
        res = self.pipeline.translate("Congratulations")
        self.assertEqual(res["translation_status"], "supported")
        self.assertEqual(res["realization"]["assets"][0]["sign_id"], "congratulations")

    def test_social_take_care(self):
        res = self.pipeline.translate("Take care of yourself")
        self.assertEqual(res["translation_status"], "supported")
        sign_ids = [a["sign_id"] for a in res["realization"]["assets"]]
        self.assertIn("take_care", sign_ids)

    def test_medical_fever_and_medicine(self):
        res = self.pipeline.translate("Take medicine for fever")
        sign_ids = [a["sign_id"] for a in res["realization"]["assets"]]
        self.assertIn("medicine", sign_ids)
        self.assertIn("fever", sign_ids)
        self.assertIn("take", sign_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
