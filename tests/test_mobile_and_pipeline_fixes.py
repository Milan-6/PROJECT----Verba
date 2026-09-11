"""
Unit and integration tests for:
1. MediaPipe model bundling and APK verification
2. Video-gated scenario suggestions (Feature 3)
3. Video-gated scenario quick phrases (Feature 4)
4. Non-restriction of free-form user typing
5. Responsive CSS rules audit
"""
import unittest
import os
import json
import zipfile
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from main import load_vocab, app
from nlp_pipeline import EnglishToSignPipeline, is_sentence_fully_supported
from fastapi.testclient import TestClient

class TestMobileAndPipelineFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = EnglishToSignPipeline()
        cls.client = TestClient(app)
        cls.scenarios = ["hospital", "bank", "bus_stop", "railway", "shopping", "government"]

    def test_task_models_bundled_in_public_and_dist(self):
        """Verify hand and pose task models exist in public/mediapipe and dist/mediapipe (>1MB)."""
        for base in ["public", "dist"]:
            hand_path = os.path.join(REPO_ROOT, "frontend", base, "mediapipe", "hand_landmarker.task")
            pose_path = os.path.join(REPO_ROOT, "frontend", base, "mediapipe", "pose_landmarker_lite.task")
            self.assertTrue(os.path.isfile(hand_path), f"Missing {hand_path}")
            self.assertTrue(os.path.isfile(pose_path), f"Missing {pose_path}")
            self.assertGreater(os.path.getsize(hand_path), 1024 * 1024, "hand_landmarker.task too small")
            self.assertGreater(os.path.getsize(pose_path), 1024 * 1024, "pose_landmarker_lite.task too small")

    def test_apk_contains_mediapipe_models(self):
        """Verify app-debug.apk packages both task models in assets/public/mediapipe/."""
        apk_path = os.path.join(
            REPO_ROOT, "frontend", "android", "app", "build", "outputs", "apk", "debug", "app-debug.apk"
        )
        self.assertTrue(os.path.isfile(apk_path), f"APK not found at {apk_path}")
        with zipfile.ZipFile(apk_path, "r") as z:
            names = z.namelist()
            self.assertIn("assets/public/mediapipe/hand_landmarker.task", names)
            self.assertIn("assets/public/mediapipe/pose_landmarker_lite.task", names)
            hand_info = z.getinfo("assets/public/mediapipe/hand_landmarker.task")
            pose_info = z.getinfo("assets/public/mediapipe/pose_landmarker_lite.task")
            self.assertGreater(hand_info.file_size, 1024 * 1024)
            self.assertGreater(pose_info.file_size, 1024 * 1024)

    def test_css_responsive_rules(self):
        """Audit styles.css to ensure 900px breakpoint collapses workspace and provides touch targets."""
        css_path = os.path.join(REPO_ROOT, "frontend", "src", "styles.css")
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        self.assertIn("@media (max-width: 900px)", css)
        self.assertIn("grid-template-columns: 1fr;", css)
        self.assertIn("min-height: 44px;", css)
        self.assertIn("-webkit-overflow-scrolling: touch;", css)

    def test_all_scenario_presets_are_100_percent_video_backed(self):
        """Feature 3: 100% of glosses in scenario presets MUST resolve to real video clips."""
        assets_dir = os.path.join(REPO_ROOT, "assets", "isl")
        for sc in self.scenarios:
            v = load_vocab(sc)
            presets = v.get("presets", [])
            self.assertGreater(len(presets), 0, f"Scenario {sc} has no presets")
            for p in presets:
                self.assertTrue(
                    is_sentence_fully_supported(p, scenario=sc, pipeline=self.nlp),
                    f"Preset '{p}' in scenario '{sc}' failed video-gating predicate!"
                )
                res = self.nlp.translate(p, scenario=sc)
                self.assertEqual(res.get("translation_status"), "supported", f"'{p}' not fully supported")
                self.assertEqual(len(res.get("missing_concepts", [])), 0, f"'{p}' has missing concepts")
                for item in res.get("items", []):
                    self.assertIsNotNone(item.get("clip"), f"'{p}' has an item without clip: {item}")
                    self.assertNotEqual(item.get("match_tier"), "fingerspelling", f"'{p}' has fingerspelling fallback")
                    # Check physical file exists
                    rel = item["clip"].replace("/media/isl/", "")
                    phys = os.path.join(assets_dir, rel)
                    self.assertTrue(os.path.isfile(phys), f"Clip file missing on disk: {phys}")

    def test_all_scenario_quick_phrases_are_100_percent_video_backed(self):
        """Feature 4: 100% of glosses in scenario quick phrases MUST resolve to real video clips."""
        assets_dir = os.path.join(REPO_ROOT, "assets", "isl")
        for sc in self.scenarios:
            v = load_vocab(sc)
            qp = v.get("quick_phrases", [])
            self.assertGreater(len(qp), 0, f"Scenario {sc} has no quick phrases")
            for p in qp:
                self.assertTrue(
                    is_sentence_fully_supported(p, scenario=sc, pipeline=self.nlp),
                    f"Quick phrase '{p}' in scenario '{sc}' failed video-gating predicate!"
                )
                res = self.nlp.translate(p, scenario=sc)
                self.assertEqual(res.get("translation_status"), "supported", f"'{p}' not fully supported")
                self.assertEqual(len(res.get("missing_concepts", [])), 0, f"'{p}' has missing concepts")
                for item in res.get("items", []):
                    self.assertIsNotNone(item.get("clip"), f"'{p}' has an item without clip: {item}")
                    self.assertNotEqual(item.get("match_tier"), "fingerspelling", f"'{p}' has fingerspelling fallback")
                    rel = item["clip"].replace("/media/isl/", "")
                    phys = os.path.join(assets_dir, rel)
                    self.assertTrue(os.path.isfile(phys), f"Clip file missing on disk: {phys}")

    def test_freeform_input_not_restricted(self):
        """Video-gating must only filter suggestions/quick phrases, NEVER free-form typed input."""
        freeform_text = "I would like to deposit five thousand dollars and check my account balance."
        resp = self.client.post("/api/english-to-sign", json={"text": freeform_text, "scenario": "bank"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("items", data)
        self.assertIn("translation_status", data)
        self.assertIn(data["translation_status"], ["partially_supported", "unsupported"])

if __name__ == "__main__":
    unittest.main()
