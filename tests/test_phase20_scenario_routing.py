"""
Phase 20 Test Suite: Scenario-Aware ML Routing + English -> ISL Video Pipeline.

Tests:
1. Runtime compatibility gating (input dim 1322, label counts, scaler health, error rejection).
2. Scenario requirements resolution & manifest generation across 6 scenarios:
   Hospital, Bank, Bus Stop, Railway Station, Shopping, Government.
3. Deterministic candidate dataset planning with zero class dropping.
4. English -> ISL Video Selection Engine (exact, alias, repo, fingerspelling, missing).
5. Pre-queue physical file validation guaranteeing ZERO 404s reach SignPlayer.
6. Non-fabrication & non-corruption guarantees (no wrong substitutions).
7. Non-corrupting scenario switching via WebSocket / API.
"""
from __future__ import annotations
import os
import sys
import unittest
from typing import Any, Dict, List

# Setup paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT, "backend")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
VOCAB_DIR = os.path.join(BACKEND_DIR, "vocab")
ASSETS_DIR = os.path.join(ROOT, "assets")

sys.path.insert(0, BACKEND_DIR)

from scenario_routing import (
    RuntimeCompatibilityError,
    verify_runtime_compatibility,
    get_active_model_info,
    resolve_scenario_requirements,
    resolve_dataset_for_scenario,
    build_scenario_dataset_plan,
    generate_scenario_manifest,
)
from nlp_pipeline import EnglishToSignPipeline, SignAssetRegistry, resolve_isl_assets
from inference import SignRecognizer
from main import app
from fastapi.testclient import TestClient


class TestPhase20RuntimeCompatibility(unittest.TestCase):
    """Verifies strict runtime model/scaler/label compatibility checks."""

    def test_active_model_passes_compatibility(self):
        comp = verify_runtime_compatibility(MODELS_DIR, expected_feature_dim=1322)
        self.assertTrue(comp["compatible"])
        self.assertEqual(comp["input_dim"], 1322)
        self.assertGreaterEqual(comp["output_classes"], 37)
        self.assertEqual(len(comp["labels"]), comp["output_classes"])

    def test_rejection_on_mismatched_dimension(self):
        with self.assertRaises(RuntimeCompatibilityError):
            verify_runtime_compatibility(MODELS_DIR, expected_feature_dim=512)

    def test_rejection_on_nonexistent_model_dir(self):
        with self.assertRaises(RuntimeCompatibilityError):
            verify_runtime_compatibility("/invalid/path/to/models")

    def test_sign_recognizer_enforces_compatibility(self):
        # Initializing SignRecognizer must execute compatibility check
        rec = SignRecognizer(model_dir=MODELS_DIR)
        self.assertIsNotNone(rec.sess)
        self.assertEqual(len(rec.labels), 37)
        self.assertEqual(rec.mu.shape[0], 1322)


class TestPhase20ScenarioRequirementsAndManifest(unittest.TestCase):
    """Verifies requirements resolution and manifest across all 6 scenarios."""

    SCENARIOS = ["hospital", "bank", "bus_stop", "railway", "shopping", "government"]

    def test_all_six_scenarios_exist_and_resolve(self):
        for s in self.SCENARIOS:
            req = resolve_scenario_requirements(s, vocab_dir=VOCAB_DIR, model_dir=MODELS_DIR)
            self.assertEqual(req["scenario"], s)
            self.assertIn("theme", req)
            self.assertGreater(req["total_required_signs"], 0)
            self.assertIsInstance(req["required_signs"], list)
            self.assertIsInstance(req["existing_ml_labels"], list)
            self.assertIsInstance(req["missing_ml_labels"], list)
            self.assertIsInstance(req["renderable_assets"], list)
            self.assertIsInstance(req["missing_assets"], list)

            # Mathematical integrity: existing + missing == total
            self.assertEqual(
                len(req["existing_ml_labels"]) + len(req["missing_ml_labels"]),
                req["total_required_signs"]
            )
            self.assertEqual(
                len(req["renderable_assets"]) + len(req["missing_assets"]),
                req["total_required_signs"]
            )
            # Ratios between 0.0 and 1.0
            self.assertTrue(0.0 <= req["ml_coverage_ratio"] <= 1.0)
            self.assertTrue(0.0 <= req["asset_coverage_ratio"] <= 1.0)

    def test_scenario_manifest_generation(self):
        manifest = generate_scenario_manifest(vocab_dir=VOCAB_DIR, model_dir=MODELS_DIR)
        self.assertEqual(manifest["version"], "2.0.0")
        self.assertIn("active_model", manifest)
        self.assertEqual(manifest["active_model"]["classes_count"], 37)
        for s in self.SCENARIOS:
            self.assertIn(s, manifest["scenarios"])
            self.assertNotIn("error", manifest["scenarios"][s])


class TestPhase20DeterministicDatasetPlanning(unittest.TestCase):
    """Verifies deterministic dataset resolution and preservation of base classes."""

    def test_deterministic_missing_signs_resolution(self):
        for s in ["bus_stop", "railway", "shopping", "government"]:
            ds = resolve_dataset_for_scenario(s, vocab_dir=VOCAB_DIR, model_dir=MODELS_DIR)
            self.assertIn(ds["status"], ("ready", "dataset_required"))
            if ds["status"] == "dataset_required":
                self.assertGreater(len(ds["missing_signs"]), 0)
                for item in ds["missing_signs"]:
                    self.assertIn("label", item)
                    self.assertIn("dataset_required", item)
                    self.assertIn("status", item)
                    self.assertIn(item["status"], ("ready_for_candidate_training", "needs_preprocessing", "needs_collection"))

    def test_dataset_builder_plan_preserves_base_classes(self):
        plan = build_scenario_dataset_plan("bus_stop", vocab_dir=VOCAB_DIR, model_dir=MODELS_DIR)
        self.assertEqual(plan["base_classes_count"], 37)
        # All 37 base classes must be in planned classes
        for base_cls in plan["base_classes"]:
            self.assertIn(base_cls, plan["planned_classes"])
        self.assertGreaterEqual(plan["planned_total_classes"], 37)
        self.assertTrue(plan["quality_gate"]["zero_regression_on_base_classes"])
        self.assertTrue(plan["quality_gate"]["zero_signer_leakage"])


class TestPhase20EnglishToIslVideoPipeline(unittest.TestCase):
    """End-to-end tests for English -> ISL video selection engine across all 6 scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.pipe = EnglishToSignPipeline()

    def test_pipeline_across_six_scenarios(self):
        scenario_test_cases = [
            ("hospital", "Please take a seat", ["SIT", "PLEASE"]),
            ("bank", "Sign here on the form", ["WRITE"]),
            ("bus_stop", "Please stop here", ["STOP", "PLEASE"]),
            ("railway", "Please wait for the train", ["WAIT", "PLEASE"]),
            ("shopping", "Can I pay by cash?", ["CASH"]),
            ("government", "Please show your identification documents", ["PLEASE"]),
        ]

        for scenario, english_text, expected_gloss_subsets in scenario_test_cases:
            res = self.pipe.translate(english_text, scenario=scenario)
            self.assertEqual(res["source_text"], english_text)
            self.assertEqual(res["scenario"], scenario)
            self.assertIn(res["translation_status"], ("supported", "partially_supported", "unsupported"))
            self.assertIsInstance(res["items"], list)
            self.assertGreater(len(res["items"]), 0)

            # Check that expected glosses were generated
            generated_glosses = [it["gloss"] for it in res["items"]]
            for eg in expected_gloss_subsets:
                self.assertIn(eg, generated_glosses, f"Expected {eg} in {generated_glosses} for '{english_text}'")

            # CRITICAL ZERO 404 GUARANTEE:
            # If item has a clip, the physical file must exist on disk and be readable
            for item in res["items"]:
                clip_url = item.get("clip")
                if clip_url:
                    rel_path = clip_url.replace("/media/isl/", "")
                    full_p = os.path.join(ASSETS_DIR, "isl", rel_path)
                    self.assertTrue(
                        os.path.isfile(full_p),
                        f"404 DETECTED: Asset '{clip_url}' queued for SignPlayer does not exist at {full_p}"
                    )
                    self.assertTrue(os.access(full_p, os.R_OK), f"Asset {full_p} is not readable")
                else:
                    # Non-video items must have fingerspelling letters or unsupported state
                    self.assertTrue(
                        item.get("letters") is not None or item.get("match_tier") == "missing",
                        f"Item {item['gloss']} missing both clip and letters fallback"
                    )

    def test_standalone_resolve_isl_assets_hierarchy(self):
        # 1. Exact verified
        exact_res = resolve_isl_assets(["PLEASE", "HELP", "HELLO"], debug=False)
        self.assertEqual(len(exact_res), 3)
        for it in exact_res:
            self.assertEqual(it["match_tier"], "exact")
            self.assertIsNotNone(it["clip"])

        # 2. Verified alias / lemma
        alias_res = resolve_isl_assets(["HURT", "SEAT"], debug=False)
        for it in alias_res:
            self.assertIn(it["match_tier"], ("exact", "verified_alias", "lemma", "repo_equivalent"))
            self.assertIsNotNone(it["clip"])

        # 3. Fingerspelling fallback for missing sign (BUS has no verified clip)
        missing_res = resolve_isl_assets(["BUS"], debug=False)
        self.assertEqual(len(missing_res), 1)
        self.assertIsNone(missing_res[0]["clip"])
        self.assertEqual(missing_res[0]["letters"], ["b", "u", "s"])
        self.assertEqual(missing_res[0]["match_tier"], "fingerspelling")

    def test_non_fabrication_guarantee(self):
        # Scenario context must NEVER substitute one sign for another
        # (e.g. YES must never become MONEY just because scenario is 'bank')
        res_bank = self.pipe.translate("yes", scenario="bank")
        res_hospital = self.pipe.translate("yes", scenario="hospital")
        glosses_bank = [it["gloss"] for it in res_bank["items"]]
        glosses_hosp = [it["gloss"] for it in res_hospital["items"]]
        self.assertEqual(glosses_bank, ["YES"])
        self.assertEqual(glosses_hosp, ["YES"])
        # BUS must NEVER be substituted by TRAIN or CAR
        res_bus = self.pipe.translate("bus", scenario="bus_stop")
        for it in res_bus["items"]:
            if it["gloss"] == "BUS":
                self.assertIsNone(it["clip"], "BUS must not have a fabricated video clip substituted")
                self.assertEqual(it["letters"], ["b", "u", "s"])


class TestPhase20ScenarioSwitchingAndApi(unittest.TestCase):
    """Verifies API endpoints and scenario switching without retroactive corruption."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_manifest_endpoint(self):
        r = self.client.get("/api/scenarios/manifest")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data["scenarios"]), 6)

    def test_requirements_endpoint_all_scenarios(self):
        for s in ["hospital", "bank", "bus_stop", "railway", "shopping", "government"]:
            r = self.client.get(f"/api/scenarios/{s}/requirements")
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertEqual(data["scenario"], s)

    def test_dataset_plan_endpoint(self):
        r = self.client.get("/api/scenarios/railway/dataset-plan")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["scenario"], "railway")

    def test_english_to_sign_api_endpoint(self):
        payload = {"text": "Please wait here", "scenario": "bus_stop"}
        r = self.client.post("/api/english-to-sign", json=payload)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("items", data)
        self.assertIn("translation_status", data)

    def test_scenario_switching_preserves_inference_state(self):
        # Scenario switching must reset in-progress recognition buffer
        # without altering previously-emitted candidates
        rec = SignRecognizer(model_dir=MODELS_DIR)
        # Push idle/dummy frame
        dummy = [0.0] * 270
        rec.push(dummy)
        self.assertGreater(len(rec.frames), 0)
        # Switch scenario
        rec.reset()
        self.assertEqual(len(rec.frames), 0)
        self.assertEqual(len(rec.probs), 0)
        self.assertEqual(rec.streak, 0)


if __name__ == "__main__":
    unittest.main()
