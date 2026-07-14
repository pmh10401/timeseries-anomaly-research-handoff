from __future__ import annotations

import ast
import inspect
import json
import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import virtual_run_policy_core as core


class VirtualRunPolicyCoreTests(unittest.TestCase):
    def test_conformal_p_value_uses_only_train_reference_scores(self):
        train_scores = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
        self.assertAlmostEqual(core.conformal_p_value(train_scores, 0.35), 2 / 5)
        self.assertAlmostEqual(core.conformal_p_value(train_scores, 0.50), 1 / 5)

    def test_conformal_prefix_is_unchanged_when_an_extra_test_run_is_appended(self):
        train_scores = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
        original = core.conformal_p_values(train_scores, np.asarray([0.15, 0.35]))
        appended = core.conformal_p_values(train_scores, np.asarray([0.15, 0.35, 9.0]))
        np.testing.assert_allclose(original, appended[:2])

    def test_fold_plan_is_deterministic_and_uses_leave_one_out_below_twenty(self):
        first = core.fold_plan(7, seed=20260717)
        second = core.fold_plan(7, seed=20260717)
        self.assertEqual(len(first), 7)
        for (fit_a, query_a), (fit_b, query_b) in zip(first, second):
            np.testing.assert_array_equal(fit_a, fit_b)
            np.testing.assert_array_equal(query_a, query_b)
            self.assertEqual(len(query_a), 1)
            self.assertEqual(len(fit_a), 6)

    def test_two_source_agreement_requires_two_independent_sources(self):
        p_values = {
            "source_a": np.asarray([0.01, 0.20, 0.01]),
            "source_b": np.asarray([0.02, 0.01, 0.20]),
            "source_c": np.asarray([0.20, 0.20, 0.01]),
        }
        selected, counts = core.agreement_indices(p_values, alpha=0.05, required_sources=2)
        self.assertEqual(selected, {0, 2})
        self.assertEqual(counts.tolist(), [2, 1, 2])

    def test_recomputed_block_b_and_block_c_evidence_create_disjoint_lanes(self):
        lanes = core.route_lanes_from_recomputed_evidence(
            candidates={1, 2, 3, 4},
            block_b_indices={1, 4},
            block_c_indices={2, 4},
            test_size=6,
        )
        self.assertEqual(lanes["hard"], {1, 4})
        self.assertEqual(lanes["priority_review"], {2})
        self.assertEqual(lanes["standard_review"], {3})
        self.assertEqual(lanes["no_alert"], {0, 5})
        self.assertFalse(lanes["hard"] & lanes["standard_review"])
        self.assertFalse(lanes["hard"] & lanes["priority_review"])
        self.assertFalse(lanes["standard_review"] & lanes["priority_review"])

    def test_low_train_policy_abstains_instead_of_claiming_review_candidates(self):
        status = core.calibration_status(n_train=4)
        self.assertEqual(status["status"], "insufficient_calibration_n_train_lt5")
        self.assertTrue(status["abstain"])
        self.assertFalse(status["review_only"])

    def test_resolution_blocker_uses_two_source_agreement_not_any_single_source(self):
        status = core.resolution_status(
            minimum_p_by_source={"a": 0.01, "b": 0.02, "c": 0.50, "d": 0.50},
            alpha=0.02,
            required_sources=2,
        )
        self.assertEqual(status["eligible_source_count"], 2)
        self.assertFalse(status["blocks_required_agreement"])

        blocked = core.resolution_status(
            minimum_p_by_source={"a": 0.01, "b": 0.50, "c": 0.50},
            alpha=0.02,
            required_sources=2,
        )
        self.assertEqual(blocked["eligible_source_count"], 1)
        self.assertTrue(blocked["blocks_required_agreement"])

    def test_b2_manifest_validation_rejects_wrong_config_seed_or_coverage(self):
        rows = [
            {
                "dataset_name": "d1",
                "threshold_method": "count_cap_2pct",
                "config_name": core.EXPECTED_B2_CONFIG_NAME,
                "random_state": str(core.SEED),
                "source_uses_family_name": "0",
                "source_uses_test_length": "0",
            },
            {
                "dataset_name": "d2",
                "threshold_method": "count_cap_2pct",
                "config_name": core.EXPECTED_B2_CONFIG_NAME,
                "random_state": str(core.SEED),
                "source_uses_family_name": "0",
                "source_uses_test_length": "0",
            },
        ]
        validated = core.validate_b2_manifest(rows, expected_datasets={"d1", "d2"})
        self.assertEqual(set(validated), {"d1", "d2"})

        broken = [dict(rows[0], random_state="1"), rows[1]]
        with self.assertRaises(ValueError):
            core.validate_b2_manifest(broken, expected_datasets={"d1", "d2"})

    def test_prediction_file_hash_must_match_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction = Path(tmp) / "predictions.jsonl"
            prediction.write_text(json.dumps({"dataset_name": "d1"}) + "\n", encoding="utf-8")
            digest = core.sha256_file(prediction)
            core.verify_frozen_prediction(prediction, digest)
            prediction.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                core.verify_frozen_prediction(prediction, digest)

    def test_alpha_grid_is_pre_registered_and_not_selected(self):
        self.assertEqual(core.ALPHAS, (0.005, 0.01, 0.02, 0.05))


class ExperimentStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).with_name(
            "run_experiment_151_152_virtual_run_conformal_policy_v2.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_database_path_is_not_changed(self):
        self.assertIn('DATA_DIR = Path("/Users/minho/Documents/Dataset")', self.source)
        self.assertIn('DB_PATH = DATA_DIR / "univariate_ts.db"', self.source)

    def test_prediction_function_does_not_accept_y_test(self):
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "predict_dataset"
        )
        argument_names = [arg.arg for arg in function.args.args]
        self.assertNotIn("y_test", argument_names)

    def test_old_exp133_exp135_index_maps_are_not_used_for_new_lanes(self):
        self.assertNotIn("high_confidence_indices", self.source)
        self.assertNotIn("review_candidate_indices", self.source)
        self.assertNotIn("load_routing_maps", self.source)

    def test_prediction_and_evaluation_are_separate_phases(self):
        function_names = {
            node.name for node in self.tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("run_prediction_phase", function_names)
        self.assertIn("run_evaluation_phase", function_names)
        self.assertIn("load_test_labels", function_names)

    def test_new_experiment_ids_preserve_exp149_exp150_outputs(self):
        self.assertIn("experiment_151_virtual_run_conformal_policy_v2", self.source)
        self.assertIn("experiment_152_virtual_run_conformal_policy_v2", self.source)
        self.assertNotIn("experiment_149_virtual_run_conformal_policy\"", self.source)
        self.assertNotIn("experiment_150_virtual_run_conformal_policy\"", self.source)


if __name__ == "__main__":
    unittest.main()
