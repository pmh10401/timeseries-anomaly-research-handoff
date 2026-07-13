import unittest
import tempfile
from pathlib import Path

import numpy as np

import run_exp137_budget_policy_validation as target


class BudgetPolicyValidationTests(unittest.TestCase):
    def test_threshold_only_candidates_require_train_calibrated_source_agreement(self):
        bundles = {
            "rocket_exp40": {"indices": {1, 2, 3}},
            "exp55_best": {"indices": {2, 3, 4}},
            "exp56_best": {"indices": {3, 4, 5}},
        }
        source = {"indices": {1, 4}, "train_exceed_rate": 0.0}

        selected = target.threshold_only_candidates(bundles, source)

        self.assertEqual(selected, {1, 2, 3, 4})

    def test_fixed_budget_does_not_depend_on_test_length(self):
        scores = np.asarray([0.1, 0.9, 0.8, 0.7])
        selected = target.apply_fixed_budget({0, 1, 2, 3}, scores, fixed_k=2)

        self.assertEqual(selected, {1, 2})

    def test_conformal_budget_is_blocked_without_verified_run_grain(self):
        self.assertFalse(target.TRAIN_RUN_GRAIN_VERIFIED)
        self.assertEqual(target.conformal_budget_status(), "blocked_unverified_train_run_grain")

    def test_cached_review_reuses_precomputed_candidates_and_block_c(self):
        cache = {
            "train_count": 20,
            "review_candidates": {1, 2, 3},
            "b_test": np.asarray([0.0, 0.4, 0.9, 0.7]),
            "c_indices": {2, 3},
        }

        review, priority = target.review_and_priority_cached(
            cache,
            candidate_indices={1},
            high=set(),
            standard={1},
        )

        self.assertEqual(review, {2})
        self.assertEqual(priority, {2})

    def test_checkpoint_round_trip_restores_dataset_result(self):
        payload = {"dataset": "demo", "audit": {"x": 1}, "fixed": [{"k": 1}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.jsonl"
            target.append_checkpoint(path, payload)

            restored = target.read_checkpoints(path)

        self.assertEqual(restored["demo"], payload)

    def test_default_worker_count_is_seven(self):
        self.assertEqual(target.WORKERS, 7)


if __name__ == "__main__":
    unittest.main()
