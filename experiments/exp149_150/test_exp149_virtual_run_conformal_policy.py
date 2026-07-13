import unittest

import numpy as np

import run_experiment_149_150_virtual_run_conformal_policy as policy


class VirtualRunConformalPolicyTests(unittest.TestCase):
    def test_conformal_p_value_uses_train_oof_scores(self):
        train = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
        self.assertAlmostEqual(policy.conformal_p_value(train, 0.35), 2 / 5)
        self.assertAlmostEqual(policy.conformal_p_value(train, 0.5), 1 / 5)

    def test_fold_plan_is_deterministic_and_uses_leave_one_out_for_small_train(self):
        first = policy.fold_plan(4)
        second = policy.fold_plan(4)
        self.assertEqual(len(first), len(second))
        for (train_a, test_a), (train_b, test_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(len(train) == 3 and len(test) == 1 for train, test in first))

    def test_two_source_agreement(self):
        p_values = {
            "rocket_exp40": np.array([0.01, 0.2, 0.01]),
            "exp55_best": np.array([0.02, 0.01, 0.2]),
            "exp56_best": np.array([0.2, 0.2, 0.01]),
        }
        selected, counts = policy.agreement_indices(p_values, alpha=0.05)
        self.assertEqual(selected, {0, 2})
        self.assertEqual(counts.tolist(), [2, 1, 2])

    def test_lane_partition_is_disjoint_and_bounded(self):
        lanes = policy.route_lanes({1, 3}, {1}, {3}, 5)
        self.assertEqual(lanes["hard"], {1})
        self.assertEqual(lanes["standard_review"], set())
        self.assertEqual(lanes["priority_review"], {3})
        self.assertEqual(lanes["hard"] | lanes["standard_review"] | lanes["priority_review"], {1, 3})
        self.assertTrue(all(0 <= index < 5 for index in lanes["no_alert"]))


if __name__ == "__main__":
    unittest.main()
