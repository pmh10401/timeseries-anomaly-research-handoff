import unittest

import run_experiment_138_policy_train_only_audit as target


class PolicyTrainOnlyAuditTests(unittest.TestCase):
    def test_rebuild_tiers_keeps_lanes_disjoint(self):
        row133 = {
            "high_confidence_indices": "1 3",
            "standard_confidence_indices": "2 3",
        }
        row135 = {"review_candidate_indices": "2 4"}

        tiers = target.rebuild_tiers(row133, row135, test_size=6)

        self.assertEqual(tiers["hard"], {1, 3})
        self.assertEqual(tiers["standard_review"], {2})
        self.assertEqual(tiers["priority_review"], {4})
        self.assertEqual(tiers["no_alert"], {0, 5})

    def test_family_neutral_source_uses_cap2_without_family_guard_metadata(self):
        fg = {
            "dataset_name": "demo",
            "threshold_method": "family_guard_v1",
            "selected_indices": "1 4 6",
            "top_score_indices": "1 4 6 7",
        }
        cap2 = {
            "dataset_name": "demo",
            "threshold_method": "count_cap_2pct",
            "selected_indices": "1 6",
            "top_score_indices": "1 6 4 7",
        }

        neutral = target.family_neutral_exp84_source(fg, cap2)

        self.assertEqual(neutral["selected_indices"], "1 6")
        self.assertEqual(neutral["top_score_indices"], "1 6 4 7")
        self.assertEqual(neutral["source_threshold_method"], "count_cap_2pct")
        self.assertEqual(neutral["policy_uses_family_name"], 0)
        self.assertEqual(neutral["policy_uses_test_length"], 0)

    def test_summary_separates_autonomous_and_assisted_metrics(self):
        rows = [
            {
                "hard_alert_count": 2,
                "hard_tp": 1,
                "hard_fp": 1,
                "hard_fn": 1,
                "hard_f1": 0.5,
                "standard_review_count": 1,
                "standard_review_tp": 1,
                "standard_review_fp": 0,
                "priority_review_count": 0,
                "priority_review_tp": 0,
                "priority_review_fp": 0,
                "combined_f1": 0.8,
            }
        ]

        summary = target.summarize_rows(rows)

        self.assertEqual(summary["hard_total_alerts"], 2)
        self.assertEqual(summary["hard_total_tp"], 1)
        self.assertEqual(summary["hard_total_fp"], 1)
        self.assertEqual(summary["mean_hard_f1"], 0.5)
        self.assertEqual(summary["mean_combined_f1"], 0.8)
        self.assertEqual(summary["mean_f1_scope"], "autonomous_hard_alert")
        self.assertEqual(summary["combined_metric_scope"], "human_assisted_diagnostic_only")


if __name__ == "__main__":
    unittest.main()
