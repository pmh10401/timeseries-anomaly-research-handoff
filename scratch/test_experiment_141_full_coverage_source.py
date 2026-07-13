import unittest

import numpy as np

import run_experiment_141_family_neutral_full_coverage as target


class FullCoverageSourceTests(unittest.TestCase):
    def test_source_row_uses_train_scores_for_count_cap_selection(self):
        row = target.source_row_from_scores(
            dataset_name="demo",
            family="demo_family",
            train_scores=np.asarray([0.1, 0.2, 0.3, 0.4]),
            test_scores=np.asarray([0.2, 0.5, 0.1]),
            rate=0.25,
            method="count_cap_2pct",
        )

        self.assertEqual(row["threshold_method"], "count_cap_2pct")
        self.assertEqual(row["selected_indices"], "1")
        self.assertEqual(row["source_uses_family_name"], 0)
        self.assertEqual(row["source_uses_test_length"], 0)

    def test_candidate_diff_distinguishes_exact_and_nearby_indices(self):
        rows = target.candidate_diff_rows(
            [
                {
                    "dataset_name": "demo",
                    "family": "demo_family",
                    "baseline_hard_indices": "10",
                    "hard_indices": "11",
                    "baseline_standard_indices": "",
                    "standard_review_indices": "",
                    "baseline_priority_indices": "",
                    "priority_review_indices": "",
                }
            ]
        )

        hard = next(row for row in rows if row["lane"] == "hard")
        self.assertEqual(hard["exact_match"], 0)
        self.assertEqual(hard["tolerance_1_matched"], 1)
        self.assertEqual(hard["added_indices"], "11")
        self.assertEqual(hard["removed_indices"], "10")


if __name__ == "__main__":
    unittest.main()
