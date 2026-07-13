import unittest

import run_exp137_policy_validation_evidence as target


class PolicyValidationEvidenceTests(unittest.TestCase):
    def test_canonical_hash_is_independent_of_field_order(self):
        first = {"dataset": "demo", "hard": "1 3", "standard": "2"}
        second = {"standard": "2", "hard": "1 3", "dataset": "demo"}

        self.assertEqual(target.canonical_row_hash(first), target.canonical_row_hash(second))

    def test_source_coverage_marks_missing_sources_without_empty_candidate_fallback(self):
        row = target.source_coverage_row(
            "demo",
            "DemoFamily",
            {"config_name": target.EXP84_CONFIG, "threshold_method": "family_guard_v1"},
            {"config_name": target.EXP84_CONFIG, "threshold_method": "count_cap_2pct"},
            None,
            eligible=True,
        )

        self.assertEqual(row["has_family_guard_v1"], 1)
        self.assertEqual(row["has_count_cap_2pct"], 1)
        self.assertEqual(row["has_count_cap_3pct"], 0)
        self.assertEqual(row["missing_reason"], "missing_count_cap_3pct")
        self.assertEqual(row["score_vector_hash_family_guard"], "not_stored_in_csv")

    def test_tolerance_match_counts_nearby_indices_once(self):
        result = target.tolerance_match({10, 30}, {11, 31, 50}, tolerance=1)

        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["left_only"], 0)
        self.assertEqual(result["right_only"], 1)


if __name__ == "__main__":
    unittest.main()
