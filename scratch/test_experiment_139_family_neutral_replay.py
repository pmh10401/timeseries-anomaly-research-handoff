import unittest

import run_experiment_139_family_neutral_common_support as target


class FamilyNeutralReplayTests(unittest.TestCase):
    def test_replace_family_guard_with_cap2_keeps_other_sources(self):
        sources = {
            ("demo", "family_guard_v1"): {"threshold_method": "family_guard_v1", "selected_indices": "1 4"},
            ("demo", "count_cap_2pct"): {"threshold_method": "count_cap_2pct", "selected_indices": "1 6"},
            ("demo", "count_cap_3pct"): {"threshold_method": "count_cap_3pct", "selected_indices": "1 4 6"},
        }

        neutral = target.replace_family_guard_with_cap2(sources, "demo")

        self.assertEqual(neutral[("demo", "family_guard_v1")]["selected_indices"], "1 6")
        self.assertEqual(neutral[("demo", "family_guard_v1")]["threshold_method"], "count_cap_2pct")
        self.assertEqual(neutral[("demo", "count_cap_3pct")]["selected_indices"], "1 4 6")

    def test_build_tiers_has_fixed_lane_precedence(self):
        tiers = target.build_tiers({1, 3}, {2, 3}, {2, 4}, test_size=6)

        self.assertEqual(tiers["hard"], {1, 3})
        self.assertEqual(tiers["standard_review"], {2})
        self.assertEqual(tiers["priority_review"], {4})
        self.assertEqual(tiers["no_alert"], {0, 5})


if __name__ == "__main__":
    unittest.main()
