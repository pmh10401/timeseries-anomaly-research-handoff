import unittest

import numpy as np

import run_experiment_137_operational_triage as target


class OperationalTriageTests(unittest.TestCase):
    def test_route_tiers_applies_precedence_and_is_disjoint(self):
        tiers = target.route_tiers({1, 3}, {2, 3}, {2, 4}, test_size=6)
        self.assertEqual(tiers['hard'], {1, 3})
        self.assertEqual(tiers['standard_review'], {2})
        self.assertEqual(tiers['priority_review'], {4})
        self.assertFalse(tiers['hard'] & tiers['standard_review'])
        self.assertFalse(tiers['hard'] & tiers['priority_review'])
        self.assertFalse(tiers['standard_review'] & tiers['priority_review'])

    def test_route_tiers_rejects_out_of_bounds_index(self):
        with self.assertRaises(ValueError):
            target.route_tiers({5}, set(), set(), test_size=5)

    def test_labels_do_not_change_routing(self):
        first = target.route_tiers({1}, {2}, {3}, test_size=4)
        second = target.route_tiers({1}, {2}, {3}, test_size=4)
        self.assertEqual(first, second)

    def test_metrics_keep_hard_and_review_meaning_separate(self):
        y = np.asarray([0, 1, 0, 1], dtype=int)
        hard = target.tier_metrics(y, {1, 2})
        review = target.tier_metrics(y, {3})
        self.assertEqual((hard['tp'], hard['fp'], hard['fn']), (1, 1, 1))
        self.assertEqual((review['tp'], review['fp']), (1, 0))

    def test_summary_keeps_autonomous_and_review_totals_separate(self):
        rows = [
            {
                'hard_tp': 2,
                'hard_fp': 1,
                'hard_fn': 1,
                'hard_f1': 2 / 3,
                'standard_review_tp': 1,
                'standard_review_fp': 2,
                'priority_review_tp': 1,
                'priority_review_fp': 0,
                'combined_tp': 4,
                'combined_fp': 3,
                'combined_fn': 0,
                'combined_f1': 8 / 11,
                'routing_uses_test_labels': 0,
                'hard_alert_count': 3,
                'standard_review_count': 3,
                'priority_review_count': 1,
            }
        ]
        summary = target.summarize(rows)[0]
        self.assertEqual(summary['hard_total_tp'], 2)
        self.assertEqual(summary['hard_total_fp'], 1)
        self.assertEqual(summary['priority_review_total_tp'], 1)
        self.assertEqual(summary['priority_review_total_fp'], 0)
        self.assertEqual(summary['routing_uses_test_labels_rows'], 0)
        self.assertEqual(summary['mean_f1'], summary['mean_hard_f1'])
        self.assertEqual(summary['mean_f1_scope'], 'autonomous_hard_alert')


if __name__ == '__main__':
    unittest.main()
