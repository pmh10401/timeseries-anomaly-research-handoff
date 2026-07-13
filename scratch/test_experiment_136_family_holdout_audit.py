import unittest

import run_experiment_136_family_holdout_review_audit as target


class FamilyHoldoutAuditTests(unittest.TestCase):
    def test_family_is_assigned_to_one_stable_fold(self):
        self.assertEqual(target.family_fold('Phoneme'), target.family_fold('Phoneme'))
        self.assertIn(target.family_fold('Phoneme'), range(target.FOLD_COUNT))
        self.assertNotEqual(target.family_fold('Phoneme'), target.family_fold('CricketZ'))

    def test_summary_reports_candidate_precision_without_policy_labels(self):
        rows = [
            {'config_name': 'policy', 'family_holdout_fold': 0, 'f1': 0.5, 'fp': 1, 'combined_f1': 0.6,
             'combined_zero_f1': 0, 'review_candidate_count': 1, 'review_tp': 1, 'review_fp': 0,
             'combined_fp': 1, 'uses_test_labels_for_policy': 0},
            {'config_name': 'policy', 'family_holdout_fold': 1, 'f1': 0.0, 'fp': 0, 'combined_f1': 0.0,
             'combined_zero_f1': 1, 'review_candidate_count': 1, 'review_tp': 0, 'review_fp': 1,
             'combined_fp': 1, 'uses_test_labels_for_policy': 0},
        ]
        summary = target.summarize(rows)[0]
        self.assertEqual(summary['num_datasets'], 2)
        self.assertEqual(summary['review_candidate_datasets'], 2)
        self.assertEqual(summary['review_total_tp'], 1)
        self.assertEqual(summary['review_total_fp'], 1)
        self.assertEqual(summary['review_alert_precision'], 0.5)
        self.assertEqual(summary['uses_test_labels_for_policy_rows'], 0)


if __name__ == '__main__':
    unittest.main()
