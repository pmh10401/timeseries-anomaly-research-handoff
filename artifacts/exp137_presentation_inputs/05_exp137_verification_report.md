# Exp137 Result Verification

Generated: 2026-07-13T08:06:29+09:00

- Detail rows: 1117; duplicate dataset names: 0
- Integrity failures: 0
- Independent aggregation matched every comparable summary value: True
- Combined TP/FP/FN are independently recomputed from detail index sets and SQLite labels; the summary CSV does not aggregate these three totals.

## Checks

- num_datasets: 1117 (summary column=num_datasets, summary=1117, match=True)
- hard_alert_count: 2005 (summary column=hard_total_alerts, summary=2005, match=True)
- hard_tp: 1691 (summary column=hard_total_tp, summary=1691, match=True)
- hard_fp: 314 (summary column=hard_total_fp, summary=314, match=True)
- standard_review_count: 639 (summary column=standard_review_total_candidates, summary=639, match=True)
- standard_review_tp: 292 (summary column=standard_review_total_tp, summary=292, match=True)
- standard_review_fp: 347 (summary column=standard_review_total_fp, summary=347, match=True)
- priority_review_count: 9 (summary column=priority_review_total_candidates, summary=9, match=True)
- priority_review_tp: 8 (summary column=priority_review_total_tp, summary=8, match=True)
- priority_review_fp: 1 (summary column=priority_review_total_fp, summary=1, match=True)
- combined_tp: 1991 (summary column=none, summary=not_aggregated_in_summary, match=not_applicable)
- combined_fp: 662 (summary column=none, summary=not_aggregated_in_summary, match=not_applicable)
- combined_fn: 1927 (summary column=none, summary=not_aggregated_in_summary, match=not_applicable)
- hard_alert_precision: 0.8433915211970074 (summary column=hard_alert_precision, summary=0.8433915211970074, match=True)
- mean_hard_f1: 0.6007724315633254 (summary column=mean_hard_f1, summary=0.6007724315633254, match=True)
- mean_combined_f1: 0.7007761778434174 (summary column=mean_combined_f1, summary=0.7007761778434174, match=True)
- routing_uses_test_labels_rows: 0 (summary column=routing_uses_test_labels_rows, summary=0, match=True)
- routing_uses_test_position_rows: 0 (summary column=routing_uses_test_position_rows, summary=0, match=True)
- routing_uses_family_performance_rows: 0 (summary column=routing_uses_family_performance_rows, summary=0, match=True)

## Integrity findings

- None.

Every count was recomputed from `experiment_137_operational_triage_results.csv` index columns and SQLite `instances.label`; labels were used here only for post-hoc verification.