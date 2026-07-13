# Experiment 141 B2 Full-Coverage Family-Neutral Results

Status: retrospective counterfactual ablation. This is not prospective validation and not end-to-end strict TRAIN-only validation.

## Fixed policy

- Exp84 feature/score configuration: `aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3`, seed `20260717`.
- Every baseline dataset receives a newly calculated Exp84 source with the universal `count_cap_2pct` threshold.
- Family and dataset names are not used to decide whether the Exp84 source is calculated.
- Existing TEST-length budget behavior remains unchanged in B2; C/D1 are separately blocked by the unresolved operating budget contract.
- Priority review remains review-only. Any combined metric is human-assisted diagnostic-only.

## Aggregate metrics

| Metric | Value |
|---|---:|
| datasets | 1117 |
| hard_alerts | 2085 |
| hard_tp | 1759 |
| hard_fp | 326 |
| hard_precision | 0.8436450839328538 |
| mean_hard_f1 | 0.6059914693348503 |
| standard_review_candidates | 647 |
| standard_review_tp | 303 |
| standard_review_fp | 344 |
| standard_review_precision | 0.46831530139103555 |
| priority_review_candidates | 7 |
| priority_review_tp | 6 |
| priority_review_fp | 1 |
| priority_review_precision | 0.8571428571428571 |
| hard_changed_datasets | 44 |
| standard_changed_datasets | 14 |
| priority_changed_datasets | 2 |

## B1 common-support comparison

```json
{
  "available": true,
  "common_support_datasets": 339,
  "exact_lane_match_datasets": 339,
  "mismatch_datasets": []
}
```

## Coverage status

- Source calculation errors: 0
- A complete 1,117-dataset result is valid only when errors are zero and `datasets` is 1,117.
