# Experiment 94b: Corrected Non-Position Rank Consensus

Date: 2026-07-09 KST

## Purpose

Exp94 failed badly because hard-alert replacement was too aggressive. The review
also found a design weakness: existing hard-alert indices were not always scored
inside the same candidate pool as replacement candidates.

Exp94b corrects this by including the existing Exp93 hard alert in the candidate
scoring pool before deciding whether it is weak.

No labels, tail position, or family-performance priors are used.

## Baseline

Operating baseline:

`experiment_93_nonpos_candidate_reranker / nonpos_weak_alert_replace`

| Metric | Value |
| --- | ---: |
| Mean F1 | 0.697575 |
| Zero-F1 | 239 |
| Mean FP | 0.590868 |

## Result Summary

| Selector | Mean F1 | Zero-F1 | Mean FP | Changed datasets | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_exp93_operating` | 0.697575 | 239 | 0.590868 | 0 | Current operating baseline |
| `corrected_replace_or_add_strict` | 0.697575 | 239 | 0.590868 | 1 | No meaningful effect |
| `corrected_replace_strict` | 0.696083 | 241 | 0.592659 | 4 | Slight regression |
| `corrected_replace_balanced` | 0.694591 | 244 | 0.595345 | 7 | Regression |
| `corrected_replace_margin` | 0.694442 | 244 | 0.595345 | 7 | Regression |

## Dataset-Level Movement

Against Exp93:

| Selector | Improved | Worsened | Zero-F1 fixed | New zero-F1 |
| --- | ---: | ---: | ---: | ---: |
| `corrected_replace_strict` | 0 | 2 | 0 | 2 |
| `corrected_replace_balanced` | 0 | 5 | 0 | 5 |
| `corrected_replace_margin` | 0 | 5 | 0 | 5 |
| `corrected_replace_or_add_strict` | 0 | 0 | 0 | 0 |

## Interpretation

The correction fixed the large Exp94 failure mode:

- Exp94 `replace_v2_strict` worsened 247 datasets.
- Exp94b `corrected_replace_strict` worsened only 2 datasets.

However, it did not create any positive improvement:

- zero-F1 fixed: `0`
- improved datasets: `0`

This closes the selector-replacement path for now. The problem is not just the
Exp94 implementation bug. Even after correction, replacing existing hard alerts
does not improve the operating baseline.

## Decision

Keep operating default:

`experiment_93_nonpos_candidate_reranker / nonpos_weak_alert_replace`

Keep review workflow:

`experiment_96_review_tier_operational_workflow / review_lane_top1_strict`

Do not pursue hard-alert replacement further unless a new feature representation
provides a much stronger and independently validated score.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_94b_corrected_nonpos_rank_consensus_results.csv`
- `/Users/minho/Documents/Dataset/experiment_94b_corrected_nonpos_rank_consensus_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_94b_corrected_nonpos_rank_consensus_stdout.log`
