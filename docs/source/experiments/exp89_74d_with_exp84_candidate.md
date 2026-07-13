# Experiment 89: Exp74d With Exp84 Candidate

Date: 2026-07-09 KST

## Purpose

Test whether Exp84 can be fully included as an additional candidate source inside the Exp74d operating selector, without using past family-performance labels as a selector prior.

The practical question was:

- Keep Exp74d/ROCKET as the operating base.
- Add Exp84 alerts as another candidate set.
- Let guarded agreement and no-alert repair decide when Exp84 is allowed to influence final alerts.

## Inputs

- Base operational result: `experiment_74d_large_rank_review_tier_split_results.csv`
- Exp84 index diagnostic result: `experiment_87_exp84_index_diagnostics_results.csv`
- Candidate sources:
  - `rocket_exp40`
  - `exp55_best`
  - `exp56_best`
  - `exp84_family_guard_v1`
  - `exp84_count_cap_3pct`

## Tested Selectors

- `baseline_74d_primary`
- `exp84_confidence_boost_only`
- `exp84_four_model_cap3_rocket_guard`
- `exp84_four_model_fg_rocket_guard`
- `exp84_four_model_fg_three_of_four`
- `exp84_review_tier_limited`
- `exp84_top1_noalert_repair`
- `exp84_four_model_fg_plus_noalert_repair`

## Result Summary

Best selector:

`exp84_four_model_fg_plus_noalert_repair`

| Metric | Value |
| --- | ---: |
| Mean F1 | 0.671613 |
| Median F1 | 1.000000 |
| P25 F1 | 0.181818 |
| Zero-F1 count | 271 |
| F1 >= 0.5 count | 776 |
| Mean TP | 1.747538 |
| Mean FP | 0.552372 |
| Mean FN | 1.760072 |
| Exp84 used datasets | 292 |
| Mean Exp84-only promoted count | 0.011638 |

Comparison with Exp74d baseline:

| Experiment | Mean F1 | Zero-F1 | Mean FP | Mean TP |
| --- | ---: | ---: | ---: | ---: |
| Exp74d baseline | 0.667296 | 272 | 0.541629 | 1.682184 |
| Exp89 best | 0.671613 | 271 | 0.552372 | 1.747538 |

Dataset-level movement versus Exp74d:

- Improved: 34 datasets
- Worsened: 6 datasets
- Unchanged: 1077 datasets
- Total TP gain: +73
- Total FP increase: +12
- Total FN reduction: -73

Largest improvements:

- `Phoneme_normal_1`: +1.000000 F1
- `UWaveGestureLibraryZ_normal_7`: +0.300000 F1
- `Crop_normal_17`: +0.243056 F1
- `Crop_normal_8`: +0.220690 F1
- `UWaveGestureLibraryY_normal_1`: +0.211765 F1

Largest regressions:

- `MelbournePedestrian_normal_1`: -0.050000 F1
- `Crop_normal_13`: -0.021849 F1
- `MelbournePedestrian_normal_3`: -0.012821 F1
- `Crop_normal_5`: -0.010870 F1
- `UWaveGestureLibraryX_normal_6`: -0.005848 F1

## Interpretation

Exp84 is useful when it is treated as a guarded additional candidate, not as a broad replacement for the operating baseline.

The best Exp89 result improves mean F1 by recovering additional true positives. The FP cost is small but real. This matters operationally because the project goal is not only to raise F1, but also to avoid excessive false alarms in a normal-heavy, label-scarce workflow.

`exp84_confidence_boost_only` produced the same alert set as Exp74d, which confirms that simply annotating confidence is not enough to improve F1. Actual candidate inclusion is needed.

The strict `three_of_four` route did not improve over baseline. It behaves like a confidence rule rather than a recovery rule: it is too conservative to recover missed anomalies.

## Decision

Use Exp89 as evidence for a cautious next selector direction:

- Base model remains Exp74d/ROCKET.
- Exp84 can be included as a guarded fourth candidate source.
- Exp84 should be especially useful for no-alert repair and two-of-four agreement.
- Avoid unguarded Exp84 promotion because FP still increases.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_89_74d_with_exp84_candidate_results.csv`
- `/Users/minho/Documents/Dataset/experiment_89_74d_with_exp84_candidate_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_89_74d_with_exp84_candidate_stdout.log`
