# Experiment 90: Zero-F1 Repair Selector

Date: 2026-07-09 KST

## Purpose

Reduce Exp89 zero-F1 cases without using test labels or prior family-performance tables.

The experiment focuses on the failure pattern found after Exp89:

- Many zero-F1 datasets had only one anomaly.
- Some had no final alert even though model score ranking was useful.
- Some had an alert, but the chosen index was wrong.

## Baseline

Operating baseline:

`experiment_89_74d_with_exp84_candidate`

Baseline selector:

`exp84_four_model_fg_plus_noalert_repair`

Baseline metrics:

| Metric | Value |
| --- | ---: |
| Mean F1 | 0.671613 |
| Zero-F1 count | 271 |
| Mean FP | 0.552372 |
| Mean TP | 1.747538 |

## Tested Selectors

- `baseline_exp89_best`
  - Exp89 best selector, unchanged.
- `noalert_top1_train_safe_repair`
  - If Exp89 emits no alert, add one train-safe top-ranked alert.
  - Disabled for tiny-train datasets.
- `candidate_union_rerank_repair`
  - If Exp89 emits no alert or only one sparse alert, form a small candidate pool from ROCKET, Exp55, Exp56, and Exp84 top candidates.
  - Re-rank with ROCKET score and add a limited number of candidates.
  - Disabled for tiny-train datasets.
- `large_case_two_model_extension`
  - For large-data cases, add limited two-model-agreement candidates.
- `tiny_train_guarded_noalert_repair`
  - Same as no-alert top-1 repair with explicit tiny-train guard.
- `reference_74d_baseline`
  - Exp74d reference only.

## Result Summary

| Selector | Mean F1 | Zero-F1 | Mean FP | Mean TP | Repair Used |
| --- | ---: | ---: | ---: | ---: | ---: |
| `candidate_union_rerank_repair` | 0.697217 | 230 | 0.627574 | 1.785139 | 126 |
| `noalert_top1_train_safe_repair` | 0.696978 | 240 | 0.591764 | 1.775291 | 75 |
| `tiny_train_guarded_noalert_repair` | 0.696978 | 240 | 0.591764 | 1.775291 | 75 |
| `baseline_exp89_best` | 0.671613 | 271 | 0.552372 | 1.747538 | 0 |
| `large_case_two_model_extension` | 0.670335 | 271 | 0.580125 | 1.753805 | 19 |
| `reference_74d_baseline` | 0.667296 | 272 | 0.541629 | 1.682184 | 0 |

## Dataset-Level Movement

`candidate_union_rerank_repair` versus Exp89:

- Improved: 42 datasets
- Worsened: 22 datasets
- Unchanged: 1053 datasets
- Zero-F1 fixed: 41
- New zero-F1 regressions: 0
- Total TP gain: +42
- Total FP increase: +84

`noalert_top1_train_safe_repair` versus Exp89:

- Improved: 31 datasets
- Worsened: 0 datasets
- Zero-F1 fixed: 31
- Total repair-used datasets: 75
- Mean FP increase is smaller than candidate-union.

## Interpretation

`candidate_union_rerank_repair` is the best research score. It cuts zero-F1 from `271` to `230` and raises mean F1 to `0.697217`.

However, it also worsens 22 datasets because it sometimes adds an extra candidate to datasets that already had one alert. This improves some sparse-alert cases, but creates FP in others.

`noalert_top1_train_safe_repair` is cleaner for operations. It only changes datasets where Exp89 emitted no alert. It fixed 31 zero-F1 datasets and produced no F1 regressions compared with Exp89.

## Decision

Recommended operating default candidate:

`noalert_top1_train_safe_repair`

Recommended research candidate:

`candidate_union_rerank_repair`

Next improvement should keep the no-alert repair rule and add a second guard for candidate-union reranking so it only activates on sparse-alert cases with stronger confidence.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_90_zero_f1_repair_selector_results.csv`
- `/Users/minho/Documents/Dataset/experiment_90_zero_f1_repair_selector_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_90_zero_f1_repair_selector_stdout.log`
