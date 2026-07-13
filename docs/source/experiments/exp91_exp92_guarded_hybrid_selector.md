# Experiments 91-92: Rejected Tail-Replacement Diagnostic

Date: 2026-07-09 KST

## Purpose

Exp90 showed two useful but different directions:

- `noalert_top1_train_safe_repair` was operationally clean: no regressions versus Exp89.
- `candidate_union_rerank_repair` had higher F1 and lower zero-F1, but worsened 22 datasets by adding extra false positives.

Exp91 and Exp92 tested whether the aggressive candidate-union gain could be kept while removing most of the FP cost.

Important decision update:

These experiments are **rejected for operating use**. The improvement comes from tail-window replacement, which uses the current synthetic evaluation layout where injected anomalies often appear late in the test sequence. That is not a valid general anomaly-detection rule for deployment.

## Key Observation

The Exp90 candidate-union regressions were mostly sparse-alert cases:

- Exp89 already selected one correct tail index.
- Candidate-union added one earlier non-tail index.
- That extra index became a false positive.

The useful sparse repairs had the opposite shape:

- Exp89 selected one earlier wrong index.
- Candidate-union found a tail candidate.
- Replacing the earlier index with the tail index improved F1.

## Exp91 Result

Best Exp91 selector:

`noalert_plus_sparse_tail_replace`

| Metric | Value |
| --- | ---: |
| Mean F1 | 0.706528 |
| Zero-F1 count | 230 |
| Mean FP | 0.582811 |
| Mean TP | 1.784244 |
| Repair-used datasets | 87 |
| Replacement-used datasets | 12 |

## Exp92 Result

Best Exp92 selector:

`operational_noalert_top1_sparse_tail_replace`

This combines:

- Exp90 no-alert top-1 repair for no-alert cases.
- Exp91 sparse tail replacement only when Exp89 had one non-tail alert and a guarded tail candidate exists.

| Metric | Value |
| --- | ---: |
| Mean F1 | 0.707721 |
| Median F1 | 1.000000 |
| P25 F1 | 0.400000 |
| Zero-F1 count | 228 |
| F1 >= 0.5 count | 819 |
| Mean FP | 0.581021 |
| Mean TP | 1.786034 |
| Mean FN | 1.721576 |
| Tail replacement used datasets | 12 |

## Comparison

| Candidate | Mean F1 | Zero-F1 | Mean FP | Operational note |
| --- | ---: | ---: | ---: | --- |
| Exp89 best | 0.671613 | 271 | 0.552372 | Previous base |
| Exp90 no-alert top1 | 0.696978 | 240 | 0.591764 | Clean no-alert repair |
| Exp90 candidate-union | 0.697217 | 230 | 0.627574 | Higher F1 but FP-heavy |
| Exp91 sparse tail replace | 0.706528 | 230 | 0.582811 | Better guarded union |
| Exp92 hybrid | 0.707721 | 228 | 0.581021 | Current operating default candidate |

Exp92 movement:

- Versus Exp89: improved 43 datasets, worsened 0, fixed 43 zero-F1 datasets.
- Versus Exp90 no-alert top1: improved 12 datasets, worsened 0, fixed 12 additional zero-F1 datasets.

## Decision

Rejected for operating use.

The previous interpretation promoted Exp92 as the current operating default. That decision is withdrawn.

Current operating-default candidate:

`experiment_90_zero_f1_repair_selector / noalert_top1_train_safe_repair`

Reason:

- It improves zero-F1 without using tail-position assumptions.
- It fixed 31 zero-F1 datasets versus Exp89 with no dataset-level F1 regressions.
- Its logic is based on no-alert repair and train-safe top ranking, not on where anomalies were placed in the evaluation sequence.

## Rejected Candidate

Rejected candidate:

`experiment_92_operational_hybrid_selector / operational_noalert_top1_sparse_tail_replace`

Why rejected:

- The rule replaces an earlier candidate with a late-sequence candidate.
- This matches the synthetic test construction too closely.
- In a real operating stream, an anomaly can happen early, middle, or late.
- Therefore the improvement is likely evaluation-layout leakage, not robust detection capability.

## Watch Point

Tail replacement works because the current evaluation construction often places injected anomalies late in the test sequence. This should be treated as a negative control / leakage warning, not as an operating strategy.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_91_guarded_candidate_union_repair_results.csv`
- `/Users/minho/Documents/Dataset/experiment_91_guarded_candidate_union_repair_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_92_operational_hybrid_selector_results.csv`
- `/Users/minho/Documents/Dataset/experiment_92_operational_hybrid_selector_summary.csv`
