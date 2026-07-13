# Exp90 Zero-F1 Deep Dive

Date: 2026-07-09 KST

## Scope

This review analyzes the `240` zero-F1 datasets that remain under the current
operating-default candidate:

`experiment_90_zero_f1_repair_selector / noalert_top1_train_safe_repair`

The goal is to identify why these datasets still have F1 `0.0`, without using
tail-position rules or any other evaluation-layout shortcut as an operating
strategy.

## Main Finding

The remaining Exp90 zero-F1 cases are not mostly "no alert" failures.

They are mostly wrong-alert failures:

- Zero-F1 rows: `240`
- Families involved: `85`
- No-alert failures: `0`
- Wrong-alert failures: `240`

In plain terms, Exp90 usually raises an alert, but the alert points at the
wrong test index.

## Alert Count Shape

Most failures are extremely sparse:

| Selected alert count | Datasets |
| ---: | ---: |
| 1 | 216 |
| 2 | 7 |
| 3 | 5 |
| 4 | 7 |
| 5 | 1 |
| 7 | 3 |
| 8 | 1 |

Most datasets also have only one true anomaly:

| True anomaly count | Datasets |
| --- | ---: |
| 1 | 193 |
| 2 | 24 |
| 3-5 | 12 |
| 6-10 | 4 |
| >10 | 7 |

This means a single wrong alert is enough to make F1 exactly `0.0`.

## Candidate Ranking Diagnostic

For each remaining zero-F1 dataset, the top-ranked anomaly position was checked
inside three candidate score sources:

- `rocket_exp40`
- `exp55_best`
- `exp56_best`

Capture counts:

| Signal | Datasets captured |
| --- | ---: |
| At least one model has anomaly at top-1 | 99 |
| At least one model has anomaly in top-3 | 201 |
| At least one model has anomaly in top-5 | 219 |
| At least one model has anomaly in top-10 | 233 |

This is important. It means the score representations often contain useful
information, but the operating selector is too narrow to pick the right
candidate.

## Root Cause Buckets

| Root cause bucket | Count | Interpretation |
| --- | ---: | --- |
| `A_selector_missed_model_top1` | 99 | A model had the true anomaly as top-1, but Exp90 selected another index. |
| `B_selector_missed_top3` | 102 | A model had the true anomaly in top-3, but not as the selected alert. |
| `C_selector_threshold_top10` | 32 | The anomaly was recoverable in top-10, but the threshold/selection was too strict. |
| `D_moderate_rank_not_alerted` | 7 | The anomaly was only moderately ranked. |
| `E_representation_score_failure` | 0 | No case where all score sources completely failed under this diagnostic. |

The biggest remaining problem is therefore not feature extraction alone. It is
the final selection rule: which candidate index becomes an alert.

## Top Families

| Family | Zero-F1 rows | Mean oracle F1 | Mean AUCPR | Mean train normals | Single-anomaly rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| PigAirwayPressure | 25 | 0.613 | 0.223 | 4.0 | 1.000 |
| Phoneme | 23 | 0.503 | 0.207 | 36.9 | 0.870 |
| NonInvasiveFetalECGThorax2 | 10 | 0.617 | 0.225 | 70.8 | 1.000 |
| GestureMidAirD3 | 9 | 0.471 | 0.162 | 10.0 | 1.000 |
| WordSynonyms | 7 | 0.619 | 0.226 | 19.4 | 1.000 |
| CricketZ | 7 | 0.430 | 0.149 | 52.0 | 1.000 |
| PigCVP | 7 | 0.667 | 0.250 | 4.0 | 1.000 |
| FiftyWords | 6 | 0.594 | 0.215 | 21.2 | 1.000 |
| NonInvasiveFetalECGThorax1 | 6 | 0.694 | 0.361 | 72.0 | 1.000 |

Two patterns stand out:

1. Several families have very few train-normal samples, especially the Pig
   families.
2. Most remaining failures are single-anomaly cases, so the evaluation is very
   sensitive to one wrong index.

## Train Size Buckets

| Train-normal count | Zero-F1 rows |
| --- | ---: |
| <=5 | 36 |
| 6-10 | 20 |
| 11-20 | 34 |
| 21-50 | 45 |
| 51-100 | 80 |
| 101-500 | 21 |
| >500 | 4 |

Only `56/240` rows are tiny-train cases with at most 10 train-normal samples.
Tiny train size matters, but it is not the only explanation.

## Important Non-Operating Diagnostic

In the current synthetic evaluation files, the true anomalies in these zero-F1
rows appear in the final test-window region, while Exp90 selected earlier
indices. This explains why Exp91/Exp92 tail replacement looked strong.

However, this is not valid as an operating rule. In a real equipment stream, an
anomaly does not promise to appear at the end of a batch. Tail-position
replacement is therefore excluded from operating use.

## Interpretation

Exp90 solved the safest first problem: no-alert cases. The remaining failures
are a harder second problem:

> The system raises an alert, but chooses the wrong candidate index.

Because `233/240` remaining failures have the true anomaly inside the top-10 of
at least one score source, the next improvement should focus on candidate
selection and confidence calibration, not on a broad model replacement.

## Recommended Next Experiments

1. **Non-position candidate reranker**
   - Build a small candidate pool from ROCKET, Exp55, Exp56, and Exp84.
   - Re-rank by score margin, rank consensus, and train-normal safety.
   - Do not use test index position.

2. **Top-k review tier**
   - Keep the operating alert conservative.
   - Add a separate "review candidate" list for top-3/top-5 cases.
   - This can reduce missed-review risk without increasing hard alerts.

3. **Tiny-train normal pooling**
   - For families such as PigAirwayPressure and PigCVP, train-normal count is
     too small to estimate a stable normal boundary.
   - Test recipe/family-level normal pooling or shrinkage calibration.

4. **Single-anomaly sensitivity report**
   - Report one-anomaly datasets separately from multi-anomaly datasets.
   - A single wrong alert creates F1 `0.0`, so this subgroup needs its own
     operating metric.

5. **Hard-family representation probes**
   - Phoneme, GestureMidAirD3, CricketZ, and EOGVerticalSignal still need better
     score separation.
   - Use this only after the non-position selector issue is addressed, because
     the current diagnostic shows many anomalies are already ranked near the top.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_90_zero_f1_rank_diagnostics.csv`
- `/Users/minho/Documents/Dataset/experiment_90_zero_f1_root_cause.csv`
