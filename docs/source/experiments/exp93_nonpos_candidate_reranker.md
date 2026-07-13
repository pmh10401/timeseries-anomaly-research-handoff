# Experiment 93: Non-Position Candidate Reranker

Date: 2026-07-09 KST

## Purpose

Exp90 deep dive showed that the remaining zero-F1 cases are wrong-alert cases,
not no-alert cases. In `233/240` remaining zero-F1 rows, at least one score
source ranked the true anomaly inside top-10.

Exp93 tests whether candidate selection can be improved without using test
position, tail-window rules, family-performance priors, or labels.

## Baseline

Operating baseline:

`experiment_90_zero_f1_repair_selector / noalert_top1_train_safe_repair`

Baseline metrics:

| Metric | Value |
| --- | ---: |
| Mean F1 | 0.696978 |
| Median F1 | 1.000000 |
| Zero-F1 count | 240 |
| Mean FP | 0.591764 |
| Mean TP | 1.775291 |

## Design

Candidate sources:

- ROCKET Exp40 top-ranked indices
- Exp55 best imaging-derived score top-ranked indices
- Exp56 best texture/imaging score top-ranked indices
- Exp84 feature-pruned official MultiROCKET/HYDRA top-ranked indices

The reranker uses only non-position signals:

- rank consensus across score sources
- best rank among score sources
- weighted rank score
- train-normal safety for Exp84 sources
- weak-alert replacement guard

It does not use:

- true labels
- tail index or test-window position
- prior family-performance gain table
- prior zero-F1 family policy

## Tested Selectors

| Selector | Description |
| --- | --- |
| `baseline_exp90_noalert_top1` | Exp90 operating default control. |
| `nonpos_weak_alert_replace` | Replace one weak sparse alert if another candidate has stronger non-position rank consensus. |
| `nonpos_replace_else_strict_add` | Replace weak alert, or otherwise add a very strong consensus candidate. |
| `nonpos_consensus_add_cap1` | Add one strong consensus candidate to sparse-alert cases. |
| `review_tier_nonpos_top_candidate` | Broad review-tier diagnostic, not a hard-alert candidate. |
| `reference_exp90_candidate_union` | Exp90 candidate-union research reference. |

## Result Summary

| Selector | Mean F1 | Zero-F1 | Mean FP | Mean TP | Rerank used |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nonpos_weak_alert_replace` | 0.697575 | 239 | 0.590868 | 1.776186 | 2 |
| `reference_exp90_candidate_union` | 0.697217 | 230 | 0.627574 | 1.785139 | 78 |
| `baseline_exp90_noalert_top1` | 0.696978 | 240 | 0.591764 | 1.775291 | 0 |
| `nonpos_replace_else_strict_add` | 0.683997 | 239 | 0.648165 | 1.776186 | 66 |
| `nonpos_consensus_add_cap1` | 0.602827 | 232 | 0.984781 | 1.785139 | 450 |
| `review_tier_nonpos_top_candidate` | 0.594846 | 227 | 1.181737 | 1.825425 | 715 |

## Dataset-Level Movement

`nonpos_weak_alert_replace` versus Exp90 operating baseline:

- Improved: `1`
- Worsened: `0`
- Unchanged: `1116`
- Zero-F1 fixed: `1`
- New zero-F1 regressions: `0`
- Total TP delta: `+1`
- Total FP delta: `-1`

Fixed dataset:

| Dataset | Family | New selected index | TP | FP | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `Phoneme_normal_28` | Phoneme | 98 | 1 | 0 | 0.666667 |

## Interpretation

The clean operating result is small but meaningful:

`nonpos_weak_alert_replace` improves Exp90 without adding regressions and without
using the invalid tail-position shortcut. It changes only two datasets, so it
is conservative enough for operating use.

The broad add/review variants confirm the earlier warning:

- adding top candidates can reduce zero-F1,
- but it creates many false positives,
- so top-k candidate expansion is better treated as a review tier, not as hard
  alert expansion.

## Decision

New operating-default candidate:

`experiment_93_nonpos_candidate_reranker / nonpos_weak_alert_replace`

This should replace Exp90 as the current operating default because it has:

- slightly higher mean F1,
- one fewer zero-F1 dataset,
- slightly lower mean FP,
- no per-dataset F1 regressions,
- no tail-position or prior-family leakage.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_93_nonpos_candidate_reranker_results.csv`
- `/Users/minho/Documents/Dataset/experiment_93_nonpos_candidate_reranker_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_93_nonpos_candidate_reranker_stdout.log`
