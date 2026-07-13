# Exp94/Exp95 Deep Analysis

Date: 2026-07-09 KST

## Baseline

Current operating baseline:

`experiment_93_nonpos_candidate_reranker / nonpos_weak_alert_replace`

| Metric | Value |
| --- | ---: |
| Mean F1 | 0.697575 |
| Median F1 | 1.000000 |
| Zero-F1 | 239 |
| Mean FP | 0.590868 |
| Mean TP | 1.776186 |

## Exp94: Non-Position Rank Consensus v2

### Purpose

Exp94 tried to reduce zero-F1 by replacing weak hard alerts using stronger
non-position rank consensus candidates.

It did not use tail position or labels.

### Results

| Selector | Mean F1 | Zero-F1 | Mean FP | Changed datasets | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_exp93_operating` | 0.697575 | 239 | 0.590868 | 0 | Exp93 unchanged |
| `replace_or_add_v2_strict` | 0.697575 | 239 | 0.590868 | 1 | No meaningful effect |
| `replace_v2_strict` | 0.478237 | 483 | 0.809311 | 263 | Failed |
| `replace_v2_balanced` | 0.476447 | 485 | 0.811101 | 266 | Failed |
| `replace_v2_margin` | 0.476447 | 485 | 0.811101 | 266 | Failed |

### Dataset-Level Movement

For `replace_v2_strict` versus Exp93:

- Improved: `3`
- Worsened: `247`
- Zero-F1 fixed: `3`
- New zero-F1 regressions: `247`
- TP delta: `-244`
- FP delta: `+244`

Improved rows:

| Dataset | Family | Selected | F1 |
| --- | --- | ---: | ---: |
| `InlineSkate_normal_4` | InlineSkate | 98 | 0.666667 |
| `ShapeletSim_normal_0` | ShapeletSim | 99 | 0.666667 |
| `ShapeletSim_normal_1` | ShapeletSim | 98 | 0.666667 |

Main regression families:

| Family | Worsened rows |
| --- | ---: |
| ShapesAll | 42 |
| NonInvasiveFetalECGThorax1 | 33 |
| NonInvasiveFetalECGThorax2 | 28 |
| Adiac | 24 |
| SwedishLeaf | 13 |
| FiftyWords | 11 |

### Root Cause

Exp94 is not a valid operating improvement.

Two causes were found:

1. **Conceptual risk**
   - Strong rank consensus does not always mean anomaly.
   - Several models can agree on the same normal-but-salient position.
   - Replacing a hard alert can discard a true positive and create a false
     positive.

2. **Design/implementation weakness**
   - Exp94 scored the candidate pool, but did not always score the existing hard
     alert itself.
   - Existing hard alerts outside the candidate pool could be judged weak by
     default.
   - This made the replacement logic too aggressive.

Even if this is corrected, the result shows a key operating lesson:

> Hard-alert replacement is much riskier than hard-alert preservation plus
> review-tier annotation.

## Exp95: Top-k Review Tier

### Purpose

Exp95 keeps the Exp93 hard alert unchanged and adds separate review candidates.

These review candidates are not meant to be counted as hard alerts in operation.
They are a second-tier queue for human or downstream review.

### Results

| Selector | Hard Mean F1 | Hard Zero-F1 | Review candidates | Review hit datasets | Combined Zero-F1 | Review precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `review_top1_strict` | 0.697575 | 239 | 579 | 23 | 230 | 0.0397 |
| `review_top2_balanced` | 0.697575 | 239 | 1441 | 88 | 223 | 0.0715 |
| `review_top3_broad` | 0.697575 | 239 | 2174 | 115 | 219 | 0.0658 |
| `review_top5_diagnostic` | 0.697575 | 239 | 3972 | 200 | 203 | 0.0763 |

Hard-alert metrics stay exactly equal to Exp93 because Exp95 does not modify
hard alerts.

### Zero-F1 Rescue

If review candidates are considered as "found during review":

| Selector | Zero-F1 rescued |
| --- | ---: |
| `review_top1_strict` | 9 |
| `review_top2_balanced` | 16 |
| `review_top3_broad` | 20 |
| `review_top5_diagnostic` | 36 |

Top rescued families:

- `review_top1_strict`: Haptics, InlineSkate, ShapeletSim, AllGestureWiimoteX/Y,
  WormsTwoClass
- `review_top5_diagnostic`: Phoneme, WordSynonyms, InlineSkate,
  AllGestureWiimoteZ, Haptics, ShapeletSim

### Interpretation

Exp95 shows that extra candidate information is useful, but not precise enough
to become hard alerts.

For example:

- `review_top1_strict` finds 9 zero-F1 misses with only one review candidate.
- But even strict review precision is only about `3.97%`.
- Broader review finds more misses but adds many more false review candidates.

Therefore:

> Review candidates are valuable as a triage layer, not as automatic alerts.

## Combined Conclusion

Exp94 and Exp95 tell a consistent story:

1. Replacing existing hard alerts is dangerous.
2. Adding broad candidates as hard alerts is also dangerous.
3. Keeping Exp93 hard alerts fixed and adding a separate review tier is the
   safer path.

## Recommendation

Keep operating default:

`experiment_93_nonpos_candidate_reranker / nonpos_weak_alert_replace`

Use Exp95 as a review-tier design candidate, not as a hard-alert model.

Most practical next step:

`Exp96`: calibrate review tier into an operational workflow:

- hard alert: Exp93 only
- review candidate: Exp95 top-1 or top-2
- display review candidates separately in dashboard
- measure review load per 100 datasets
- prioritize candidates by score support and family difficulty

Do not promote Exp94 replacement rules without a corrected Exp94b rerun and a
strict no-regression guard.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_94_nonpos_rank_consensus_v2_results.csv`
- `/Users/minho/Documents/Dataset/experiment_94_nonpos_rank_consensus_v2_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_95_topk_review_tier_results.csv`
- `/Users/minho/Documents/Dataset/experiment_95_topk_review_tier_summary.csv`
