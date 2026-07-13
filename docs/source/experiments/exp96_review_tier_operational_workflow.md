# Experiment 96: Review Tier Operational Workflow

Date: 2026-07-09 KST

## Purpose

Exp96 converts Exp95 review-tier results into operating workflow metrics.

The hard-alert model is not changed:

`experiment_93_nonpos_candidate_reranker / nonpos_weak_alert_replace`

Review candidates are evaluated as a separate review lane, not as automatic hard
alerts.

## Result Summary

| Policy | Hard Mean F1 | Hard Zero-F1 | Review candidates | Candidates / 100 datasets | Review hits | Zero-F1 rescued | Review precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hard_only_exp93` | 0.697575 | 239 | 0 | 0.0 | 0 | 0 | 0.0000 |
| `review_lane_top1_strict` | 0.697575 | 239 | 579 | 51.8 | 23 | 9 | 0.0397 |
| `review_lane_top2_balanced` | 0.697575 | 239 | 1441 | 129.0 | 88 | 16 | 0.0715 |
| `review_lane_top3_broad` | 0.697575 | 239 | 2174 | 194.6 | 115 | 20 | 0.0658 |

## Interpretation

Hard-alert performance remains equal to Exp93 for all policies because Exp96
does not change hard alerts.

If review candidates are incorrectly treated as hard alerts, combined F1 drops.
This is expected because review precision is low. Therefore review candidates
must be shown as a separate "needs review" lane, not as automatic anomaly
alarms.

## Operating Recommendation

Recommended default workflow:

1. **Hard alert**
   - Use Exp93 `nonpos_weak_alert_replace`.
   - This remains the operating default.

2. **Review lane**
   - Use `review_lane_top1_strict` as the default review lane.
   - It rescues 9 zero-F1 cases while adding the smallest review load:
     about 52 review candidates per 100 datasets.

3. **Investigation mode**
   - Use `review_lane_top2_balanced` only when a user explicitly wants broader
     inspection.
   - It rescues 16 zero-F1 cases but raises review load to about 129 candidates
     per 100 datasets.

4. **Diagnostic only**
   - `review_lane_top3_broad` is not recommended for regular operation.
   - It rescues 20 zero-F1 cases, but the review load is high.

## Practical Meaning

Exp96 does not solve zero-F1 by making the model more aggressive. Instead, it
separates two user experiences:

- "This is an anomaly alert" for high-confidence hard alerts.
- "This may be worth checking" for lower-confidence review candidates.

This matches the operating goal: avoid alarm fatigue while still giving the
system a way to surface missed anomalies for labeling and feedback.

## Decision

Current operating default:

`experiment_93_nonpos_candidate_reranker / nonpos_weak_alert_replace`

Recommended review default:

`experiment_96_review_tier_operational_workflow / review_lane_top1_strict`

## Output Files

- `/Users/minho/Documents/Dataset/experiment_96_review_tier_operational_workflow_results.csv`
- `/Users/minho/Documents/Dataset/experiment_96_review_tier_operational_workflow_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_96_review_tier_operational_workflow_stdout.log`
