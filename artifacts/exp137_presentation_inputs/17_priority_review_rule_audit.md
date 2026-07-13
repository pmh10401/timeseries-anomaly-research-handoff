# Priority Review Rule Audit

- Total candidates: 9
- Offline TP/FP: 8 / 1
- Precision: 88.889%
- 95% Wilson confidence interval: [56.500%, 98.011%]

## Rule

The final configuration is `review_tail1pct_all_standard_and_block_c`. It takes a candidate from Exp134 only when all existing Exp133 candidates are in the general-review tier and the supplementary independent ROCKET check agrees.

## Why It Is Retrospective

Exp137 writes `priority_review_posthoc_research_rule=1` and `prospective_priority_validation_required=1`. The observed 8/9 precision is based on 9 cases only, so it is too small and too retrospective to justify autonomous alerts. It may be shown only as a review-prioritization finding.

`tail1pct` denotes a 1% TRAIN score-tail threshold in Exp134, not a TEST sequence position.