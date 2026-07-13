# Exp137 Operational Triage Design

## Purpose

Exp137 converts the validated anomaly signals into an operational three-way decision without changing the underlying scores or thresholds. The objective is to keep automatic alerts trustworthy, send uncertain cases to human review, and avoid using test labels or test anomaly positions in runtime decisions.

The experiment is an integration and operating-policy evaluation. It is not a new model-performance claim.

## Inputs

- Exp133 `tiered_all_validated_exp93_hard_alerts`
  - `high_confidence_indices`: validated Exp93 hard-alert indices also supported by ROCKET Block B.
  - `standard_confidence_indices`: validated Exp93 hard-alert indices not supported by Block B.
- Exp135 `review_tail1pct_all_standard_and_block_c`
  - One narrow review candidate per eligible dataset, requiring the previously defined all-Standard condition and independent Block C confirmation.

Both inputs must cover the same `1117` datasets. Exp137 must stop with a coverage error if names differ, indices are out of bounds, or output tiers overlap.

## Runtime Decision Policy

For each dataset:

1. `high_confidence_indices` become `hard_alert_indices`.
2. `standard_confidence_indices` become `standard_review_indices`.
3. Exp135 review candidates become `priority_review_indices` after removing any index already present in hard or standard review.
4. Every remaining test index is `no_alert`.

No test label, anomaly count, family performance table, tail position, or oracle score may participate in this routing decision. Labels are loaded only after routing to calculate offline evaluation metrics.

## Output Contract

The detail CSV contains one row per dataset and records:

- hard-alert, standard-review, and priority-review indices and counts;
- TP, FP, FN, precision, recall, and F1 separately for hard alerts;
- TP and FP separately for both review tiers;
- combined recoverable TP/FP/FN/F1 for diagnostic comparison only;
- hard alerts and review requests per 100 datasets;
- explicit leakage flags confirming that labels were not used for routing;
- train-normal count, test size, family, and existing source diagnostics.

The summary CSV contains one operating-policy row across all `1117` datasets. Dashboard comparison must identify Exp137 as an operational triage result rather than treating combined review F1 as automatic-alert F1.

## Evaluation Interpretation

- Hard-alert metrics measure what the system would automatically declare abnormal.
- Review metrics measure human workload and how often that workload contains a true anomaly.
- Combined metrics show the maximum recoverable result if every review item were correctly resolved by a human. They must not be reported as autonomous model performance.

Expected reference values from the unchanged inputs are:

- High-confidence hard alerts: `2005` alerts, `1691` TP, `314` FP, precision `84.339%`.
- Standard review: `639` candidates, `292` TP, `347` FP, precision `45.696%`.
- Priority review: `9` candidates, `8` TP, `1` FP, precision `88.889%`.

## Acceptance Criteria

- Full and exact coverage: `1117` detail rows and no dataset errors.
- Tier disjointness: no index appears in more than one output tier.
- Routing leakage: zero rows use test labels, anomaly position, or oracle metrics for tier assignment.
- Hard-alert non-regression: totals exactly reproduce the Exp133 High tier (`1691` TP and `314` FP).
- Priority-review non-regression: totals exactly reproduce the Exp135 narrow review tier (`8` TP and `1` FP).
- Dashboard and sequential-runner registration are complete.

## Decision Rule

Exp137 can become the operating architecture candidate if the acceptance criteria pass. This does not promote the post-hoc Exp135 priority rule to autonomous alerting. That branch remains a research-backed high-priority review lane until validated on genuinely new equipment, recipe, or time-period data.

## Testing

- Unit-test disjoint routing and precedence.
- Unit-test that changing labels does not change routed indices.
- Unit-test metric separation between automatic alerts and review candidates.
- Run a small smoke dataset subset before the full queued run.
- Verify full CSV coverage, exact source-total reproduction, queue completion, and dashboard reachability after the full run.
