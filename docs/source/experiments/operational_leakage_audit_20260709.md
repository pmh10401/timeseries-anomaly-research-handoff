# Operational Leakage Audit

Date: 2026-07-09 KST

## Purpose

Check whether any high-scoring experiments use information that would not be available in real operation.

Leakage/cheating criteria:

- Uses `y_test`, oracle F1, or true labels to choose a strategy.
- Uses benchmark family gains from previous labeled experiments as a selector rule.
- Uses the synthetic evaluation layout, such as anomalies being appended late in the test sequence.
- Uses family failure modes derived from labeled failure analysis instead of train-only or real operating metadata.

## Findings

### Rejected For Operating Use

| Experiment | Issue | Status |
| --- | --- | --- |
| Exp59 `oracle_candidate_upper_bound` | Uses labeled oracle candidate selection. | Already marked non-operational with `operational_candidate=0`. |
| Exp70 `zero_mode_family_repair_selector` | Uses family zero-mode sets derived from prior labeled failure diagnostics. | Exclude from operating-candidate ranking. |
| Exp85 `gain_family_exp84_*` | Uses prior benchmark family gains to decide when to switch to Exp84. | Exclude from operating-candidate ranking. |
| Exp91 | Uses tail-window add/replace. | Rejected. |
| Exp92 | Uses tail-window replacement hybrid. | Rejected. |

### Allowed / Still Valid As Operating Candidates

| Experiment | Reason |
| --- | --- |
| Exp74d | Uses rank/ROCKET guard and large-data budgets, no label or tail-position rule. |
| Exp86 | Uses train-only safety and agreement proxy, no family-performance prior. |
| Exp88 | Uses true index agreement between model outputs, not labels; conservative but not label leakage. |
| Exp89 | Adds Exp84 as guarded candidate without prior family-performance selector. |
| Exp90 `noalert_top1_train_safe_repair` | Uses no-alert state plus train-safe top-ranked model output. No label, family-gain, or tail-position rule. |

## Notes

`stable_tail` in Exp84/87 is not the same as tail-position leakage. It refers to the tail of the **train-normal score distribution/features**, not the end of the test sequence. It is acceptable as a feature-stability concept.

Family metadata itself is not automatically cheating. It becomes risky when the family rule is derived from labeled benchmark outcomes. A production-safe family/recipe prior would need to come from real operating logs, equipment metadata, or train-only distribution diagnostics.

## Current Decision

Current operating-default candidate remains:

`experiment_90_zero_f1_repair_selector / noalert_top1_train_safe_repair`

Rejected results can remain in the dashboard and reports as diagnostics, but must not be used for operating default selection.
