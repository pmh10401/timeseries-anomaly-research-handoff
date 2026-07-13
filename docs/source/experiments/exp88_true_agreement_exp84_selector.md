# Experiment 88: True Agreement Exp84 Selector

## Purpose

Exp87 saved Exp84 selected test indices. Exp88 uses those indices to test true agreement selectors.

Question:

> If Exp74d and Exp84 flag the same test index, can we use that agreement to improve operating performance without increasing false alarms?

## Inputs

- Exp74d baseline:
  - `experiment_74d_large_rank_review_tier_split_results.csv`
- Exp87 indexed Exp84 specialist:
  - `experiment_87_exp84_index_diagnostics_results.csv`

## Selector Variants

- `baseline_74d_primary`
  - Control. Existing operating baseline.
- `baseline_74d_review_limited`
  - Control. Existing review-limited baseline.
- `agreement_cap3_intersection_else_primary`
  - On hard-subset datasets, use `Exp74d selected_indices ∩ Exp87 count_cap_3pct selected_indices` when non-empty; otherwise use Exp74d.
- `agreement_fg_intersection_else_primary`
  - Same as above, but Exp87 `family_guard_v1`.
- `agreement_cap2_intersection_else_primary`
  - Same as above, but Exp87 `count_cap_2pct`.
- `agreement_cap3_only_else_primary`
  - Use only the agreement intersection on hard-subset datasets. If no agreement, alert nothing.
- `agreement_fg_only_else_primary`
  - Same as above, but Exp87 `family_guard_v1`.
- `agreement_or_top1_noalert_fg_else_primary`
  - Use true agreement when present. If Exp74d has no alert, allow Exp87 top-1 repair when train-safe.
- `top1_noalert_margin_fg_else_primary`
  - Use Exp74d by default. If Exp74d has no alert, allow Exp87 top-1 repair when train-safe.
- `top1_noalert_strong_margin_fg_else_primary`
  - Same as top-1 repair, with stronger score-margin requirements.

## Results

Best row:

- Selector: `top1_noalert_margin_fg_else_primary`
- Mean F1: `0.668192`
- Median F1: `1.000000`
- Zero-F1 count: `271`
- Mean FP: `0.541629`
- Mean TP: `1.683080`
- Exp87-used datasets: `1`

Primary baseline:

- Selector: `baseline_74d_primary`
- Mean F1: `0.667296`
- Zero-F1 count: `272`
- Mean FP: `0.541629`
- Mean TP: `1.682184`

Best true-agreement intersection:

- Selector: `agreement_cap3_intersection_else_primary`
- Mean F1: `0.664710`
- Zero-F1 count: `275`
- Mean FP: `0.435989`
- Mean TP: `1.623993`
- Exp87-used datasets: `275`

## Interpretation

True agreement works as a false-positive reducer, but it is too conservative as a broad selector.

The best intersection selector reduced mean FP from `0.541629` to `0.435989`, but mean TP also fell from `1.682184` to `1.623993`. That TP loss caused mean F1 to drop below the Exp74d baseline.

The top-1 no-alert repair remains the safest production-like improvement:

- It changed only one dataset: `Phoneme_normal_1`.
- It fixed one zero-F1 case.
- It did not increase mean FP.

## Decision

Do not use strict true-agreement intersection as the default operating selector.

Keep the safer pattern:

- Exp74d remains the default.
- Exp84/Exp87 can be used as a narrow no-alert repair.
- Agreement should be used as a confidence annotation or review-tier signal, not as a hard replacement rule.
