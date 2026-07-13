# Experiment 87: Exp84 Index Diagnostics

## Purpose

Exp86 could not test true Exp84-vs-Exp74d agreement because Exp84 did not store selected test indices.

Exp87 reruns the strongest Exp84 specialist config and saves the missing diagnostic fields.

## Config

- Target: Exp75/Exp84 hard-family subset, `339` datasets
- Feature extractor: `aeon_multirocket_hydra`
- Config: `aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3`
- Thresholds:
  - `count_cap_2pct`
  - `count_cap_3pct`
  - `family_guard_v1`

## New Detail Columns

- `selected_indices`
  - Test indices where `test_score > threshold`.
- `top_score_indices`
  - Top test indices by score, preserving score rank order.
- `top_score_values`
  - Scores for `top_score_indices`.
- `selected_score_max`
  - Highest score among threshold-selected indices.
- `selected_score_min`
  - Lowest score among threshold-selected indices.
- `top1_score`
  - Highest test score.
- `top2_score`
  - Second-highest test score.
- `top1_top2_margin`
  - Confidence gap between top-1 and top-2 candidates.
- `top1_threshold_margin`
  - Distance from top-1 score to threshold.

## Why This Matters

These fields let us build production-friendlier selectors without using prior family performance tables.

Next experiments enabled by Exp87:

- True agreement selector:
  - Use Exp84 only when Exp84 and Exp74d flag the same test index.
- Top-1 no-alert repair:
  - If Exp74d flags nothing, allow only Exp84's strongest candidate.
- Score-margin selector:
  - Allow Exp84 only when top-1 is clearly separated from top-2 and threshold.

## Status

Started on 2026-07-09 12:01 KST through the sequential queue runner.
