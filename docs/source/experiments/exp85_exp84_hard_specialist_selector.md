# Experiment 85: Exp84 Hard Specialist Selector

## Purpose

Exp84 showed that feature-pruned MultiROCKET+HYDRA is useful on the hard subset, but it is not a full replacement for the current operating selector.

Exp85 tests a narrower operating question:

> Can we keep the full Exp74d operating baseline and use Exp84 only as a hard-subset specialist?

This is a selector-only experiment. It does not retrain feature extractors and does not tune thresholds from test labels.

## Inputs

- Full baseline: `experiment_74d_large_rank_review_tier_split_results.csv`
- Hard specialist: `experiment_84_feature_pruning_operational_stability_results.csv`
- Output detail: `experiment_85_exp84_hard_specialist_selector_results.csv`
- Output summary: `experiment_85_exp84_hard_specialist_selector_summary.csv`

## Selector Variants

- `baseline_74d_primary`
  - Control. Uses Exp74d primary operating baseline for all 1117 datasets.
- `baseline_74d_review_limited`
  - Control. Uses Exp74d review-limited baseline for all 1117 datasets.
- `hard_exp84_cap3_else_primary`
  - Uses Exp84 `count_cap_3pct` on all 339 hard-subset datasets, Exp74d primary elsewhere.
- `hard_exp84_family_guard_else_primary`
  - Uses Exp84 `family_guard_v1` on all 339 hard-subset datasets, Exp74d primary elsewhere.
- `hard_exp84_cap2_else_primary`
  - Uses Exp84 `count_cap_2pct` on all 339 hard-subset datasets, Exp74d primary elsewhere.
- `hard_exp84_guarded_else_primary`
  - Uses Exp84 only when `train_exceed_rate <= 0.015` and predicted rate is at most 6%.
- `gain_family_exp84_cap3_else_primary`
  - Research diagnostic. Uses Exp84 `count_cap_3pct` only for families where Exp84 previously showed clear gains.
- `gain_family_exp84_guarded_else_primary`
  - Research diagnostic plus operational guard.

## Results

Best row:

- Selector: `gain_family_exp84_cap3_else_primary`
- Mean F1: `0.677638`
- Median F1: `1.000000`
- Zero-F1 count: `246`
- Mean FP: `0.667860`
- Mean TP: `1.814682`
- Mean train exceed rate: `0.006002`
- Exp84-used datasets: `138`

Primary baseline:

- Selector: `baseline_74d_primary`
- Mean F1: `0.667296`
- Median F1: `1.000000`
- Zero-F1 count: `272`
- Mean FP: `0.541629`
- Mean TP: `1.682184`

## Interpretation

Exp85 gives a useful but cautious signal.

The best selector improves mean F1 by about `+0.0103` and reduces zero-F1 datasets by `26` compared with the Exp74d primary baseline. That means Exp84 is helping recover some datasets where the primary operating selector completely misses anomalies.

The cost is higher false positives: mean FP rises from `0.541629` to `0.667860`. This is not a catastrophic increase, but it matters because the operating goal is low false-alarm fatigue.

The strongest result is also a research diagnostic, not a production rule yet. The family list comes from prior benchmark outcomes. Before using it operationally, we need to justify the selector with train-only evidence, recipe metadata, or stable historical operating logs.

## Current Decision

Do not replace Exp74d globally.

Treat Exp84 as a promising hard-family specialist. The next useful experiment should make the selector more production-faithful by choosing Exp84 from train-only signals, recipe/family metadata, or cross-model confidence rather than prior test-set family gains.
