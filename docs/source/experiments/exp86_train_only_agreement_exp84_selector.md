# Experiment 86: Train-Only Agreement Exp84 Selector

## Purpose

Exp85 improved the full benchmark by using Exp84 on families where prior experiments showed strong gains. That is useful for research, but it is not ideal for production because it depends on past family-level benchmark outcomes.

Exp86 removes that family-performance prior.

Question:

> Can Exp84 be enabled using only train-side safety signals and label-free agreement signals?

## Design

Baseline:

- `baseline_74d_primary`
- `baseline_74d_review_limited`

Exp84 candidates:

- `family_guard_v1`
- `count_cap_2pct`
- `count_cap_3pct`

Allowed selector signals:

- Exp84 `train_exceed_rate`
- Exp74d internal model overlap:
  - `rocket_exp55_overlap`
  - `rocket_exp56_overlap`
  - `exp55_exp56_overlap`
  - `rocket_exp55_or_exp56_overlap_any`
- Exp74d no-alert condition:
  - `predicted_count == 0`

Blocked signal:

- Prior family performance table

The output includes `selector_uses_family_performance_prior=0`, and the script fails if that value is nonzero.

## Important Limitation

Exp84 result CSV does not store selected test indices. Therefore Exp86 cannot yet check true index-level agreement between Exp84 and Exp74d.

The agreement variants use Exp74d internal agreement as a proxy. A future rerun that stores Exp84 selected indices is needed for true agreement selection.

## Results

Best row:

- Selector: `exp84_noalert_repair_te015_fg_else_primary`
- Mean F1: `0.668192`
- Median F1: `1.000000`
- Zero-F1 count: `271`
- Mean FP: `0.541629`
- Mean TP: `1.683080`
- Exp84-used datasets: `2`

Primary baseline:

- Selector: `baseline_74d_primary`
- Mean F1: `0.667296`
- Zero-F1 count: `272`
- Mean FP: `0.541629`
- Mean TP: `1.682184`

Broad switching example:

- Selector: `exp84_train_te015_cap3_else_primary`
- Mean F1: `0.666461`
- Zero-F1 count: `234`
- Mean FP: `0.641898`
- Exp84-used datasets: `122`

## Interpretation

The conservative no-alert repair variant gives a tiny positive gain without increasing mean FP. It used Exp84 only for two datasets:

- `Phoneme_normal_1`
- `WordSynonyms_normal_4`

The broader train-safe and agreement-proxy variants fixed many zero-F1 cases, but they also increased false positives. That FP cost was large enough to reduce mean F1.

## Decision

Exp86 is safer than Exp85 from a production-evidence perspective, but its gain is much smaller.

Current best operating interpretation:

- Keep Exp74d as the default.
- Exp84 can be used as a very narrow no-alert repair fallback.
- For a stronger selector, rerun Exp84 with selected indices saved and test true Exp84-vs-Exp74d agreement.
