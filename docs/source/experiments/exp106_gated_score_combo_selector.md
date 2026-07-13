# Experiment 106: Gated Score Combo Selector

## Purpose

Exp106 tests the next operational step after Exp105:

> Use high-dimensional score combinations only when Exp93 looks weak.

This is a selector experiment. It does not recompute features. It uses completed Exp105 score-combination outputs and evaluates whether gated activation improves review or hard-alert behavior.

## Inputs

Baseline:

- Exp93 `nonpos_weak_alert_replace`

Review context:

- Exp95 `review_top1_strict`
- Exp103 `review_all_higher_dim_sources_when_exp93_weak`

Exp105 combo sources:

- `spectrogram_agreement_3of3`
- `spectrogram_agreement_2of3`
- `spectrogram_glcm_rp_all_dims_rank_min / count_cap_2pct`
- `glcm_rp_agreement_2of3`

## Selectors

| Selector | Meaning |
|---|---|
| `baseline_exp93_hard_only` | Exp93 hard alerts only |
| `review_combo_conservative_when_exp93_weak` | Add conservative combo candidates only when Exp93 is weak and source agrees with context |
| `review_combo_sensitive_when_exp93_weak` | Add more sensitive combo candidates under the same weak/context guard |
| `review_existing_plus_combo_cap3` | Existing review lane plus conservative combo candidates |
| `review_exp103_plus_combo_cap3` | Exp103 review lane plus conservative combo candidates |
| `hard_single_combo_replace_when_exp93_weak` | Diagnostic hard replacement only when one conservative combo proposes one agreed candidate |

## Results

| Selector | Hard Mean F1 | Hard Zero-F1 | Combined Mean F1 | Combined Zero-F1 | Review Candidates | Review TP | Review FP | Review Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `review_exp103_plus_combo_cap3` | 0.697575 | 239 | 0.716644 | 191 | 162 | 48 | 114 | 0.296 |
| `review_combo_sensitive_when_exp93_weak` | 0.697575 | 239 | 0.715570 | 194 | 165 | 45 | 120 | 0.273 |
| `review_combo_conservative_when_exp93_weak` | 0.697575 | 239 | 0.714137 | 198 | 150 | 41 | 109 | 0.273 |
| `baseline_exp93_hard_only` | 0.697575 | 239 | 0.697575 | 239 | 0 | 0 | 0 | 0.000 |
| `hard_single_combo_replace_when_exp93_weak` | 0.697575 | 239 | 0.697575 | 239 | 0 | 0 | 0 | 0.000 |
| `review_existing_plus_combo_cap3` | 0.697575 | 239 | 0.613237 | 189 | 710 | 64 | 646 | 0.090 |

## Interpretation

The gated combo sources are useful as review evidence, but they do not beat Exp103's review lane.

Key points:

- Combo-only conservative review finds `41` TP review hits with `109` FP.
- Combo-only sensitive review finds `45` TP review hits with `120` FP.
- Exp103+combo does not improve beyond Exp103 because the guard selects mostly the same useful candidates.
- Existing review + combo is too noisy: it creates `710` review candidates with only `9.0%` precision.
- Hard replacement is correctly inactive under this conservative guard.

## Decision

Keep Exp93 as the hard-alert baseline.

For review:

- Exp103 remains the stronger review-lane candidate.
- Exp106 combo guards are useful as supporting evidence or explanation, not as a better standalone review lane.

Recommended next step:

- Do not add more broad score combinations immediately.
- Instead, analyze where Exp103 and Exp106 disagree.
- If combo candidates find unique true positives that Exp103 misses, build a smaller disagreement-only review rule.
