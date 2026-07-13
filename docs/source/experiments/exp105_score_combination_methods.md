# Experiment 105: Score Combination Methods

## Purpose

Exp105 follows Exp104 and asks:

> If 64/128/256-dimensional scores are useful but unstable, can a better combination method make them operationally safer?

Exp104 used only rank-mean combination. Exp105 compares additional score-combination methods.

## Tested Methods

Within each source family:

- `rank_mean`
- `rank_max`
- `rank_min`
- `weighted_50_35_15`
- `weighted_20_30_50`
- `agreement_2of3`
- `agreement_3of3`

Across spectrogram + GLCM/RP:

- `spectrogram_glcm_rp_all_dims_rank_mean`
- `spectrogram_glcm_rp_all_dims_rank_max`
- `spectrogram_glcm_rp_all_dims_rank_min`

All rank combinations use train-normal percentile ranks before combination.

Agreement methods use dimension-level threshold exceedance:

- `agreement_2of3`: at least two of 64/128/256 agree.
- `agreement_3of3`: all three agree.

## Main Results

| Config | Method | Threshold | Mean F1 | Zero-F1 | Mean FP | Precision | Oracle F1 |
|---|---|---|---:|---:|---:|---:|---:|
| baseline_exp93_nonpos_weak_alert_replace | baseline | selector | 0.697575 | 239 | 0.591 | 0.750 | 0.837 |
| glcm_rp_agreement_2of3 | agreement_2of3 | fixed | 0.364751 | 235 | 2.852 | 0.334 | 0.769 |
| glcm_rp_agreement_3of3 | agreement_3of3 | fixed | 0.364751 | 235 | 2.852 | 0.334 | 0.769 |
| spectrogram_agreement_3of3 | agreement_3of3 | fixed | 0.352146 | 138 | 5.201 | 0.234 | 0.872 |
| spectrogram_glcm_rp_all_dims_rank_min | rank_min | count_cap_3pct | 0.335069 | 274 | 3.686 | 0.299 | 0.793 |
| spectrogram_glcm_rp_all_dims_rank_mean | rank_mean | count_cap_2pct | 0.334322 | 263 | 3.486 | 0.309 | 0.790 |
| spectrogram_agreement_2of3 | agreement_2of3 | fixed | 0.296189 | 70 | 10.304 | 0.158 | 0.872 |
| spectrogram_rank_max | rank_max | count_cap_2pct | 0.068683 | 832 | 9.794 | 0.128 | 0.872 |

## Baseline Movement

Compared with Exp93:

| Config | Improved | Worsened | Zero-F1 Fixed | New Zero-F1 | Mean FP |
|---|---:|---:|---:|---:|---:|
| glcm_rp_agreement_2of3 | 152 | 800 | 131 | 127 | 2.852 |
| spectrogram_agreement_3of3 | 231 | 791 | 192 | 91 | 5.201 |
| spectrogram_agreement_2of3 | 247 | 838 | 214 | 45 | 10.304 |
| spectrogram_glcm_rp_all_dims_rank_min / 2pct | 149 | 793 | 120 | 177 | 3.143 |
| spectrogram_glcm_rp_all_dims_rank_mean / 2pct | 155 | 807 | 130 | 154 | 3.486 |
| spectrogram_rank_max / 2pct | 86 | 847 | 60 | 593 | 9.794 |

## Interpretation

No score-combination method is safe as a direct hard-alert replacement for Exp93.

The important finding is more specific:

- `rank_max` is too aggressive. It finds some missed anomalies, but creates many false positives and many new zero-F1 regressions.
- `rank_min` is more conservative and works better than `rank_max`, especially across spectrogram + GLCM/RP.
- `agreement_2of3` is sensitive but still too FP-heavy.
- `agreement_3of3` is safer than `agreement_2of3` for spectrogram, but still not safe enough for automatic hard alerts.
- GLCM/RP dimensions behave almost identically across 64/128/256 in this setup, so agreement does not add much beyond the base GLCM/RP score.

The best operational lesson is:

> Better combination methods do not make high-dimensional imaging scores good hard-alert replacements, but they provide useful specialist evidence for review or guarded repair.

## Recommended Next Step

Exp106 should not try another broad replacement.

Recommended direction:

- Use Exp93 as the hard-alert base.
- Use high-dimensional spectrogram/GLCM combination only when Exp93 is weak.
- Candidate guards to test:
  - Exp93 predicted count <= 1
  - Exp93 train exceed rate <= 0.015
  - high-dim score must agree with Exp93 top candidate or Exp103 review candidate
  - hard replacement only when FP budget is low
  - otherwise add to review lane only

The strongest candidates for Exp106 are:

- `spectrogram_agreement_3of3`
- `spectrogram_glcm_rp_all_dims_rank_min`
- `glcm_rp_agreement_2of3`

These should be used as gated evidence, not as standalone alert models.
