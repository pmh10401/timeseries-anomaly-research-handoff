# Experiment 104: Score Dimensionality Sweep

## Purpose

Exp104 tests the user's intended question directly:

> If the feature dimension used to fit anomaly-score sources is increased, does the score itself improve?

This is different from Exp103. Exp103 used higher-dimensional sources only to add review candidates. Exp104 evaluates the score sources themselves.

## Tested Scores

Single-source dimensionality sweep:

- `spectrogram_pca64`
- `spectrogram_pca128`
- `spectrogram_pca256`
- `glcm_rp_pca64`
- `glcm_rp_pca128`
- `glcm_rp_pca256`

Dimension-combination scores:

- `spectrogram_pca64_128_256_rank_mean`
- `glcm_rp_pca64_128_256_rank_mean`
- `spectrogram_glcm_rp_all_dims_rank_mean`

Combination method:

1. Compute each score separately.
2. Convert each score to train-normal percentile rank.
3. Average the percentile-rank scores.
4. Apply the same count-cap threshold policy.

This avoids mixing raw scores with incompatible units.

## Main Results

| Config | Threshold | Mean F1 | Median F1 | Zero-F1 | Mean FP | Alert Precision | Mean Oracle F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_exp93_nonpos_weak_alert_replace | selector | 0.697575 | 1.000000 | 239 | 0.591 | 0.750 | 0.837 |
| glcm_rp_pca64 | count_cap_2pct | 0.364751 | 0.400000 | 235 | 2.852 | 0.334 | 0.658 |
| glcm_rp_pca128 | count_cap_2pct | 0.364751 | 0.400000 | 235 | 2.852 | 0.334 | 0.658 |
| glcm_rp_pca256 | count_cap_2pct | 0.364751 | 0.400000 | 235 | 2.852 | 0.334 | 0.658 |
| spectrogram_pca64 | count_cap_2pct | 0.354287 | 0.333333 | 125 | 5.493 | 0.234 | 0.667 |
| spectrogram_pca128 | count_cap_2pct | 0.295419 | 0.285714 | 70 | 10.574 | 0.155 | 0.646 |
| spectrogram_pca256 | count_cap_2pct | 0.287280 | 0.285714 | 52 | 14.371 | 0.131 | 0.645 |
| spectrogram_glcm_rp_all_dims_rank_mean | count_cap_2pct | 0.334322 | 0.333333 | 263 | 3.486 | 0.309 | 0.790 |
| spectrogram_pca64_128_256_rank_mean | count_cap_2pct | 0.140826 | 0.000000 | 734 | 5.026 | 0.200 | 0.873 |
| glcm_rp_pca64_128_256_rank_mean | count_cap_2pct | 0.145119 | 0.000000 | 829 | 1.027 | 0.459 | 0.769 |

## Interpretation

Increasing score dimensionality does expose more missed anomalies, but it does not produce a better hard-alert score by itself.

The clearest pattern:

- Spectrogram higher dimensions reduce zero-F1 strongly.
- But they also increase false positives sharply.
- Mean F1 drops far below Exp93.

For example:

- `spectrogram_pca64` fixes `193` Exp93 zero-F1 datasets, but worsens `799` datasets.
- `spectrogram_pca128` fixes `214` zero-F1 datasets, but worsens `838` datasets.
- `spectrogram_pca256` fixes `220` zero-F1 datasets, but worsens `849` datasets.

So the higher-dimensional spectrogram score is sensitive, but not operationally stable as an automatic alert source.

The 64/128/256 rank-mean combinations are not good hard-alert replacements. They have good oracle potential in some cases, but thresholded F1 is poor. This means the score contains useful ranking information, but the threshold policy cannot safely turn it into hard alerts without an additional guard.

## Family Signals

The spectrogram higher-dimensional scores most often fixed zero-F1 cases in:

- `PigAirwayPressure`
- `Phoneme`
- `NonInvasiveFetalECGThorax2`
- `GestureMidAirD3`
- `CricketZ`
- `PigCVP`
- `WordSynonyms`
- `FiftyWords`

This supports using higher-dimensional score sources as specialist evidence, review candidates, or gated repair signals.

## Decision

Do not replace Exp93 hard alerts with Exp104 scores directly.

Recommended next direction:

- Keep Exp93 as the hard-alert default.
- Use higher-dimensional spectrogram scores as a gated review or repair source.
- Focus on a guard that activates spectrogram high-dim only when Exp93 is weak and FP risk is controlled.

The important finding is not "higher dimension solves the problem." The finding is:

> Higher dimension improves sensitivity to missed anomalies, but it needs a guard because it also increases false positives.
