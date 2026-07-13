# Experiment 52 Imaging Multiscale Fusion Analysis

Date: 2026-07-07

## Purpose

Experiment 52 tested the highest-priority improvements to time-series imaging:

1. Spectrogram + GASF + RP fusion
2. Wavelet-like scalogram
3. GASF + GADF fusion
4. Multi-scale recurrence plot
5. 64x64 / PCA64 resolution expansion

The goal was not to replace Exp40 immediately, but to see whether richer image encodings improve the hard-case route discovered in Exp51.

## Completion

Experiment 52 completed successfully.

- Datasets: 1117
- Rows: 16755
- Expected rows: 1117 datasets x 5 configs x 3 thresholds = 16755
- Missing datasets: 0
- Runtime: 15.73 minutes

The final bottleneck was `StarLightCurves_normal_3`, which has 4261 train series and 5400 test series. The original scalogram implementation was slow for this case, so scalogram generation was vectorized for future runs.

## Top Results

| Rank | Config | Threshold | Mean F1 | Median F1 | AUC-PR | Oracle F1 | Zero F1 | Mean FP |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `image_scalogram_64_pca64_knn3` | `count_cap_3pct` | 0.3629 | 0.3333 | 0.5220 | 0.6717 | 151 | 4.1603 |
| 2 | `fusion_spectrogram_gasf_rp_64_pca64_knn3` | `count_cap_3pct` | 0.3622 | 0.3333 | 0.5179 | 0.6671 | 154 | 4.1961 |
| 3 | `image_scalogram_64_pca64_knn3` | `family_guard_v1` | 0.3565 | 0.3333 | 0.5220 | 0.6717 | 164 | 3.9937 |
| 4 | `image_scalogram_64_pca64_knn3` | `count_cap_2pct` | 0.3564 | 0.3333 | 0.5220 | 0.6717 | 166 | 3.9445 |
| 5 | `fusion_spectrogram_gasf_rp_64_pca64_knn3` | `family_guard_v1` | 0.3560 | 0.3333 | 0.5179 | 0.6671 | 166 | 4.0233 |

## Comparison With Exp51

Exp51 winner:

`image_spectrogram_32_pca32_knn3 + count_cap_3pct`

Exp52 winner:

`image_scalogram_64_pca64_knn3 + count_cap_3pct`

| Metric | Exp51 winner | Exp52 winner | Direction |
|---|---:|---:|---|
| Mean F1 | 0.3672 | 0.3629 | Worse |
| Median F1 | 0.4000 | 0.3333 | Worse |
| AUC-PR | 0.5378 | 0.5220 | Worse |
| Oracle F1 | 0.6820 | 0.6717 | Worse |
| Zero-F1 count | 246 | 151 | Better |
| Mean FP | 3.0492 | 4.1603 | Worse |

Exp52 reduces the number of zero-F1 datasets, but it does this by predicting more alerts. The extra recall is not clean enough for the current operational goal because false positives rise meaningfully.

## Comparison With Exp40 Baseline

Exp40 baseline:

`rocket_256_knn3_local_gap + count_cap_3pct`

Exp52 winner versus Exp40:

| Metric | Value |
|---|---:|
| Mean F1 delta | -0.2185 |
| Wins | 267 |
| Losses | 663 |
| Ties | 187 |
| Mean FP delta | +2.3187 |
| AUC-PR delta | -0.2382 |
| Oracle F1 delta | -0.1658 |

Exp52 should not replace Exp40.

## Difficulty-Type Breakdown

| Difficulty type | Datasets | Exp52 F1 | Exp40 F1 | Delta | Wins | Losses | FP delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| evaluation_hard | 789 | 0.3888 | 0.7003 | -0.3115 | 108 | 543 | +3.0380 |
| model_hard | 218 | 0.2047 | 0.0728 | +0.1318 | 138 | 33 | +2.7982 |
| normal | 110 | 0.4900 | 0.7355 | -0.2455 | 21 | 87 | -3.7909 |

The useful signal remains concentrated in `model_hard`. But compared with Exp51, Exp52 gets that signal with much higher false positives.

## Interpretation

The richer image encodings did what we expected in one sense:

- They found more non-zero detections.
- Scalogram and fusion captured some hard-case signal.
- `model_hard` still improved over Exp40.

But they did not improve the operational tradeoff:

- Mean F1 did not beat Exp51.
- Median F1 dropped.
- AUC-PR dropped.
- FP increased substantially.

This means the main limitation is probably not just image resolution or image type. The next bottleneck is calibration and route selection.

## Recommendation

Do not use Exp52 as the next operational route.

Keep Exp51 as the stronger imaging baseline:

`image_spectrogram_32_pca32_knn3 + count_cap_3pct`

Use Exp52 findings selectively:

- Scalogram is useful as a candidate auxiliary score because it reduces zero-F1 count.
- `fusion_spectrogram_gasf_rp_64` is worth keeping as a research signal, but not as an alerting model yet.
- GASF+GADF has high AUC-PR and family macro F1, but too many false positives.

Next experiment should be a selector/calibration experiment, not another larger image encoding:

`Exp53: Exp40 + Exp51/Exp52 hard-route selector with FP guard`

Candidate rule:

- default to Exp40,
- activate imaging only when dataset is `model_hard` or family is imaging-positive,
- prefer Exp51 spectrogram unless scalogram has clearly better train-normal separation,
- apply stricter FP guard before emitting alerts.
