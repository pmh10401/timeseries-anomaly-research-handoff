# Experiment 53a/53b Texture and Train-Robust Scaling Analysis

Date: 2026-07-07

## Purpose

Two untested imaging improvements were separated so their effects could be interpreted cleanly.

- Exp53a: HOG / LBP texture feature extraction
- Exp53b: train-normal based robust scaling before image encoding

Both experiments used the full original evaluation set.

## Completion

Both experiments completed successfully.

| Experiment | Purpose | Rows | Datasets | Configs | Thresholds | Runtime |
|---|---|---:|---:|---:|---:|---:|
| Exp53a | HOG/LBP texture features | 13404 | 1117 | 4 | 3 | 1.32 min |
| Exp53b | train-normal robust scaling | 13404 | 1117 | 4 | 3 | 2.81 min |

No missing dataset marker files were found.

## Baseline Context

Current best default model:

`Exp40 rocket_256_knn3_local_gap + count_cap_3pct`

Best imaging baseline:

`Exp51 image_spectrogram_32_pca32_knn3 + count_cap_3pct`

## Exp53a: HOG / LBP Texture Features

Best strategy:

`lbp_rp_32_pca32_knn3 + count_cap_3pct`

| Metric | Value |
|---|---:|
| Mean F1 | 0.3293 |
| Median F1 | 0.3333 |
| Mean AUC-PR | 0.4395 |
| Mean Oracle F1 | 0.6066 |
| Zero-F1 count | 268 |
| Mean FP | 3.0850 |
| Family macro F1 | 0.2441 |

### Compared With Exp51

| Metric | Exp53a minus Exp51 |
|---|---:|
| Mean F1 delta | -0.0378 |
| Wins | 156 |
| Losses | 243 |
| Ties | 718 |
| FP delta | +0.0358 |
| AUC-PR delta | -0.0983 |
| Oracle F1 delta | -0.0754 |

Exp53a is not better than Exp51 globally. It is also not a better hard-route default.

### Difficulty-Type View

| Difficulty type | Exp53a F1 | Exp51 F1 | Delta | Wins | Losses | FP delta |
|---|---:|---:|---:|---:|---:|---:|
| evaluation_hard | 0.3603 | 0.4003 | -0.0399 | 87 | 150 | +0.1749 |
| model_hard | 0.1997 | 0.2061 | -0.0064 | 31 | 28 | +0.1009 |
| normal | 0.3639 | 0.4490 | -0.0851 | 38 | 65 | -1.0909 |

Texture features are close to Exp51 on model_hard, but still slightly worse.

### Where Texture Helps

Families where Exp53a improved over Exp51:

- InsectEPGSmallTrain
- InsectEPGRegularTrain
- RefrigerationDevices
- ScreenType
- ElectricDevices
- Distal/Middle/Proximal Phalanx variants
- SyntheticControl
- ItalyPowerDemand

This suggests HOG/LBP can help when local texture or recurrence-like binary patterns matter.

### Where Texture Hurts

Families where Exp53a regressed:

- FaceAll / FacesUCR
- ECGFiveDays / CinCECGTorso
- InsectWingbeatSound
- GunPoint
- Trace
- BME
- CBF

The likely issue is that HOG/LBP discards too much continuous amplitude and temporal ordering. It converts the image into local edge/texture summaries, which is useful for some pattern families but destructive for smooth shape or amplitude-sensitive signals.

## Exp53b: Train-Normal Robust Scaling

Best strategy:

`trainrobust_spectrogram_32_pca32_knn3 + count_cap_3pct`

| Metric | Value |
|---|---:|
| Mean F1 | 0.2474 |
| Median F1 | 0.2857 |
| Mean AUC-PR | 0.3348 |
| Mean Oracle F1 | 0.5126 |
| Zero-F1 count | 429 |
| Mean FP | 2.6625 |
| Family macro F1 | 0.1709 |

### Compared With Exp51

| Metric | Exp53b minus Exp51 |
|---|---:|
| Mean F1 delta | -0.1198 |
| Wins | 40 |
| Losses | 288 |
| Ties | 789 |
| FP delta | -0.3868 |
| AUC-PR delta | -0.2030 |
| Oracle F1 delta | -0.1694 |

Train-normal robust scaling reduces false positives a little, but it loses too much detection power.

### Difficulty-Type View

| Difficulty type | Exp53b F1 | Exp51 F1 | Delta | Wins | Losses | FP delta |
|---|---:|---:|---:|---:|---:|---:|
| evaluation_hard | 0.2788 | 0.4003 | -0.1215 | 26 | 166 | -0.1812 |
| model_hard | 0.1642 | 0.2061 | -0.0418 | 4 | 36 | -0.3991 |
| normal | 0.1870 | 0.4490 | -0.2620 | 10 | 86 | -1.8364 |

The current train-normal scaling formulation should not be used as a replacement.

### Why Train-Robust Scaling Failed

The current implementation normalizes each time point using train median and IQR across normal train samples.

That is operationally reasonable, but it can be harmful when:

- train normal data has phase misalignment,
- recipe-normal variation is not synchronized point-by-point,
- the key anomaly is shape/phase/frequency rather than pointwise amplitude deviation,
- IQR is too small at many time points, amplifying noise,
- clipping compresses true anomaly amplitude.

In short, pointwise train-normal scaling assumes aligned sensor curves. Many UCR-style datasets violate that assumption.

## Main Findings

1. Exp51 remains the best imaging baseline.
   - `image_spectrogram_32_pca32_knn3 + count_cap_3pct`

2. HOG/LBP is not globally better, but it is useful as an auxiliary family-specific signal.

3. Train-normal robust scaling, as currently implemented, is too destructive.

4. The bottleneck is no longer image representation alone.
   The next problem is routing and calibration:
   - when to use Exp40,
   - when to use Exp51,
   - when to use texture features,
   - and how to keep false positives low.

## Recommended Next Direction

Do not continue broad image-feature expansion.

The better direction is:

`Exp54: family-aware hard-route selector`

Inputs:

- Exp40 default score
- Exp51 spectrogram score
- Exp53a texture score for selected families

Routing logic:

- default to Exp40,
- use Exp51 for model_hard datasets,
- allow Exp53a texture only for families where it empirically wins,
- require a false-positive guard before switching away from Exp40.

The selector should be evaluated by:

- overall Mean F1,
- model_hard Mean F1,
- Mean FP,
- number of datasets where selector hurts Exp40,
- family-level regression count.

## Practical Conclusion

HOG/LBP and train-normal scaling answered useful questions, but neither should become the main model.

- Keep Exp40 as production default.
- Keep Exp51 as the primary imaging fallback.
- Keep Exp53a only as a candidate family-specific auxiliary signal.
- Do not use Exp53b in its current form.
