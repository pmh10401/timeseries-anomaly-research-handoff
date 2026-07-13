# Experiment 51 Full Time-Series Imaging Analysis

Date: 2026-07-07

## Purpose

Experiment 50 showed that time-series imaging helps several model-hard datasets. Experiment 51 tested the same imaging family on the full original evaluation set to answer one question:

Should imaging replace the current default strategy, or should it only be used as a selective helper?

## Completion

Experiment 51 completed successfully.

- Target datasets: 1117
- Result rows: 17872
- Configs: 4 image transforms
- Threshold methods: 4
- Missing dataset file: none
- Elapsed time: about 1.6 minutes
- Dashboard: experiment 51 is visible in completed results

## Best Full-Dataset Imaging Result

Best strategy:

`image_spectrogram_32_pca32_knn3 + count_cap_3pct`

| Metric | Value |
|---|---:|
| Mean F1 | 0.3672 |
| Median F1 | 0.4000 |
| Mean AUC-PR | 0.5378 |
| Mean Oracle F1 | 0.6820 |
| Zero-F1 datasets | 246 |
| Mean FP | 3.0492 |
| Mean predicted count | 4.6276 |
| Family macro F1 | 0.2906 |

## Top Imaging Strategies

| Rank | Config | Threshold | Mean F1 | Median F1 | AUC-PR | Zero F1 | Mean FP |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `image_spectrogram_32_pca32_knn3` | `count_cap_3pct` | 0.3672 | 0.4000 | 0.5378 | 246 | 3.0492 |
| 2 | `image_spectrogram_32_pca32_knn3` | `family_guard_v1` | 0.3645 | 0.4000 | 0.5378 | 253 | 2.7735 |
| 3 | `image_spectrogram_32_pca32_knn3` | `adaptive_v0` | 0.3623 | 0.3951 | 0.5378 | 256 | 2.9740 |
| 4 | `image_spectrogram_32_pca32_knn3` | `count_cap_2pct` | 0.3598 | 0.3750 | 0.5378 | 261 | 2.6732 |
| 5 | `image_gasf_32_pca32_knn3` | `adaptive_v0` | 0.3509 | 0.4000 | 0.5263 | 237 | 3.0976 |

Spectrogram is the best global image transform. GASF is close in median F1 and family macro F1, but lower in mean F1.

## Direct Comparison With Current Baseline

Current production baseline:

`experiment_40_original_score_normalization_sweep`

`rocket_256_knn3_local_gap + count_cap_3pct`

Best Exp51 imaging strategy versus Exp40:

| Comparison | Value |
|---|---:|
| Mean F1 delta | -0.2142 |
| Wins | 220 |
| Losses | 647 |
| Ties | 250 |
| Mean FP delta | +1.2077 |
| AUC-PR delta | -0.2225 |
| Oracle F1 delta | -0.1555 |

Conclusion: imaging should not replace Exp40 as the global default.

## Why Imaging Still Matters

Although imaging is weaker globally, it improves the exact kind of datasets we were worried about.

For `model_hard` datasets:

| Difficulty type | Datasets | Exp51 F1 | Exp40 F1 | Delta | Wins | Losses | Ties | FP delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| model_hard | 218 | 0.2061 | 0.0728 | +0.1332 | 103 | 26 | 89 | +0.5917 |
| evaluation_hard | 789 | 0.4003 | 0.7003 | -0.3001 | 105 | 525 | 159 | +1.7275 |
| normal | 110 | 0.4490 | 0.7355 | -0.2865 | 12 | 96 | 2 | -1.3000 |

This is the key finding:

Imaging is not a universal model. It is a hard-case rescue signal.

## Family-Level Improvements

Families where the best imaging strategy improved over Exp40:

| Family | Datasets | Mean F1 delta | Wins | Losses | Exp51 F1 | Exp40 F1 | FP delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wine | 2 | +0.5000 | 1 | 0 | 0.5000 | 0.0000 | +1.5000 |
| ArrowHead | 3 | +0.3889 | 2 | 0 | 0.3889 | 0.0000 | +1.3333 |
| MiddlePhalanxOutlineAgeGroup | 3 | +0.3333 | 3 | 0 | 0.4286 | 0.0952 | +3.3333 |
| Fish | 7 | +0.3122 | 4 | 2 | 0.5265 | 0.2143 | +1.1429 |
| DiatomSizeReduction | 4 | +0.1806 | 2 | 0 | 0.8056 | 0.6250 | +1.0000 |
| FordB | 2 | +0.1058 | 2 | 0 | 0.3263 | 0.2205 | -27.0000 |
| EthanolLevel | 4 | +0.0625 | 1 | 0 | 0.0625 | 0.0000 | +1.2500 |
| Phoneme | 39 | +0.0462 | 16 | 10 | 0.2787 | 0.2325 | +2.9487 |

The strongest practical signal is in shape/outline and model-hard families.

## Family-Level Regressions

Families where imaging should not be used as default:

| Family | Datasets | Mean F1 delta | Exp51 F1 | Exp40 F1 |
|---|---:|---:|---:|---:|
| InsectEPGSmallTrain | 3 | -0.8000 | 0.0000 | 0.8000 |
| TwoPatterns | 4 | -0.7908 | 0.0907 | 0.8814 |
| SemgHandMovementCh2 | 6 | -0.7194 | 0.0000 | 0.7194 |
| Plane | 7 | -0.7024 | 0.2500 | 0.9524 |
| Chinatown | 2 | -0.7000 | 0.0000 | 0.7000 |
| FreezerRegularTrain | 2 | -0.6869 | 0.0982 | 0.7852 |
| GesturePebbleZ1 | 6 | -0.6667 | 0.1944 | 0.8611 |
| CricketX | 12 | -0.5556 | 0.0000 | 0.5556 |
| ACSF1 | 10 | -0.4833 | 0.3333 | 0.8167 |

This explains why full replacement is risky: many easy or moderately hard datasets already work well with Exp40 and are damaged by image conversion.

## Selector Upper Bound

If we could choose the best imaging config per dataset after seeing labels:

| Metric | Value |
|---|---:|
| Mean F1 | 0.4648 |
| Median F1 | 0.5000 |
| Zero-F1 datasets | 145 |
| Mean FP | 2.4414 |
| Mean F1 delta vs Exp40 | -0.1165 |
| Wins vs Exp40 | 312 |
| Losses vs Exp40 | 531 |
| Ties vs Exp40 | 274 |

Even the imaging-only selector upper bound does not beat Exp40 globally. Therefore the next selector should compare Exp40 and imaging together, not select only among imaging variants.

## Recommendation

Do not replace Exp40.

Use Exp51 as evidence for a routed model:

1. Default route: `rocket_256_knn3_local_gap + count_cap_3pct`
2. Hard-case route: `image_spectrogram_32_pca32_knn3 + family_guard_v1` or `count_cap_3pct`
3. Candidate activation rule:
   - dataset is `model_hard`, or
   - family belongs to known imaging-positive families, or
   - Exp40 score separation is weak but image score separation is strong
4. Add an FP guard before production use, because imaging improves model-hard F1 but can add false positives.

Operationally, imaging is valuable as a second opinion for difficult datasets, not as the main alarm engine.
