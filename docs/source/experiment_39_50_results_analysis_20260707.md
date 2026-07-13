# Experiment 39-50 Results Analysis

Date: 2026-07-07

## Completion Check

All queued experiments through `experiment_50_model_hard_timeseries_imaging` are complete.

- Queue state: `running = null`, `queue = []`
- Latest runner log: `QUEUE empty; runner exiting`
- Active experiment processes: none
- Missing dataset marker files: none found

## Main Conclusion

The best deployable single strategy remains:

`experiment_40_original_score_normalization_sweep`

`rocket_256_knn3_local_gap + count_cap_3pct`

| Metric | Value |
|---|---:|
| Datasets | 1117 |
| Mean F1 | 0.5813 |
| Median F1 | 0.6667 |
| Mean AUC-PR | 0.7603 |
| Mean Oracle F1 | 0.8375 |
| Zero-F1 datasets | 259 |
| Mean FP | 1.8415 |
| Mean predicted count | 4.0698 |

This is the current operational baseline because it has the best global F1/AUC-PR balance while keeping false positives relatively controlled.

## Full Original Benchmark

| Experiment | Best config | Threshold | Mean F1 | Median F1 | AUC-PR | Oracle F1 | Zero F1 | Mean FP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 39 | `retest_actual_median_len_rocket_256_knn3` | `count_cap_3pct` | 0.5577 | 0.6667 | 0.7177 | 0.8004 | 260 | 2.2229 |
| 40 | `rocket_256_knn3_local_gap` | `count_cap_3pct` | 0.5813 | 0.6667 | 0.7603 | 0.8375 | 259 | 1.8415 |
| 41 | `multi_aug_robust_vae` | `adaptive` | 0.3793 | 0.4000 | 0.5322 | 0.6450 | 344 | 2.2748 |
| 42 | `multi_aug_robust_vae` | `adaptive` | 0.3793 | 0.4000 | 0.5322 | 0.6450 | 344 | 2.2748 |
| 43 | `time_z_rocket_256_knn3` | `count_cap_3pct` | 0.5545 | 0.6667 | 0.7158 | 0.7995 | 265 | 2.2158 |
| 44 | `fft_mag_16_knn3` | `count_cap_3pct` | 0.4795 | 0.5000 | 0.6179 | 0.7177 | 368 | 1.5891 |

Interpretation:

- Experiment 40 is better than 39 because `local_gap` separates anomalies more clearly than the retest variants.
- Multi-Aug robust experiments covered all 1117 datasets, but the score itself was weaker. They are not currently better than ROCKET/KNN.
- Explanation-space transforms did not beat the existing `time_z/local_gap` direction.
- Classical embeddings are weaker as a single global model, but they have useful complementary signal.

## Direct Comparison Against Experiment 40 Winner

Baseline: `rocket_256_knn3_local_gap + count_cap_3pct`

| Compared strategy | Mean F1 delta | Wins | Losses | Ties | Mean FP delta | AUC-PR delta |
|---|---:|---:|---:|---:|---:|---:|
| Exp39 actual-median retest | -0.0236 | 169 | 267 | 681 | +0.3814 | -0.0425 |
| Exp41 Multi-Aug adaptive | -0.2020 | 155 | 624 | 338 | +0.4333 | -0.2280 |
| Exp42 Multi-Aug adaptive | -0.2020 | 155 | 624 | 338 | +0.4333 | -0.2280 |
| Exp43 time-z ROCKET | -0.0269 | 166 | 269 | 682 | +0.3742 | -0.0445 |
| Exp44 FFT-mag 16 | -0.1018 | 213 | 414 | 490 | -0.2525 | -0.1424 |

Important point: no global replacement beats experiment 40. But experiment 44 wins on 213 datasets and has lower FP, so it is useful as a secondary signal or selector candidate.

## Model-Hard Subset

| Experiment | Direction | Best mean F1 | Median F1 | Zero-F1 count | Mean FP | Comment |
|---|---|---:|---:|---:|---:|---|
| 45 | diagnostic harness | 0.0660 | 0.0000 | 15/20 | 6.9500 | Confirms hard cases are truly hard |
| 46 | interval / DrCIF-lite | 0.0725 | 0.0000 | 9/13 | 8.9231 | Some signal for ElectricDevices |
| 47 | frequency ROCKET | 0.1434 | 0.0000 | 23/32 | 2.7812 | Helps Ford/Ethanol more than others |
| 48 | shapelet prototype | 0.0531 | 0.0000 | 14/17 | 2.5882 | Current prototype is too weak |
| 49 | anomaly injection | 0.0404 | 0.0000 | 34/46 | 17.4130 | Poor calibration; over-alerts |
| 50 | time-series imaging | 0.2254 | 0.2222 | 35/86 | 3.9186 | Best hard-subset direction |

Experiment 50 is the most promising model-hard experiment. It performs especially well on:

- Fish: mean best F1 0.7000
- GestureMidAir D1/D2/D3: mean best F1 around 0.4000
- ElectricDevices: mean best F1 0.3705
- FordB: mean best F1 0.3524
- MelbournePedestrian: mean best F1 0.3019
- FordA: mean best F1 0.2956
- UWaveGestureLibraryX: mean best F1 0.2814

For the 86 overlapping hard datasets, the best imaging strategy beat experiment 40 by mean F1 +0.1565, with 47 wins, 12 losses, and 27 ties. This is the strongest evidence that imaging should be refined for hard families rather than discarded.

## Recommended Next Experiments

1. Keep experiment 40 as the default production baseline.
2. Build a selector/ensemble that uses experiment 40 by default and activates imaging for families where experiment 50 clearly helps.
3. Add a conservative FP guard to imaging before considering it operational.
4. Do not continue the current anomaly-injection path until calibration is redesigned.
5. Treat classical embeddings as auxiliary features, not as a standalone replacement.
6. Investigate still-hard families separately: ScreenType, Earthquakes, Computers, RefrigerationDevices, and many Phoneme subsets.

## Practical Strategy

The next best direction is not one more universal model. The evidence says the project needs a family-aware routing strategy:

- Default route: `rocket_256_knn3_local_gap + count_cap_3pct`
- Hard visual/shape route: time-series imaging
- Frequency/sensor route: FFT-band score
- Auxiliary low-FP route: classical FFT embedding

This matches the operational goal: find clear anomalies first, reduce false alarms, and only ask the user to label cases that are worth reviewing.
