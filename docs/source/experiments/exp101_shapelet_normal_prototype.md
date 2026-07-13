# Exp101 Shapelet Normal Prototype

Date: 2026-07-09 KST

## Purpose

Exp101 tested whether shapelet-style normal prototypes can rescue the shape and
outline families identified by Exp97.

The implementation used train-normal data only:

- resample each series to at most `512` points
- sample normal subsequences as shapelet prototypes
- measure each series by minimum distance to those prototypes
- robust-scale the shapelet distance feature vector
- select high-deviation test rows by train-normal quantile

## Results

| Selector | Mean F1 | Zero-F1 | Mean FP | Research target F1 | Research target zero-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_exp93_operating` | `0.697575` | `239` | `0.590868` | `0.000000` | `49` |
| `research_shapelet_q98_cap2` | `0.699067` | `237` | `0.601611` | `0.034014` | `47` |
| `train_family_shapelet_q99_cap1` | `0.614589` | `332` | `0.617726` | `0.020408` | `48` |
| `train_family_shapelet_q98_cap2` | `0.589121` | `327` | `0.810206` | `0.034014` | `47` |
| `train_family_shapelet_topk4_q98_cap2` | `0.581767` | `343` | `0.810206` | `0.013606` | `48` |

Movement versus Exp93:

| Selector | Improved | Worsened | Zero-F1 fixed | New zero-F1 | TP delta | FP delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `research_shapelet_q98_cap2` | `2` | `0` | `2` | `0` | `+2` | `+12` |
| `train_family_shapelet_q99_cap1` | `3` | `103` | `2` | `95` | `-134` | `+30` |
| `train_family_shapelet_q98_cap2` | `3` | `201` | `3` | `91` | `-123` | `+245` |
| `train_family_shapelet_topk4_q98_cap2` | `3` | `192` | `2` | `106` | `-137` | `+245` |

## What Improved

The clean research-only wins were:

| Dataset | Family | Exp93 F1 | Exp101 F1 |
| --- | --- | ---: | ---: |
| `ProximalPhalanxTW_normal_6` | ProximalPhalanxTW | `0.0` | `1.0` |
| `SwedishLeaf_normal_5` | SwedishLeaf | `0.0` | `0.666667` |

## Interpretation

Shapelet normal prototype features have a real but narrow signal.

They should not be applied to broad shape families as hard alerts. Broad family
activation damages many datasets that Exp93 already handles correctly.

Operationally, shapelet should be treated like spectral:

- useful as review/specialist evidence
- not safe as broad hard-alert replacement
- useful mainly when Exp93 already looks weak

## Output Files

- `/Users/minho/Documents/Dataset/experiment_101_shapelet_normal_prototype_results.csv`
- `/Users/minho/Documents/Dataset/experiment_101_shapelet_normal_prototype_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_101_shapelet_normal_prototype_stdout.log`
