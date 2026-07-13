# Exp99 Spectral/Derivative Feature Score

Date: 2026-07-09 KST

## Purpose

Exp97 showed that some remaining Exp93 zero-F1 datasets may need spectral or
derivative-oriented features rather than another selector rule.

Exp99 tested a lightweight feature score:

- z-normalized time series
- first and second derivative statistics
- FFT magnitude band distribution
- spectral centroid, entropy, peak ratio
- train-normal robust scaling
- train-normal quantile threshold

The goal was not to replace ROCKET broadly. The goal was to check whether this
feature family can rescue any remaining zero-F1 cases without relying on tail
position shortcuts or labeled family-performance priors.

## Tested Selectors

| Selector | Meaning |
| --- | --- |
| `baseline_exp93_operating` | Current hard-alert default, Exp93 `nonpos_weak_alert_replace` |
| `research_spectral_q98_cap2` | Apply spectral score only to Exp97 spectral/derivative diagnostic rows |
| `train_family_spectral_q99_cap1` | Apply spectral score to pre-defined spectral families, q99 cap1 |
| `train_family_spectral_q98_cap2` | Apply spectral score to pre-defined spectral families, q98 cap2 |
| `train_family_spectral_topk4_q98_cap2` | Same as q98 cap2, but score uses top-4 abnormal feature dimensions |

`research_spectral_q98_cap2` is research-only because Exp97 diagnostic
membership is based on previous test-outcome analysis.

The train-family selectors are closer to operationally usable because they use
dataset family and train-normal distribution only, but they are intentionally
broad and therefore risky.

## Results

| Selector | Mean F1 | Zero-F1 | Mean FP | Mean TP |
| --- | ---: | ---: | ---: | ---: |
| `research_spectral_q98_cap2` | `0.699962` | `235` | `0.595345` | `1.779767` |
| `baseline_exp93_operating` | `0.697575` | `239` | `0.590868` | `1.776186` |
| `train_family_spectral_q99_cap1` | `0.606538` | `357` | `0.587287` | `1.509400` |
| `train_family_spectral_q98_cap2` | `0.601605` | `338` | `0.769024` | `1.529096` |
| `train_family_spectral_topk4_q98_cap2` | `0.596532` | `346` | `0.776186` | `1.521038` |

Movement versus Exp93:

| Selector | Improved | Worsened | Zero-F1 Fixed | New Zero-F1 | TP Delta | FP Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `research_spectral_q98_cap2` | `4` | `0` | `4` | `0` | `+4` | `+5` |
| `train_family_spectral_q99_cap1` | `4` | `121` | `1` | `119` | `-298` | `-4` |
| `train_family_spectral_q98_cap2` | `9` | `169` | `6` | `105` | `-276` | `+199` |
| `train_family_spectral_topk4_q98_cap2` | `7` | `170` | `5` | `112` | `-285` | `+207` |

## What Improved

The research selector fixed four zero-F1 cases:

| Dataset | Family | Exp93 F1 | Exp99 F1 | TP | FP |
| --- | --- | ---: | ---: | ---: | ---: |
| `CricketZ_normal_9` | CricketZ | `0.0` | `0.666667` | `1` | `1` |
| `Phoneme_normal_13` | Phoneme | `0.0` | `0.666667` | `1` | `1` |
| `Phoneme_normal_19` | Phoneme | `0.0` | `0.666667` | `1` | `1` |
| `Phoneme_normal_24` | Phoneme | `0.0` | `0.666667` | `1` | `1` |

This is a real signal: derivative/frequency features can see some anomalies
that the current ROCKET/rank path misses.

## What Got Worse

Broad train-family application failed.

The reason is simple: many datasets inside broad spectral families were already
handled correctly by Exp93. Replacing their alert with the spectral score often
moved the alert away from the true anomaly.

Examples of new failures include:

- `GestureMidAirD1_normal_*`
- `EOGHorizontalSignal_normal_*`
- `EOGVerticalSignal_normal_*`
- `InlineSkate_normal_7`
- `Phoneme_normal_25`

So the spectral score is not a safe family-level replacement.

## Interpretation

Exp99 confirms that a new representation can help, but only narrowly.

Operationally:

1. Do not promote Exp99 as the hard-alert default.
2. Keep Exp93 as the hard-alert default.
3. Keep Exp96 as the review lane default.
4. Use spectral/derivative features as a review signal or a very narrowly gated
   specialist, especially for Phoneme and CricketZ-style failures.

The important lesson is that zero-F1 repair and broad model replacement are
different problems. Exp99 can repair a few hard cases, but broad activation
creates many regressions.

## Next Step

Build the next experiment around gated use, not replacement:

- Only add spectral candidates to the review lane.
- Or use spectral score only when Exp93 is already weak/noisy and a confidence
  guard agrees.
- Do not replace correct Exp93 alerts just because a dataset belongs to a
  spectral-looking family.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_99_spectral_derivative_feature_score_results.csv`
- `/Users/minho/Documents/Dataset/experiment_99_spectral_derivative_feature_score_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_99_spectral_derivative_feature_score_stdout.log`
