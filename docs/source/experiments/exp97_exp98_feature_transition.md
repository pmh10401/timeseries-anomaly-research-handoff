# Exp97/Exp98 Feature Transition Results

Date: 2026-07-09 KST

## Purpose

After Exp94b closed the hard-alert replacement path, Exp97 started the feature
and score improvement phase.

The goal was to classify remaining Exp93 zero-F1 cases and test the first
feature-side idea: tiny-train normal pooling.

## Exp97: Zero-F1 Feature Need Diagnosis

Exp97 classified the remaining `239` Exp93 zero-F1 rows.

| Feature need | Rows | Main families |
| --- | ---: | --- |
| `E_score_calibration_candidate` | 67 | NonInvasiveFetalECGThorax1/2, DodgerLoopDay, ACSF1 |
| `A_tiny_train_normal_pooling` | 56 | PigAirwayPressure, PigCVP, GestureMidAirD3, Phoneme |
| `D_shapelet_prototype_feature` | 49 | WordSynonyms, Adiac, FiftyWords, Fish |
| `C_spectral_derivative_feature` | 45 | Phoneme, CricketZ, EOGVerticalSignal |
| `B_review_tier_candidate` | 20 | InlineSkate, AllGestureWiimote, Haptics |
| `F_new_representation_needed` | 2 | Earthquakes, PhalangesOutlinesCorrect |

Interpretation:

- The largest group is not new feature extraction, but score calibration.
- Tiny-train is still a large and concrete problem.
- Spectral and shape/prototype features are the next two feature families to
  test.

## Exp98: Tiny-Train Normal Pooling

Exp98 tested family-pooled train-normal thresholds for Exp97's
`A_tiny_train_normal_pooling` rows only.

Best research result:

| Selector | Mean F1 | Zero-F1 | Mean FP | Target zero-F1 |
| --- | ---: | ---: | ---: | ---: |
| Exp93 baseline | 0.697575 | 239 | 0.590868 | 56 |
| `multi_source_pool_union_q98_cap2` | 0.723239 | 196 | 0.602507 | 13 |

Dataset movement versus Exp93:

- Improved: `43`
- Worsened: `0`
- Zero-F1 fixed: `43`
- New zero-F1: `0`
- TP delta: `+43`
- FP delta: `+13`

This is a strong signal that tiny-train normal pooling can help.

However, Exp98 is a research diagnostic, not an operating model, because it uses
Exp97 zero-F1 membership to decide where to apply the pooling rule. That target
membership is based on test-set outcome analysis.

## Exp98b: Train-Only Tiny Pooling

Exp98b removed that leakage by applying pooling using only train-time
information:

- `train_normal_count <= 10`, or
- Pig family.

Result:

| Selector | Mean F1 | Zero-F1 | Mean FP | Target zero-F1 |
| --- | ---: | ---: | ---: | ---: |
| Exp93 baseline | 0.697575 | 239 | 0.590868 | 56 |
| `multi_source_pool_union_q98_cap2` | 0.645352 | 196 | 0.836168 | 13 |
| `multi_source_pool_union_q99_cap1` | 0.686832 | 251 | 0.601611 | 68 |

Movement for `multi_source_pool_union_q98_cap2`:

- Improved: `43`
- Worsened: `261`
- Zero-F1 fixed: `43`
- New zero-F1: `0`
- TP delta: `+43`
- FP delta: `+274`

Interpretation:

The pooling signal is real, but applying it broadly as hard alerts is too
aggressive. It fixes the same target rows but adds many false positives to
datasets that were already handled well.

## Decision

Do not promote Exp98 or Exp98b as hard-alert operating defaults.

Use the finding as follows:

1. Keep Exp93 as the hard-alert default.
2. Keep Exp96 `review_lane_top1_strict` as the review default.
3. Convert tiny-train pooling into a review-lane or gated specialist, not a
   hard-alert replacement.

## Next Step

Proceed to:

`Exp99 spectral_derivative_feature_score`

Reason:

- Exp97 identified `45` spectral/derivative candidates.
- Main affected families include Phoneme, CricketZ, EOGVerticalSignal,
  EthanolLevel, Haptics, and gesture families.
- This tests a genuinely new representation rather than another selector rule.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_97_zero_f1_feature_need_diagnosis_results.csv`
- `/Users/minho/Documents/Dataset/experiment_97_zero_f1_feature_need_diagnosis_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_98_tiny_train_normal_pooling_results.csv`
- `/Users/minho/Documents/Dataset/experiment_98_tiny_train_normal_pooling_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_98b_train_only_tiny_pooling_results.csv`
- `/Users/minho/Documents/Dataset/experiment_98b_train_only_tiny_pooling_summary.csv`
