# Exp83 - MultiROCKET + HYDRA Hard339

## Purpose

Check whether combining MultiROCKET's broad convolutional features with HYDRA's pattern-count features improves the Exp75 hard-family subset.

## Scope

- Dataset scope: Exp75 hard-family subset.
- Expected datasets: 339.

## Method

- Use aeon `MultiRocket` and `HydraTransformer` as feature extractors.
- Concatenate both feature spaces.
- Do not use `MultiRocketHydraClassifier`.
- Score with the existing train-normal KNN/local-gap method.
- Thresholds: `count_cap_1pct`, `count_cap_2pct`, `count_cap_3pct`, `family_guard_v1`.

## Success Criteria

- Improve hard subset mean F1 and family macro F1 over ROCKET-only references.
- Reduce zero-F1 count.
- Keep mean FP and train exceed rate within an operationally acceptable range.

## Status

Queued after Exp82.
