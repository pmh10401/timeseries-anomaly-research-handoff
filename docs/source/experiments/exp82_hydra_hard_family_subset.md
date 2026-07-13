# Exp82 - HYDRA Hard-Family Subset

## Purpose

Test whether HYDRA pattern-count features help families where ROCKET-style random convolutions still struggle.

## Scope

- Dataset scope: selected hard families.
- Families:
  - `Phoneme`
  - `CricketZ`
  - `InlineSkate`
  - `GestureMidAirD3`
- Expected datasets: 84.

## Method

- Use aeon `HydraTransformer` as a feature extractor only.
- Do not use `HydraClassifier`.
- Convert HYDRA features into the existing train-normal KNN/local-gap scoring path.
- Thresholds: `count_cap_1pct`, `count_cap_2pct`, `count_cap_3pct`, `family_guard_v1`.

## Success Criteria

- Improve hard-family AUC-PR or oracle F1.
- Reduce zero-F1 datasets inside the four target families.
- Avoid unstable FP growth.

## Status

Queued after Exp81.
