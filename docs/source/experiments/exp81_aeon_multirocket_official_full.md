# Exp81 - Official aeon MultiROCKET Full Validation

## Purpose

Validate whether the official aeon `MultiRocket` transformer improves the current ROCKET-family operating model.

## Scope

- Dataset scope: full original repeated-normal benchmark.
- Expected datasets: 1,117.
- Configs:
  - `aeon_multirocket_1024_local_gap_knn3`
  - `aeon_multirocket_2048_local_gap_knn3`

## Method

- Use aeon `MultiRocket` as a feature extractor only.
- Do not use the aeon classifier wrapper.
- Convert features into the existing train-normal anomaly score path:
  - robust feature scaling
  - KNN local-gap score
  - automatic thresholds: `count_cap_2pct`, `count_cap_3pct`, `family_guard_v1`

## Success Criteria

- Improve or match the current ROCKET baseline on mean F1 and median F1.
- Reduce zero-F1 count or improve hard-family coverage.
- Avoid a large FP increase.
- Keep train exceed rate operationally reasonable.

## Status

Running as of 2026-07-09 10:23 KST.
