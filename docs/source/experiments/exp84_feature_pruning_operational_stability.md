# Exp84 - Feature Pruning For Operational Stability

## Purpose

Test feature pruning as an operating-stability tool, not just a performance optimizer.

## Scope

- Dataset scope: Exp75 hard-family subset.
- Expected datasets: 339.

## Method

- Start from official aeon MultiROCKET and MultiROCKET+HYDRA feature spaces.
- Apply stable-tail pruning on train-normal features.
- Keep features with less unstable train-normal tail behavior.
- Score with KNN/local-gap and automatic thresholds.

## Success Criteria

- Reduce FP or train exceed rate.
- Preserve most of the useful anomaly ranking signal.
- Reduce zero-F1 only if FP does not become operationally noisy.

## Status

Queued after Exp83.
