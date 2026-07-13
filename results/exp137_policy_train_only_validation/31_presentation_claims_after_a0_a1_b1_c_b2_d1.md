# Presentation Claims After A0/A1/B1/B2

## Verified

- **Exp137 final routing does not use TEST labels or anomaly positions for lane selection.** A1 replay matched 1,117/1,117 stored lanes.
- **In B1 common support, family-guard replacement did not change autonomous hard alerts.** The comparison covers 339 datasets, not all 1,117.
- **B2 calculated the universal Exp84 `count_cap_2pct` source for all 1,117 datasets with zero source errors, and reproduced B1 neutral lanes on its 339-dataset common support.** B2 completed on 1117 datasets: hard alerts 2085, TP 1759, FP 326, micro precision 0.843645, mean hard F1 0.605991.

## Conditional

- **Removing the family-dependent source coverage is feasible under the frozen Exp84 configuration.** Full-coverage B2 changed the final hard lane, so it must be reported as a retrospective policy counterfactual rather than as an invariant result.
- **FP 661 to 314 is the current retrospective Exp137 hard-alert result.** B2 has 326 hard FPs, so the original 314 count must not be attributed to family-neutral B2.

## Not allowed

- TEST-length budget can be removed while preserving the FP reduction. C has not run because the budget unit is not registered.
- Policy-level TRAIN-only preserves the FP reduction. C/D1 have not completed.
- The whole pipeline is end-to-end strict TRAIN-only.
- Exp137 has prospective validation on real equipment.
