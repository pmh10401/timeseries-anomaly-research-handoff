# Operational Flow

1. **1st detection**: train-normal calibrated ROCKET and image-feature scores create candidate indices.
2. **Cross confirmation**: a separate ROCKET feature partition checks whether it independently identifies the same index.
3. **Supplementary confirmation**: a third ROCKET feature partition adds narrow review evidence.
4. **Automated alert**: Exp133 high-confidence indices.
5. **General review**: Exp133 standard-confidence indices not promoted to automated alert.
6. **Priority review**: Exp135 narrow supplementary-confirmed candidates, research-only.
7. **No alert**: every remaining TEST index.

Allowed: TRAIN normal distributions, TEST values, scores computed from them, and TEST size only for index-bound validation in Exp137.

Forbidden for routing: TEST labels, TEST anomaly positions, and past family TEST performance.

Important upstream audit caveat: Exp93 inherits historical Exp84 family_guard rows and Exp89/90 ancestor logic uses TEST-size budgets; see `09_train_only_lineage_audit.md`.