# Exp84 Source Coverage Report

| Check | Count |
|---|---|
| Exp137 datasets | 1117 |
| FG/cap2/cap3 all present | 339 |
| cap2 only | 0 |
| No existing Exp84 source | 778 |
| Eligible under HARD_SCORE_FAMILIES | 339 |
| Requires new B2 source computation | 778 |

Existing source rows use the same recorded feature configuration when present. The result CSV does not store raw score vectors, so score-vector equality cannot be independently hashed; this is recorded as `not_verifiable_score_vectors_not_stored` rather than inferred.
The code configuration fixes Exp84 random_state to 20260717, but row-level seed provenance is not stored in the historical CSV.
B2 requires all 1,117 datasets. Missing rows must be recomputed, not treated as empty candidates or silently excluded.
