# Alert Budget Decision Memo

## Decision status

**No primary C/D1 budget is approved.** The current data does not verify whether a user reviews alerts per run, wafer, sensor-step, batch, day, or whole test dataset. Therefore no arbitrary K and no `ceil(rate * n_train)` policy will be used as a primary operating budget.

## Options

| Option | Required metadata | Current implementability | Operating interpretation | Strict TRAIN-only suitability | Primary C/D1 use |
|---|---|---|---|---|---|
| A. Fixed operating cap K | verified review unit and user/safety capacity | not verifiable from current data | maximum alerts per operating unit | compatible when K is pre-registered | only after owner fixes K and unit |
| B. TRAIN normal block percentile | block/run boundaries equivalent to operating unit | not verifiable from current data | upper normal-operation alert volume, e.g. 99th percentile | compatible if blocks and percentile are frozen | only after block definition is available |
| C. `clip(ceil(rate*n_train), min, max)` | train count only | technically possible | research sensitivity only; train count is not review capacity | parameter-only compatible, operationally weak | not primary |

## Consequence

C and D1 are blocked. Code may be prepared, but no primary result will be run or selected until an operating unit and either K or the TRAIN normal block rule are supplied. TEST results must not choose this policy.
