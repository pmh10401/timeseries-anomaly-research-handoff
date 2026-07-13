# Experiment 115: Local Normal-State Score

Date: 2026-07-10 KST

## Purpose

Test whether a test sequence should be compared with its nearest normal
operating state instead of the full normal population.

All score candidates use the same 256-kernel ROCKET representation (`512`
features). Datasets with fewer than `30` normal train samples fall back to the
global ROCKET local-gap score. Datasets with at least `30` normals test:

- cross-fitted diagonal GMM, BIC up to 3 components,
- cross-fitted diagonal GMM, BIC up to 5 components,
- cross-fitted KMeans with 3 states.

The count-cap thresholds are diagnostic only. AUC-PR and oracle F1 evaluate
the score representation separately from alert policy.

## Coverage

- Datasets: `1117 / 1117`
- Detail rows: `14521 / 14521`
- Runtime errors: `0`
- Global-fallback datasets (`n < 30`): `522`
- Active local-state datasets (`n >= 30`): `595`

## Overall Results

| Candidate | Diagnostic threshold | Mean F1 | Zero-F1 | Mean FP | Mean AUC-PR | Mean Oracle F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Exp93 operating baseline | selector | 0.6976 | 239 | 0.591 | 0.760 | 0.837 |
| Global ROCKET local-gap | 3% | 0.5818 | 259 | 1.391 | 0.767 | 0.840 |
| GMM BIC up to 5 | 3% | 0.5120 | 327 | 3.217 | 0.701 | 0.785 |
| GMM BIC up to 3 | 3% | 0.5103 | 322 | 3.599 | 0.690 | 0.771 |
| KMeans 3 states | 3% | 0.4981 | 298 | 4.281 | 0.659 | 0.746 |

Exp93 remains stronger because its selector uses multiple representations and
bounded candidate logic. The relevant score comparison is global local-gap
against the local-state variants: global local-gap has the best AUC-PR and
oracle F1.

## Active Local-State Subset (`n >= 30`)

At the 3% diagnostic threshold:

| Candidate | Mean F1 | Zero-F1 | Mean FP | Mean AUC-PR | Mean Oracle F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global ROCKET local-gap | 0.5512 | 140 | 2.062 | 0.696 | 0.775 |
| GMM BIC up to 5 | 0.4203 | 208 | 5.489 | 0.573 | 0.671 |
| GMM BIC up to 3 | 0.4170 | 203 | 6.207 | 0.552 | 0.646 |
| KMeans 3 states | 0.3941 | 179 | 7.487 | 0.493 | 0.598 |

GMM BIC up to 5 versus global local-gap:

- Improved datasets: `87`
- Worsened datasets: `266`
- Zero-F1 fixed: `10`
- New zero-F1: `78`
- Mean AUC-PR delta: `-0.123`
- Mean oracle F1 delta: `-0.104`

## Why It Failed

This is not primarily a small-train failure. The result remains negative in
every active train-size bin, including `>=1000` normals:

| Train normals | Global AUC-PR | GMM5 AUC-PR | Global FP | GMM5 FP |
| --- | ---: | ---: | ---: | ---: |
| 30-49 | 0.722 | 0.620 | 0.57 | 0.34 |
| 100-199 | 0.730 | 0.594 | 1.01 | 1.35 |
| 500-999 | 0.662 | 0.498 | 6.14 | 22.30 |
| >=1000 | 0.647 | 0.367 | 13.40 | 57.76 |

The likely mechanism is feature-space fragmentation:

1. The ROCKET representation has `512` features.
2. Cross-fitting reduces each GMM fit to roughly 80% of the normal set.
3. Splitting that data into local normal states makes each density estimate
   less stable than the global nearest-normal geometry.
4. Hard KMeans state assignment is worst at state boundaries; diagonal GMM is
   softer but still mis-ranks normal variation as rare density.

The score ranking itself falls, so changing only the threshold would not fix
this experiment.

## Decision

Reject the GMM and KMeans local normal-state scores as Exp93 or global-local-
gap replacements. Do not spend another full run on a simple cluster-count or
covariance sweep.

The normal-state idea can be revisited only after a different representation
or a strongly reduced, cluster-stable embedding is independently shown to
improve ranking. The next score-improvement effort should move to Exp114's
generic pseudo-anomaly representation probe rather than another threshold or
local-cluster sweep.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_115_local_normal_state_score_results.csv`
- `/Users/minho/Documents/Dataset/experiment_115_local_normal_state_score_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_115_local_normal_state_score_stdout.log`
