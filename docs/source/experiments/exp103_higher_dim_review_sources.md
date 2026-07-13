# Experiment 103: Higher-Dim Review Sources

## Purpose

Exp99-102 showed that spectral, image, and shapelet-like features can find a few missed anomalies, but hard-alert replacement is still risky. Exp103 tested whether those feature sources become more useful when their fitted feature space is widened and used only as a review lane.

The operating rule remains:

- Exp93 is the hard-alert baseline.
- Higher-dim sources may add review candidates.
- Review candidates are not automatic alerts.

## Method

Baseline:

- `baseline_exp93_hard_only`

Review sources:

- `spectrogram_pca64`
- `spectrogram_pca128`
- `glcm_rp_pca64`
- `rocket_512_local_gap`

Selectors:

- `review_pca64_sources_when_exp93_weak`
- `review_pca64_pca128_sources_when_exp93_weak`
- `review_all_higher_dim_sources_when_exp93_weak`
- `hard_guard_single_higher_dim_source_when_exp93_weak`

The review selectors only add candidates when Exp93 looks weak and the candidate has agreement/support from the feature-source pool.

## Results

| Selector | Hard Mean F1 | Hard Zero-F1 | Combined Mean F1 | Combined Zero-F1 | Review Candidates | Review TP | Review FP | Review Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_exp93_hard_only | 0.697575 | 239 | 0.697575 | 239 | 0 | 0 | 0 | 0.000 |
| review_pca64_sources_when_exp93_weak | 0.697575 | 239 | 0.716644 | 191 | 162 | 48 | 114 | 0.296 |
| review_pca64_pca128_sources_when_exp93_weak | 0.697575 | 239 | 0.716644 | 191 | 162 | 48 | 114 | 0.296 |
| review_all_higher_dim_sources_when_exp93_weak | 0.697575 | 239 | 0.716644 | 191 | 162 | 48 | 114 | 0.296 |
| hard_guard_single_higher_dim_source_when_exp93_weak | 0.697575 | 239 | 0.697575 | 239 | 0 | 0 | 0 | 0.000 |

Compared with Exp93, the best review selector:

- fixed `48` zero-F1 datasets in the combined review metric
- added `162` review candidates across `1117` datasets
- found `48` true anomaly review hits
- added `114` false review candidates
- did not change hard alerts

Compared with Exp102's best review selector:

- Exp102: `14` review candidates, `2` TP, `12` FP
- Exp103: `162` review candidates, `48` TP, `114` FP

So Exp103 is much stronger as a review-lane generator. It finds more useful review targets, and review precision also improves.

## Interpretation

Higher-dimensional feature sources did not improve automatic hard-alert decisions in this design, but they did improve the ability to surface likely missed anomalies for review.

That distinction matters operationally:

- Good: use Exp103 to prioritize samples for human review.
- Risky: do not promote Exp103 review candidates directly to hard alerts.

The combined F1 improvement assumes review candidates are inspected or accepted in a review workflow. It should not be read as automatic alert F1.

## Decision

Recommended next operating posture:

- Keep Exp93 as the hard-alert default.
- Use `review_all_higher_dim_sources_when_exp93_weak` as the strongest review-lane candidate so far.
- Keep `hard_guard_single_higher_dim_source_when_exp93_weak` diagnostic only, because it produced no hard-alert improvement.

## Dashboard Note

While Exp103 was running, the dashboard could show stale or inflated live progress because completed CSV rows were allowed to override stdout progress. The dashboard was adjusted so that live progress and current dataset are shown only when a real running process exists.
