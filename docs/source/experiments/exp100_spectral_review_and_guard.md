# Exp100 Spectral Review and Guard

Date: 2026-07-09 KST

## Purpose

Exp99 showed that spectral/derivative features can rescue a few zero-F1 cases,
but broad hard-alert replacement is unsafe.

Exp100 tested two safer directions:

1. Add spectral candidates only to the review lane.
2. Use spectral hard-alert replacement only when Exp93 looks weak and a
   confidence/agreement guard also agrees.

## Tested Selectors

| Selector | Type | Meaning |
| --- | --- | --- |
| `baseline_exp93_hard_only` | hard alert | Current Exp93 operating default |
| `review_spectral_research_only_q98_cap2` | review, research-only | Add Exp99 spectral candidates only for Exp97 spectral diagnostic rows |
| `review_spectral_family_q98_cap2` | review | Add spectral candidates for broad spectral families |
| `review_spectral_family_when_exp93_weak_agrees` | review | Add spectral candidates only when Exp93 is weak and spectral agrees with existing context |
| `hard_guard_research_spectral_when_exp93_weak` | hard alert, research-only | Replace hard alert with spectral only for Exp97 spectral diagnostic rows and weak Exp93 |
| `hard_guard_spectral_agreement_when_exp93_weak` | hard alert | Replace hard alert only when Exp93 is weak and spectral agrees with existing context |

Research-only selectors are useful for diagnosing possible upside, but they
should not be promoted to production because the target membership comes from
previous test-outcome analysis.

## Results

| Selector | Mean F1 | Zero-F1 | Combined F1 | Combined Zero-F1 | Review / 100 datasets | Review TP | Review FP | Hard replacements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_exp93_hard_only` | `0.697575` | `239` | `0.697575` | `239` | `0.00` | `0` | `0` | `0` |
| `review_spectral_family_when_exp93_weak_agrees` | `0.697575` | `239` | `0.698470` | `237` | `1.07` | `2` | `10` | `0` |
| `hard_guard_spectral_agreement_when_exp93_weak` | `0.698470` | `237` | `0.698470` | `237` | `0.00` | `0` | `0` | `10` |
| `review_spectral_family_q98_cap2` | `0.697575` | `239` | `0.657796` | `233` | `31.42` | `11` | `340` | `0` |
| `review_existing_top1_only` | `0.697575` | `239` | `0.596675` | `230` | `51.84` | `23` | `556` | `0` |
| `review_existing_top1_plus_spectral_family_cap3` | `0.697575` | `239` | `0.564094` | `225` | `82.63` | `33` | `890` | `0` |
| `review_spectral_research_only_q98_cap2` | `0.697575` | `239` | `0.699515` | `235` | `5.01` | `4` | `52` | `0` |
| `hard_guard_research_spectral_when_exp93_weak` | `0.699962` | `235` | `0.699962` | `235` | `0.00` | `0` | `0` | `14` |

## Interpretation

The safest operational signal is:

`review_spectral_family_when_exp93_weak_agrees`

It does not change hard alerts. It adds only `12` review candidates across
`1117` datasets, and `2` of them are true anomaly hits.

This is small but useful because the review burden is low:

- about `1.07` extra review candidates per `100` datasets
- `2` zero-F1 cases rescued in review view
- no hard-alert replacement risk

The hard-alert guarded selector:

`hard_guard_spectral_agreement_when_exp93_weak`

improved `2` datasets but worsened `1` dataset. That is not clean enough for an
operating default yet. The worsened case shows that even when spectral agrees
with existing context, replacement can still move a previously correct alert to
a weaker two-index alert.

Broad spectral review is also not recommended. It finds more true anomalies,
but it adds too many false review candidates:

- `review_spectral_family_q98_cap2`: `11` TP, `340` FP
- `existing_top1_plus_spectral`: `33` TP, `890` FP

## Decision

Keep Exp93 as the hard-alert default.

Promote only this as a candidate review-lane addition:

`review_spectral_family_when_exp93_weak_agrees`

Do not promote spectral hard-alert replacement yet.

## Next Step

The next improvement should refine the review guard, not broaden spectral use.

Good next candidates:

- require spectral candidate count `1` instead of `2`
- require agreement with both Exp93 hard alert and existing review candidate
- show spectral candidates only as low-priority review annotations
- add a per-family review budget for Phoneme, CricketZ, and GestureMidAirD3

## Output Files

- `/Users/minho/Documents/Dataset/experiment_100_spectral_review_and_guard_results.csv`
- `/Users/minho/Documents/Dataset/experiment_100_spectral_review_and_guard_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_100_spectral_review_and_guard_stdout.log`
