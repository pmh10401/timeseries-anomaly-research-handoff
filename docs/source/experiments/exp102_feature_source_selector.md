# Exp102 Feature Source Selector

Date: 2026-07-09 KST

## Purpose

Exp102 combined the current operating baseline with the new feature-source
signals from Exp99/100 and Exp101.

The goal was to decide whether spectral and shapelet sources should be:

- promoted to hard alerts,
- attached as review candidates,
- or rejected as too noisy.

## Results

| Selector | Type | Combined F1 | Combined zero-F1 | Review / 100 datasets | Review TP | Review FP | Hard replacements |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_exp93_hard_only` | hard baseline | `0.697575` | `239` | `0.00` | `0` | `0` | `0` |
| `review_spectral_shapelet_weak_agreement` | operating review | `0.698470` | `237` | `1.25` | `2` | `12` | `0` |
| `hard_guard_single_feature_source_when_exp93_weak` | hard diagnostic | `0.698470` | `237` | `0.00` | `0` | `0` | `11` |
| `review_existing_top1_only` | broad review baseline | `0.596675` | `230` | `51.84` | `23` | `556` | `0` |
| `review_existing_plus_feature_sources_cap3` | broad review diagnostic | `0.597421` | `228` | `53.00` | `25` | `567` | `0` |
| `review_research_spectral_shapelet_upper_bound` | research-only | `0.700708` | `233` | `10.92` | `6` | `116` | `0` |

Movement versus Exp93:

| Selector | Improved | Worsened | Zero-F1 fixed | New zero-F1 | Review TP | Review FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `review_spectral_shapelet_weak_agreement` | `2` | `1` | `2` | `0` | `2` | `12` |
| `hard_guard_single_feature_source_when_exp93_weak` | `2` | `1` | `2` | `0` | `0` | `0` |
| `review_research_spectral_shapelet_upper_bound` | `6` | `0` | `6` | `0` | `6` | `116` |

## Interpretation

The research-only upper bound confirms there are still a few recoverable cases:

- spectral contributes Phoneme/CricketZ-style recoveries
- shapelet contributes ProximalPhalanxTW/SwedishLeaf-style recoveries

But the operational selector is still modest. It adds only `14` review
candidates across `1117` datasets and finds `2` true hits.

The hard-alert diagnostic is not clean enough. It improves `2` datasets but
worsens `1`, so it should not replace Exp93 as the operating default.

## Decision

Keep:

- Exp93 as hard-alert default.
- Exp96/Exp100-style review lane as the operating path.
- `review_spectral_shapelet_weak_agreement` as a candidate low-burden review
  addition.

Do not promote:

- broad existing+feature review as default, because review FP is high.
- hard feature-source replacement, because it still has a regression.

## Next Step

The remaining improvement path is not broader feature activation. It should be
better review prioritization:

- rank review candidates by source reliability
- show source labels and confidence instead of treating review candidates as
  hard positives
- add family-specific review budgets for the small set of recoverable families

## Output Files

- `/Users/minho/Documents/Dataset/experiment_102_feature_source_selector_results.csv`
- `/Users/minho/Documents/Dataset/experiment_102_feature_source_selector_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_102_feature_source_selector_stdout.log`
