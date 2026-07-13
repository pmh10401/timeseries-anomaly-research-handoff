# Experiment 107: Exp103 and Combo Disagreement

## Purpose

Exp107 checks whether Exp106 combo candidates provide unique value beyond Exp103.

Question:

> Are there cases where Exp103 misses but the gated combo score finds a true anomaly?

## Method

Compared review candidates from:

- Exp103 `review_all_higher_dim_sources_when_exp93_weak`
- Exp106 `review_combo_conservative_when_exp93_weak`
- Exp106 `review_combo_sensitive_when_exp93_weak`

Then evaluated:

- Exp103 only
- combo-only candidates
- combo candidates that are not already in Exp103
- Exp103 plus combo-only candidates

## Results

| Selector | Combined Mean F1 | Combined Zero-F1 | Review Candidates | Review TP | Review FP | Review Precision |
|---|---:|---:|---:|---:|---:|---:|
| `review_exp103_plus_unique_sensitive_cap4` | 0.716823 | 189 | 179 | 50 | 129 | 0.279 |
| `review_exp103_only` | 0.716644 | 191 | 162 | 48 | 114 | 0.296 |
| `review_exp103_plus_unique_conservative_cap3` | 0.716644 | 191 | 162 | 48 | 114 | 0.296 |
| `review_combo_sensitive_only` | 0.715570 | 194 | 165 | 45 | 120 | 0.273 |
| `review_combo_conservative_only` | 0.714137 | 198 | 150 | 41 | 109 | 0.273 |
| `review_unique_sensitive_not_exp103` | 0.698769 | 237 | 33 | 2 | 31 | 0.061 |
| `review_unique_conservative_not_exp103` | 0.697575 | 239 | 27 | 0 | 27 | 0.000 |

## Unique Wins

The sensitive combo source found two true positives that Exp103 missed:

- `WordSynonyms_normal_1`
- `WordSynonyms_normal_15`

But the unique sensitive candidates also added:

- `33` total unique review candidates
- `2` true positives
- `31` false positives

When added back to Exp103:

- combined zero-F1 improves from `191` to `189`
- review TP improves from `48` to `50`
- review FP worsens from `114` to `129`
- review precision drops from `29.6%` to `27.9%`

## Interpretation

The combo score has a small unique signal, but it is weak and noisy.

The conservative combo adds no unique true positives beyond Exp103.

The sensitive combo adds two useful WordSynonyms recoveries, but the cost is high: many extra false review candidates and some combined-F1 regressions.

## Decision

Do not broadly add combo-only candidates to the default review lane.

Recommended next step:

- Investigate the two WordSynonyms wins.
- Build a very narrow family/local-condition guard only if the evidence is operationally justifiable without using benchmark labels as a production prior.
- Otherwise keep Exp103 as the current best review lane.
