# Experiment 113: Train-Normal Conformal Fusion

Date: 2026-07-10 KST

## Purpose

Test a direct replacement for Exp93 without labels, family-performance priors,
test-position rules, or test-count caps.

The three all-dataset sources were retained:

- ROCKET Exp40 local-gap score
- Exp55 spectrogram score
- Exp56 GLCM/RP score

Each source was converted to a leave-self-out empirical train-normal tail
p-value. The test p-values were combined using Bonferroni min-p, Cauchy,
Fisher, or 2-of-3 agreement. Alerts used nominal `0.5%` and `1%` p-value
targets.

## Coverage

- Datasets: `1117 / 1117`
- Detail rows: `10053 / 10053`
- Runtime errors: `0`

## Main Results

| Candidate | Nominal target | Mean F1 | Median F1 | Zero-F1 | Mean FP | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp93 baseline | selector | 0.6976 | 1.0000 | 239 | 0.591 | 0.750 | 0.506 |
| Fisher fusion | 1% | 0.3753 | 0.0000 | 572 | 1.045 | 0.638 | 0.526 |
| Fisher fusion | 0.5% | 0.2705 | 0.0000 | 725 | 0.639 | 0.708 | 0.442 |
| Cauchy fusion | 1% | 0.0815 | 0.0000 | 967 | 0.245 | 0.810 | 0.299 |
| 2-of-3 fusion | 1% | 0.0729 | 0.0000 | 976 | 0.120 | 0.860 | 0.210 |
| Bonferroni min-p | 1% | 0.0433 | 0.0000 | 1025 | 0.206 | 0.798 | 0.233 |

## What Happened

### 1. Small normal sets cannot resolve a 1% tail

With `n` train-normal samples, the smallest leave-self-out empirical p-value
is approximately `1 / (n + 1)`. A source with fewer than about `99` normals
cannot express a p-value below `1%` from empirical ranks alone.

This was decisive in the current dataset:

- `317` datasets have `<=10` train normals.
- Fisher 1% emitted no alert on all `317` of them and had mean F1 `0.000`.
- Exp93 had mean F1 `0.823` in the same group because its candidate policy does
  not require estimating a 1% absolute tail from a tiny reference set.

The strict conformal candidates had high precision because they abstained, not
because they replaced Exp93's detection capability.

### 2. Fisher recovered sensitivity at an unacceptable operating cost

Fisher 1% was the strongest direct replacement. It recovered `45` Exp93
zero-F1 rows and slightly exceeded Exp93 recall (`0.526` versus `0.506`).
However, it also:

- created `378` new zero-F1 rows,
- worsened `550` datasets,
- turned `338` Exp93-perfect rows into F1 `0`,
- increased mean FP from `0.591` to `1.045`.

Fisher assumes source p-values are independent. ROCKET and the two imaging
scores are correlated views of the same sequence, so this combination is a
research diagnostic, not a calibrated operating probability.

### 3. The result distinguishes abstention from detection

Exp93 emitted at least one bounded candidate on every dataset. Cauchy,
Bonferroni, and 2-of-3 emitted no alert on roughly `85%` to `95%` of datasets.
Their low FP and high precision therefore come mainly from abstention. That is
potentially useful as a *high-confidence review signal*, but it is not a
replacement for an anomaly detector in the present evaluation.

### 4. Large-train Fisher signal is still not enough

For `301-1000` normal samples, Fisher 1% reached mean F1 `0.569`, close to
Exp93's `0.565` in that bin, but mean FP was `6.47` versus Exp93's `1.82`.
Even where empirical tail resolution is adequate, the direct fusion does not
meet the false-alarm objective.

### 5. Conformal evidence is useful as an Exp93 confidence annotation

The direct-replacement result should not discard the p-value signal entirely.
At the `1%` target, when a conformal alert selected the *same test index* as an
existing Exp93 alert, the observed precision improved materially:

| Confirmation source | Shared Exp93 alerts | Precision of shared alerts | Precision of Exp93-only alerts |
| --- | ---: | ---: | ---: |
| Bonferroni min-p | 784 | 0.886 | 0.693 |
| 2-of-3 agreement | 949 | 0.865 | 0.686 |
| Cauchy | 1078 | 0.876 | 0.664 |
| Fisher | 1948 | 0.790 | 0.641 |

This is an offline evaluation result, not a deployment rule. The valid lesson
is narrow: use Bonferroni or 2-of-3 agreement to mark an **already existing
Exp93 alert** as high-confidence or to rank the review queue. Do not add their
candidate-only indices as hard alerts.

## Decision

Reject all Exp113 variants as direct Exp93 replacements.

The experiment does not show that train-normal calibration is useless. It
shows that an absolute p-value threshold cannot replace Exp93's candidate
generation when normal references are small and source scores are correlated.

The next valid improvement should separate two questions:

1. **Candidate generation:** improve the score representation with a
   train-only method, such as generic pseudo-anomaly training or a more local
   normal-state score.
2. **Confidence calibration:** use conformal evidence only as an alert
   confidence/review annotation on an existing candidate set, not as the sole
   rule that decides whether an alert exists.

## Output Files

- `/Users/minho/Documents/Dataset/experiment_113_train_normal_conformal_fusion_results.csv`
- `/Users/minho/Documents/Dataset/experiment_113_train_normal_conformal_fusion_summary.csv`
- `/Users/minho/Documents/Dataset/experiment_113_train_normal_conformal_fusion_stdout.log`
