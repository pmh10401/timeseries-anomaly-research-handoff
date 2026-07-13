# Exp137 Detailed Handoff

## 1. Project Objective

The system is a normal-only time-series anomaly detector for operational use. Training data is expected to be almost entirely normal. The immediate operating objective is not to maximize a single mean F1 score: it is to reduce false alarms, automatically report only clear anomalies, and send ambiguous cases to a human review lane.

The user wants the system to be trustworthy in a real equipment-monitoring setting. A detector that always marks some samples as anomalous, even during normal operation, is not acceptable as the default behavior.

## 2. Dataset Contract

- Source DB: `/Users/minho/Documents/Dataset/univariate_ts.db`
- DB verification on 2026-07-13: `PRAGMA integrity_check = ok`
- Catalog: 1,119 datasets, 396,737 instances
- `datasets` table: metadata and train/test normal/anomaly counts
- `instances` table: `TRAIN` and `TEST` series stored as blobs
- Training contract: use TRAIN normal series only for feature fitting, score calibration, and thresholds.
- Evaluation contract: TEST labels are only for offline metrics. They must not control routing, thresholds, selected positions, or family-specific model selection.

The full Exp137 run covers 1,117 datasets. The DB catalog count is 1,119; keep this distinction explicit in reports.

## 3. What Exp137 Is

Exp137 is an operational routing layer, not a new learned model.

It consumes precomputed candidates from earlier experiments and routes each test-series index into one mutually exclusive action:

| Public presentation term | Code-era name | Source | Meaning |
|---|---|---|---|
| 1st detection | Exp93 candidate generation | `run_experiment_93_nonpos_candidate_reranker.py` | Builds anomaly candidates from normal-trained score sources. |
| Cross confirmation | Block B / Exp133 | `run_experiment_133_block_b_confidence_tiers.py` | An independent ROCKET feature partition checks whether it points to the same index. |
| Supplementary confirmation | Block C / Exp135 | `run_experiment_135_block_c_review_confirmation.py` | A third ROCKET feature partition provides narrow additional evidence for a review candidate. |
| Hard alert | Exp133 high-confidence | Exp137 input | Autonomous alert. |
| Standard review | Exp133 standard-confidence | Exp137 input | A candidate exists but lacks enough evidence for automatic alerting. |
| Priority review | Exp135 narrow confirmation | Exp137 input | Review candidate with supplementary confirmation. It is research-only until prospectively validated. |
| No alert | all remaining indices | Exp137 | No operational action. |

## 4. Algorithms Used

The public explanation should name both the role and the algorithm once, without centering the internal Block labels.

1. **1st detection**: ROCKET random convolutional features plus KNN local-gap anomaly score; Exp93 also ranks image-derived candidates such as Spectrogram + PCA + KNN.
2. **Cross confirmation**: another ROCKET feature partition plus KNN local-gap. It must independently identify the same sample index before an alert is promoted to Hard alert.
3. **Supplementary confirmation**: a third ROCKET feature partition plus KNN local-gap. It narrows a small review-only candidate set.

Technical detail:

- `run_experiment_132_block_b_review_integration.py` creates 512 ROCKET kernels, transforms the normal reference and test series, then applies KNN local-gap (`k=3`) separately to two feature halves. The split is called Block A/Block B in code.
- `run_experiment_135_block_c_review_confirmation.py` takes the later 256-kernel slice from a 768-kernel ROCKET construction and scores it with KNN local-gap (`k=3`). This is Block C in code.
- Code labels remain as historical implementation terms. Company-facing material should use 1st detection, cross confirmation, and supplementary confirmation.

## 5. Verified Exp137 Results

Source: `results/experiment_137_operational_triage_summary.csv`.

| Lane | Candidate count | TP | FP | Precision | Operational meaning |
|---|---:|---:|---:|---:|---|
| Hard alert | 2,005 | 1,691 | 314 | 84.339% | Autonomous alert lane |
| Standard review | 639 | 292 | 347 | 45.696% | Human review request |
| Priority review | 9 | 8 | 1 | 88.889% | Small, high-value review request |

Additional summary:

- Autonomous Hard alert mean F1: `0.600772`
- Hard alert zero-F1 datasets: `362`
- Human-assisted union mean F1: `0.700776`
- Human-assisted union zero-F1 datasets: `233`
- Routing uses test labels: `0` rows
- Routing uses test positions: `0` rows
- Routing uses historical family performance: `0` rows

## 6. Metric Scope: Do Not Mix These

`mean_f1` in Exp137 is intentionally the same as `mean_hard_f1`. Its scope is `autonomous_hard_alert`.

`mean_combined_f1` includes Hard alert + Standard review + Priority review and is explicitly marked `human_assisted_diagnostic_only`. It is useful to estimate the value of a user review workflow, but it is not the autonomous model score and must not be presented as one.

## 7. Safety and Validity Guardrails

1. Do not use test labels, known anomaly positions, or historical family-by-family test performance to decide a live route.
2. Do not promote Priority review to autonomous alert. The result CSV marks it as `priority_review_posthoc_research_rule=1` and `prospective_priority_validation_required=1`.
3. Do not reintroduce tail-position rules. Earlier tail replacement experiments were rejected as unfair because they exploit the construction of the evaluation set rather than a train-only signal.
4. Keep normal-only training separate from TEST evaluation.
5. For all new summaries, report Hard alert and review-lane metrics separately.
6. Preserve the distinction between 1,119 DB catalog datasets and the 1,117-dataset Exp137 coverage.

## 8. Core Files and Their Roles

| File | Purpose |
|---|---|
| `core_code/run_experiment_137_operational_triage.py` | Final three-tier routing and summary contract. |
| `core_code/scratch/test_experiment_137_operational_triage.py` | Routing, disjoint-tier, and summary tests. |
| `core_code/run_experiment_132_block_b_review_integration.py` | Cross-confirmation candidate construction. |
| `core_code/run_experiment_133_block_b_confidence_tiers.py` | High/standard confidence tiers used by Exp137. |
| `core_code/run_experiment_134_block_b_review_tail_guard.py` | Historical intermediate review experiment; do not promote its tail rule. |
| `core_code/run_experiment_135_block_c_review_confirmation.py` | Supplementary confirmation for priority review. |
| `core_code/run_experiment_93_nonpos_candidate_reranker.py` | Candidate generation and rank combination. |
| `core_code/run_experiment_119a_exp93_rank_order_validation.py` | Validated, deterministic Exp93 ordering used upstream. |
| `core_code/run_experiment_26_rocket.py` | ROCKET transform primitives. |
| `core_code/run_experiment_40_original_score_normalization_sweep.py` | Train-normal count-cap threshold helper. |
| `core_code/run_balanced_improvement_experiment.py` | KNN local-gap scoring helper. |
| `core_code/run_rank_experiments_sequential.py` | One-at-a-time queue runner. |
| `core_code/serve_rank_dashboard.py` | Read-only dashboard service. |

## 9. Reproduce or Verify Exp137

From the repository root:

```bash
python3 -m unittest scratch/test_experiment_137_operational_triage.py
python3 run_experiment_137_operational_triage.py
```

Expected completion message:

```text
experiment_137_operational_triage finished rows=1117 datasets=1117 errors=0
```

The run reads these upstream result files from `/Users/minho/Documents/Dataset`:

```text
experiment_133_block_b_confidence_tiers_results.csv
experiment_135_block_c_review_confirmation_results.csv
```

Before any long rerun, verify no competing rank experiment remains alive:

```bash
pgrep -af 'run_experiment|run_rank_experiments_sequential'
```

## 10. Dashboard and Sequential Runner

The dashboard is read-only and derives status from CSV summaries, sequential state JSON, and logs. It does not control experiments.

```bash
python3 serve_rank_dashboard.py
```

Then open `http://127.0.0.1:8765`.

Use `run_rank_experiments_sequential.py` when experiments need to run one at a time. Avoid overlapping ROCKET/VAE runs unless resource use has been checked explicitly.

## 11. Recommended Next Work

1. **Prospective validation of review routing**: validate the priority rule on genuinely new equipment/recipe data before any deployment promotion.
2. **Human-review workflow design**: record whether a reviewer confirms or dismisses each review-lane candidate; use that feedback later for train-only calibration research.
3. **Hard-alert calibration**: keep the current 84.3% precision lane as the baseline and investigate only train-normal, label-free ways to reduce the remaining 314 false positives.
4. **Reporting discipline**: retain separate headline metrics for autonomous alerts and human-assisted review outcomes.

## 12. Current Presentation Artifact

`presentation/exp137_operational_triage_draft_20260713.pptx` is the current Korean company-report draft. It starts from the SQLite database, explains the normal-only training contract, shows the complete routing flow, presents tier results, and ends with a conceptual user alert UI.
