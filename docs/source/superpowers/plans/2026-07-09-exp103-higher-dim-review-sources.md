# Exp103 Higher-Dim Review Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether increasing feature dimensions improves review-candidate quality without replacing the Exp93 hard-alert default.

**Architecture:** Build Exp103 as a feature-source probe that recomputes selected higher-dimensional sources, records hard-alert diagnostic rows, and evaluates review-only selectors. Exp93 remains the hard-alert baseline; higher-dimensional sources are promoted only to review candidates unless a diagnostic variant proves perfectly regression-free.

**Tech Stack:** Python experiment scripts, NumPy, scikit-learn PCA/KNN/scalers, existing dataset loaders, existing sequential runner and dashboard.

---

## Design Decision

Exp99-Exp102 showed the same pattern:

- new features can rescue a few zero-F1 cases,
- broad hard-alert replacement creates regressions,
- low-burden review candidates are safer than automatic replacement.

Therefore Exp103 should not ask:

> Can PCA64/PCA128 replace Exp93?

It should ask:

> Can higher-dimensional feature sources produce better review candidates than the current low-dimensional sources, with small review load?

## Candidate Sources

### Include First

1. `spectrogram_pca64`
   - Based on Exp55 style.
   - Reason: most direct test of “PCA 32 may be over-compressing”.

2. `spectrogram_pca128`
   - Same source, stronger information retention.
   - Reason: checks whether PCA64 is enough or if more dimensions help.

3. `glcm_rp_pca64`
   - Based on Exp56 style.
   - Reason: tests whether texture/recurrence information benefits from less compression.

4. `rocket_512_local_gap`
   - Based on Exp40 style.
   - Reason: checks whether more ROCKET kernels improve candidate ranking.

### Defer Unless Needed

5. `exp84_less_pruned_review`
   - Reason: Exp84/MultiROCKET pruning changes are more complex and may require aeon feature extractor runtime. Do this after the cheaper PCA/ROCKET probes.

## Success Metrics

Primary operating-safe metrics:

- review candidates per 100 datasets
- review TP total
- review FP total
- review precision
- zero-F1 rescued by review
- no hard-alert change for recommended selector

Diagnostic metrics:

- hard replacement improved/worsened counts
- new zero-F1 count
- total FP delta
- family-level concentration of wins/losses

Promotion rule:

- A selector can be recommended only if hard alerts remain unchanged.
- Hard replacement can be considered only if improved > 0 and worsened = 0.

## Task 1: Exp103 Higher-Dim Feature Probe

**Files:**
- Create: `run_experiment_103_higher_dim_review_sources.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`

- [ ] **Step 1: Create source computation helpers**

Implement helpers in `run_experiment_103_higher_dim_review_sources.py`:

```python
def compute_spectrogram_pca_source(dataset_name, pca_dim):
    # Load train/test with load_original_record and target_len_for_record.
    # Use the same imaging path as Exp55 where possible.
    # Fit PCA on train features only.
    # Return train_scores, test_scores, selected_indices.
```

```python
def compute_glcm_rp_pca_source(dataset_name, pca_dim):
    # Use the Exp56 RP/GLCM path where possible.
    # Fit PCA on train features only.
    # Return train_scores, test_scores, selected_indices.
```

```python
def compute_rocket_source(dataset_name, num_kernels):
    # Use Exp40 score_pair_for_config with a copied config and larger kernel count.
    # Return train_scores, test_scores, selected_indices.
```

- [ ] **Step 2: Add selectors**

Implement these selectors:

```text
baseline_exp93_hard_only
review_pca64_sources_when_exp93_weak
review_pca64_pca128_sources_when_exp93_weak
review_all_higher_dim_sources_when_exp93_weak
hard_guard_single_higher_dim_source_when_exp93_weak
```

Rules:

- `baseline_exp93_hard_only`: copy Exp93 hard alert.
- `review_*`: keep Exp93 hard alert; add source candidates only to review lane.
- `hard_guard_*`: diagnostic only; replace hard alert only when exactly one source is active and it agrees with Exp93 context.

- [ ] **Step 3: Register experiment**

Add to `EXPERIMENTS` in `run_rank_experiments_sequential.py`:

```python
{
    "id": "experiment_103_higher_dim_review_sources",
    "script": ROOT / "run_experiment_103_higher_dim_review_sources.py",
    "detail_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_results.csv",
    "summary_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_summary.csv",
    "stdout_log": DATA_DIR / "experiment_103_higher_dim_review_sources_stdout.log",
}
```

Add dashboard entry in `serve_rank_dashboard.py`:

```python
"experiment_103_higher_dim_review_sources": {
    "label": "Experiment 103 · Higher-dim review sources",
    "detail_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_results.csv",
    "summary_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_summary.csv",
    "stdout_log": DATA_DIR / "experiment_103_higher_dim_review_sources_stdout.log",
    "expected_datasets": 1117,
},
```

- [ ] **Step 4: Verify syntax**

Run:

```bash
/opt/homebrew/bin/python3 -m py_compile run_experiment_103_higher_dim_review_sources.py run_rank_experiments_sequential.py serve_rank_dashboard.py
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Smoke test**

Run:

```bash
EXP103_WORKERS=2 /opt/homebrew/bin/python3 run_experiment_103_higher_dim_review_sources.py --dataset-limit 100
```

Expected:

- finishes without exception
- writes `experiment_103_higher_dim_review_sources_results.csv`
- writes `experiment_103_higher_dim_review_sources_summary.csv`
- summary contains baseline plus review selectors

- [ ] **Step 6: Full run**

Run:

```bash
/opt/homebrew/bin/python3 run_rank_experiments_sequential.py queue experiment_103_higher_dim_review_sources
EXP103_WORKERS=4 /opt/homebrew/bin/python3 run_rank_experiments_sequential.py run experiment_103_higher_dim_review_sources
```

Expected:

- runner marks experiment complete
- dashboard can read result and summary

## Task 2: Exp103 Analysis Report

**Files:**
- Create: `docs/experiments/exp103_higher_dim_review_sources.md`
- Modify: `docs/task.md`

- [ ] **Step 1: Summarize source-level results**

Include table:

```text
selector_name
mean_combined_f1
combined_zero_f1_count
review_candidates_per_100_datasets
review_tp_total
review_fp_total
review_precision
hard_replaced_count
```

- [ ] **Step 2: Compare with Exp100/102**

Answer:

- Did higher dimensions find more true review hits than Exp102?
- Did they add too many review false positives?
- Did hard replacement become clean enough?

- [ ] **Step 3: Record operating decision**

Update `docs/task.md` with:

- whether any Exp103 selector should join the review lane,
- whether hard replacement remains rejected,
- what source should be tested next.

## Expected Outcomes

Likely outcomes:

1. PCA64 improves review recall a little.
2. PCA128 may increase review FP more than useful TP.
3. GLCM/RP PCA64 may help shape/texture families but could be noisy.
4. ROCKET512 may improve ranking but could duplicate existing ROCKET256.

Expected operating decision:

- keep Exp93 as hard default,
- accept only low-burden review additions,
- reject any source that adds broad review FP without clear zero-F1 rescue.

## Stop Conditions

Stop and inspect before full run if smoke test shows:

- > `20` review candidates per `100` datasets for the narrow selector,
- missing dataset coverage,
- any selector accidentally changes baseline hard alerts,
- hard diagnostic has new zero-F1 regressions on smoke data.
