# Exp101/Exp102 Feature Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test shapelet normal prototype signals and combine feature sources without promoting unsafe hard-alert replacement.

**Architecture:** Exp101 creates shapelet-derived candidate alerts using train-normal prototypes and records research-only versus train/family gating. Exp102 combines Exp100 spectral review signals and Exp101 shapelet signals into a conservative review-lane feature source selector while keeping Exp93 as the hard-alert baseline.

**Tech Stack:** Python CSV experiment scripts, NumPy/scikit-learn helpers, existing rank experiment runner and dashboard.

---

### Task 1: Exp101 Shapelet Normal Prototype Feature Score

**Files:**
- Create: `run_experiment_101_shapelet_normal_prototype.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`

- [ ] Create a shapelet normal prototype scorer that loads Exp93 operating rows, identifies Exp97 shapelet diagnostic targets and train/family shape families, and emits baseline plus shapelet selector rows.
- [ ] Use train-normal-only shapelet prototypes, robust deviation scores, q99 cap1 and q98 cap2 candidate policies.
- [ ] Compile and run the full experiment through the sequential runner.

### Task 2: Exp102 Feature Source Selector

**Files:**
- Create: `run_experiment_102_feature_source_selector.py`
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`

- [ ] Combine Exp93, Exp95, Exp100, and Exp101 outputs.
- [ ] Keep Exp93 hard alerts unchanged for the main operating-safe selectors.
- [ ] Add spectral and shapelet candidates only when Exp93 is weak and the candidate source agrees with existing context.
- [ ] Include hard-alert guard variants only as diagnostic rows, not as recommended defaults.
- [ ] Compile and run the full experiment through the sequential runner.

### Task 3: Reporting

**Files:**
- Create: `docs/experiments/exp101_shapelet_normal_prototype.md`
- Create: `docs/experiments/exp102_feature_source_selector.md`
- Modify: `docs/task.md`

- [ ] Summarize which selectors improved zero-F1, which added review burden, and whether any hard-alert replacement was clean enough.
- [ ] Record the operating decision in `docs/task.md`.
