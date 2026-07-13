# Research Handoff Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a public, data-safe research handoff repository that gives GPT and human reviewers enough context to inspect the anomaly-detection experiments without the original database.

**Architecture:** A standalone Git repository mirrors code and prose into stable directories, extracts dataset-level SQLite metadata into small public files, and copies only reviewable results. A generated manifest binds every public artifact to a source path and checksum while recording excluded local-only artifacts.

**Tech Stack:** Git, GitHub CLI, Python 3 standard library, SQLite, SHA256, rsync.

## Global Constraints

- Never copy `univariate_ts.db`, raw time-series blobs, virtual environments, credentials, or files over 90 MB.
- Preserve source filenames where possible so result references remain understandable.
- Results remain retrospective; do not use test labels for new policy selection.
- Do not alter the original `/Users/minho/Documents/timesries project` worktree.

### Task 1: Build the data-safe file set

**Files:** Create `scripts/*.py`, `tests/*.py`, `docs/source/`, `results/`, `artifacts/`, and `.gitignore`.

- [ ] Copy root Python scripts and available test scripts, excluding bytecode and virtual environments.
- [ ] Copy all Markdown/Mermaid documentation plus reviewable CSV/JSON/figure artifacts, excluding duplicate archives and logs.
- [ ] Enforce the deny-list: DBs, raw data, checkpoints, JSONL resume files, environments, and files over 90 MB.

### Task 2: Produce a database understanding package

**Files:** Create `data/DATABASE_GUIDE.md`, `data/schema.sql`, `data/dataset_catalog.csv`, and `data/dataset_summary.json`.

**Interfaces:** Consumes `/Users/minho/Documents/Dataset/univariate_ts.db`; produces metadata-only files that never contain `values_blob` or `labels_blob`.

- [ ] Export the full `datasets` table ordered by name.
- [ ] Record schema, aggregate counts, evaluation exclusions, normal-only training policy, and limits on industrial interpretation.

### Task 3: Add entry points and traceability

**Files:** Create `README.md`, `MANIFEST.csv`, `MANIFEST.md`, and `REPRODUCIBILITY.md`.

- [ ] Write a plain-language research map, operational objective, Exp137 evidence, and GPT handoff steps.
- [ ] Generate SHA256 entries for each included source/result artifact and record excluded classes.

### Task 4: Validate and publish

- [ ] Scan staged text for credential values and confirm every large file remains below the GitHub limit.
- [ ] Run source compilation, selected tests, `git fsck --full`, and a clean staged-file review.
- [ ] Create and push public repository `pmh10401/timeseries-anomaly-research-handoff`.
