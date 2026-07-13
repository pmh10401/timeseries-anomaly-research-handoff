# Exp137 GPT Handoff Package

This package is a focused continuation bundle for the time-series anomaly-detection project.

## Start Here

1. Read `HANDOFF_EXP137_DETAILED_KO.md` for the project objective, verified results, guardrails, and next decisions.
2. Read `GPT_CONTINUATION_PROMPT_KO.md` and use it as the first prompt in a new GPT conversation.
3. Read `EXPERIMENT_137_FLOW.md` for the plain-language and code-name mapping of the anomaly-routing flow.
4. Use `DATASET_DB_REFERENCE.md` before opening the dataset database.

## Package Contents

- `core_code/`: Exp137 routing, upstream Exp93/133/135 code, score utilities, sequential runner, dashboard, and the Exp137 test.
- `results/`: Exact Exp133, Exp135, and Exp137 CSV inputs/outputs used for the Exp137 routing result.
- `presentation/`: The current Korean company-report presentation draft.
- `*.md`: Detailed handoff, ready-to-paste GPT prompt, flow explanation, database reference, and package manifest.

## Important Paths On This Mac

- Repository: `/Users/minho/Documents/timesries project`
- Dataset/results directory: `/Users/minho/Documents/Dataset`
- SQLite database: `/Users/minho/Documents/Dataset/univariate_ts.db`
- Dashboard: `http://127.0.0.1:8765` when `serve_rank_dashboard.py` is running

## Deliberate Exclusion

`univariate_ts.db` is about 1.6 GB, so it is not copied into this archive. The database remains at the absolute path above. Its schema, counts, integrity check, and verification commands are documented in `DATASET_DB_REFERENCE.md`.
