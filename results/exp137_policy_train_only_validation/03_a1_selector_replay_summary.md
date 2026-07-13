# A1 Selector Replay Summary

- Dataset coverage: 1117
- All-lane matches: 1117/1117
- Mismatched datasets: 0
- Hard mismatches: 0
- Standard mismatches: 0
- Priority mismatches: 0
- Replay command: `python3 run_experiment_138_policy_train_only_audit.py`
- Replay code: `run_experiment_138_policy_train_only_audit.py`, `run_experiment_137_operational_triage.py`
- Commit: `05c471fd650f34ca641944d7524f6484a84f4fb2`
- Input/output SHA256: see `06_evaluation_contract_v2.json` and `MANIFEST.csv`.
- Seed: no new model fit; A1 replays frozen Exp133/Exp135 selector inputs.
- Environment: Python 3.14.6

A1 is a final-lane selector replay. It does not reproduce upstream feature extraction or score generation.
