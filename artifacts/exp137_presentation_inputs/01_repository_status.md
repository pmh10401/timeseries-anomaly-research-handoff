# Repository and Reproducibility Audit

## Captured State

- Branch: `main...origin/main [ahead 5]`
- Commit: `05c471fd650f34ca641944d7524f6484a84f4fb2`
- Exact command outputs:
  - `audit_sources/git_status_raw.txt` for `git status --short --branch`
  - `audit_sources/git_head_raw.txt` for `git rev-parse HEAD`
  - `02_unit_test_full_log.txt` for the full Exp137 unit-test output
  - `03_environment_versions.txt` for Python and analysis-library versions

## Working Tree

The exact status output contains three modified tracked documentation files:

- `README.md`
- `docs/research_report.md`
- `docs/walkthrough.md`

It also contains many pre-existing untracked experiment scripts, documentation files, `outputs/`, and a local virtual environment. This audit did not stage, alter, or remove any of them.

## Test Result

`python3 -m unittest scratch/test_experiment_137_operational_triage.py` passed:

```text
Ran 5 tests
OK
```

No test failure occurred. The full unedited test output is in `02_unit_test_full_log.txt`.

## Direct Absolute-Path Dependencies

The implementation and its dependency chain use `/Users/minho/Documents/Dataset` directly:

| File | Location / purpose |
|---|---|
| `run_rank_ensemble_calibration.py` | line 25: `univariate_ts.db`; lines 162-199 load TRAIN and TEST series |
| `run_experiment_40_original_score_normalization_sweep.py` | lines 29-32: result and log paths |
| `run_experiment_60_62_rocket_imaging_selector_variants.py` | lines 128-143: Exp40/55/56 result CSV locations |
| `run_experiment_119a_exp93_rank_order_validation.py` | lines 24-27: Exp93 result path |
| `run_experiment_132_block_b_review_integration.py` | lines 18-21: Exp119a and Exp55 result paths |
| `run_experiment_133_block_b_confidence_tiers.py` | lines 13-17: Exp119a and Exp131 result paths |
| `run_experiment_135_block_c_review_confirmation.py` | lines 17-22: Exp134 and Exp133 result paths |
| `run_experiment_137_operational_triage.py` | lines 13-18: Exp133 and Exp135 result paths |

## Missing Inputs for a Fresh Exp137 Rerun

At minimum, a rerun needs the SQLite DB plus upstream result CSVs under `/Users/minho/Documents/Dataset`, including Exp93/119a, Exp131, Exp133, Exp134, and Exp135 files. It also needs the repository modules imported by the listed scripts. The final Exp137 detail and summary CSV alone are sufficient for the verification artifacts in this folder, but not for a fresh end-to-end model rerun.

## Can the Earlier Handoff Package Reproduce This Alone?

No. The handoff ZIP intentionally excludes the 1.6 GB SQLite DB and many upstream result dependencies. It is sufficient for orientation and code review; it is not a self-contained runtime image. Full reproduction requires the repository and `/Users/minho/Documents/Dataset` directory described above.
