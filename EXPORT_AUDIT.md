# Export Audit

## Scope

This document records the safety and reproducibility checks run when the public research handoff was prepared. It is an export audit, not a claim that all historical experiments are end-to-end reproducible.

## Included Snapshot

- 1,119-row dataset metadata catalog from the datasets table
- 194 root Python source files
- 41 scratch test/validation files
- Existing research documents, Exp137 evidence, policy-validation results, and presentation artifacts
- 18 small selected presentation cases with annotations and editable SVG/PNG figures

## Exclusion Check

The staged export contains no DB, SQLite, JSONL checkpoint, virtual-environment, or Python bytecode path. The full univariate_ts.db remains local-only at /Users/minho/Documents/Dataset/univariate_ts.db.

No staged file exceeded 90 MB. The repository payload before Git object compression was about 20 MB.

## Credential Pattern Check

A staged-text scan found no direct private-key block, GitHub personal-access-token pattern, AWS access-key pattern, or API-key assignment pattern. The code contains environment-variable references for dashboard authentication; no username or password values are exported.

The gitleaks executable was not installed on this machine, so this is a targeted pattern scan rather than a full third-party secret-scanner certification.

## Verification Executed

    python3 -m py_compile *.py scratch/*.py
    python3 -m unittest scratch/test_experiment_137_operational_triage.py
    python3 -m unittest scratch/test_exp137_policy_validation_evidence.py
    python3 -m unittest scratch/test_exp137_budget_policy_validation.py
    python3 -m pytest -q scratch/test_rank_dashboard_operational.py

Observed results:

- Python compilation succeeded.
- Exp137 operational triage tests: 5 passed.
- Policy validation evidence tests: 3 passed.
- Budget policy validation tests: 6 passed.
- Dashboard operational tests: 24 passed.

## Formatting Caveat

The copied historical source and CSV artifacts contain pre-existing trailing whitespace and CRLF line endings. Running git diff --cached --check reported 141,949 formatting findings. These were not auto-rewritten because this repository is intended to preserve the source snapshot rather than to make a broad, unrelated formatting rewrite. This warning does not indicate DB inclusion, credential exposure, or Python syntax failure.

## Result Interpretation Caveat

All result CSVs are retrospective research evidence. Exp137 Hard alert metrics are autonomous. Standard/Priority review and mean combined F1 are human-assisted diagnostics. B2, Exp143, C0, and D1a do not prove end-to-end strict TRAIN-only provenance or prospective equipment performance.
