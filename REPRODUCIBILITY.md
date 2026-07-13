# Reproducibility and Limits

## What can be inspected without the DB

- Python source and available test source
- Exp137 result and summary CSV
- A0/A1/B1/B2 audit artifacts
- Exp143 TEST-length budget binding audit
- C0/D1a no-budget counterfactual results
- Dataset-level metadata catalog and database schema

## What requires local data

Most experiment scripts expect the local SQLite database:

    /Users/minho/Documents/Dataset/univariate_ts.db

The file is intentionally not part of this repository. A clone without it cannot reproduce feature extraction or full score computation.

## Suggested local checks

    python3 --version
    python3 -m unittest scratch/test_experiment_137_operational_triage.py
    python3 -m unittest scratch/test_exp137_policy_validation_evidence.py
    python3 -m unittest scratch/test_exp137_budget_policy_validation.py
    python3 -m pytest -q scratch/test_rank_dashboard_operational.py

Some tests or scripts may require optional packages such as NumPy, SciPy, scikit-learn, PyTorch, aeon, pyts, scikit-image, librosa, UMAP, TensorFlow, or Parametric UMAP. Install only the dependencies required by the specific experiment; the source is a research workspace rather than a locked production environment.

## Result integrity

The public MANIFEST.csv records SHA256 values at export time. Verify a file with:

    shasum -a 256 results/exp137_gpt_handoff_20260713/results/experiment_137_operational_triage_summary.csv

## Interpretation limits

- All Exp137 policy comparisons here are retrospective on already observed TEST data.
- Do not choose a threshold, a fixed K, or an alert budget because it is best on these TEST metrics.
- Priority review remains review-only.
- Policy-level TRAIN-only does not establish end-to-end strict TRAIN-only provenance.
- No artifact in this repository proves prospective performance in real equipment operation.
