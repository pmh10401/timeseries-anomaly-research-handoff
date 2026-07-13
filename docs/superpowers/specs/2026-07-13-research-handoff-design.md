# Time-Series Anomaly Research Handoff Design

## Purpose

Create a public, self-contained research handoff repository for the time-series anomaly-detection project. It must let a reviewer or GPT understand the data construction, experiment history, Exp137 operational-triage results, and strict TRAIN-only validation status without publishing the 1.6 GB SQLite database or raw time-series values.

## Boundaries

- Include Python experiment code, test code, research documents, selected CSV/JSON/Markdown results, figure assets, and presentation inputs.
- Include database schema, dataset-level catalog, aggregate statistics, integrity commands, and a precise local-data contract.
- Exclude `univariate_ts.db`, raw UCR directories, Python virtual environments, caches, temporary logs, local dashboard credentials, and any artifact over 90 MB.
- Preserve results as retrospective evidence. Do not present Exp137 as end-to-end strict TRAIN-only or prospectively validated.

## Repository Layout

- `scripts/`: source experiment and orchestration scripts, kept at their original filenames.
- `tests/`: available scratch tests, kept at their original filenames.
- `docs/`: original research documents plus this handoff's design and execution plan.
- `results/`: CSV/JSON/Markdown artifacts suitable for analysis, including Exp137 and policy-budget validation outputs.
- `artifacts/`: editable figures and the existing Exp137 presentation input package, excluding duplicate archives.
- `data/`: database documentation, SQL schema, and a dataset-level metadata catalog only.

## Data Contract

The source database is local-only at `/Users/minho/Documents/Dataset/univariate_ts.db`. A dataset row provides counts and length metadata; an instance row contains an individual time-series value blob and its label. The handoff catalog exposes only dataset-level metadata. Candidate indices in the experiment outputs are TEST instance indices, not positions within a time series.

## Success Criteria

1. A public GitHub repository exists with no database or raw time-series blobs.
2. The export includes all root Python scripts, available test scripts, all Markdown documentation, and non-duplicate analysis outputs under the GitHub file-size limit.
3. The repository explains how to acquire the local database separately and how to reproduce only where the environment/data are available.
4. A manifest lists source paths, exported paths, sizes, SHA256 values, and exclusion reasons.
5. A secret scan and `git fsck` run before the initial push.
