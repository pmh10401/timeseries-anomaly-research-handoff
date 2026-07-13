# Public Handoff Manifest

## Export Snapshot

- Export repository: pmh10401/timeseries-anomaly-research-handoff
- Export branch before merge: agent/research-handoff-export
- Public artifacts listed: 464
- Approximate repository payload: 20 MB
- Dataset metadata rows: 1,119
- Source DB: /Users/minho/Documents/Dataset/univariate_ts.db

The file-level inventory, source reference, byte size, and SHA256 digest are in [MANIFEST.csv](MANIFEST.csv).

## Included Categories

| Category | Files | Purpose |
|---|---:|---|
| source_code | 194 | Original root Python experiment, evaluation, DB, and dashboard scripts |
| test_code | 41 | Available scratch tests and validation scripts |
| research_document | 56 | Existing reports, experimental notes, and plans |
| exp137_handoff_evidence | 15 | Focused Exp137 documents and original result CSVs |
| policy_validation_evidence | 35 | A0/A1/B1/B2, Exp143, C0, D1a audit outputs |
| presentation_evidence | 112 | Presentation inputs, case metadata, selected case CSVs, SVG/PNG figures |
| database_metadata_export | 4 | Schema, metadata-only catalog, aggregate DB information |
| handoff_document | 7 | Public README, reproducibility notes, design, plan, audit, and export configuration |

## Explicit Exclusions

| Excluded item | Reason |
|---|---|
| univariate_ts.db | About 1.6 GB; contains the complete raw dataset and is intentionally local-only |
| Raw UCR directories | Large raw corpus; not required to understand the code/results |
| Python virtual environments and caches | Platform-specific and not source artifacts |
| JSONL checkpoints | Resume state rather than stable analysis evidence |
| ZIP duplicates and duplicate PPTX copies | Equivalent content is represented by unpacked docs/results/figures |
| Dashboard credential values | Authentication reads environment variables; no values are exported |

## Selected Case Data

The presentation package intentionally includes 18 selected cases with TRAIN/TEST CSV plus annotations. They are limited explanatory samples, not the full raw corpus. Actual anomaly labels in those samples are for retrospective evaluation and presentation explanation, never for prediction routing input.

## Integrity Check

To verify a listed file:

    shasum -a 256 path/to/file

Compare the result with the SHA256 column in MANIFEST.csv. The manifest is an export record, not a replacement for the original DB integrity check.
