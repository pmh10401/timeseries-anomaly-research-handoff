# Prospective Validation Available Fields

## Confirmed available now

- Dataset name, TRAIN/TEST split, instance index, label, and values blob: SQLite `datasets` and `instances` tables.
- Dataset-level sequence length and train/test counts: `datasets` table.
- Experiment-level candidate indices, lane assignments, and offline metrics: result CSVs in `/Users/minho/Documents/Dataset`.

## Not confirmed in current DB or Dataset directory

- equipment ID
- recipe ID
- run end time or event timestamp
- user-visible alert timestamp
- user confirmation/rejection record
- maintenance or process-action history
- user/assessor ID
- key joining model output to a real operational action
- fields distinguishing new equipment or new recipe

The repository DB stores benchmark-style dataset instances, not demonstrated production equipment runs. Therefore the real collection unit, action timing, and shadow-mode persistence schema are **not verifiable from current files**. No prospective protocol is asserted here.