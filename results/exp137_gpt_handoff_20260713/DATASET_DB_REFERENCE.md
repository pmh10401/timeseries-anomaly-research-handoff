# Dataset Database Reference

## Location and Size

- Absolute path: `/Users/minho/Documents/Dataset/univariate_ts.db`
- Size at handoff: about 1.6 GB
- It is intentionally excluded from this archive.

## Verified State

```text
PRAGMA integrity_check -> ok
datasets -> 1119 rows
instances -> 396737 rows
```

## Tables

```sql
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    univariate BOOLEAN NOT NULL,
    equal_length BOOLEAN NOT NULL,
    series_length INTEGER,
    has_missing BOOLEAN NOT NULL,
    train_normal_count INTEGER NOT NULL,
    train_anomaly_count INTEGER NOT NULL,
    train_total_count INTEGER NOT NULL,
    test_normal_count INTEGER NOT NULL,
    test_anomaly_count INTEGER NOT NULL,
    test_total_count INTEGER NOT NULL,
    total_normal_count INTEGER NOT NULL,
    total_anomaly_count INTEGER NOT NULL,
    total_count INTEGER NOT NULL
);

CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    instance_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    values_blob BLOB NOT NULL,
    labels_blob BLOB,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
```

## Verification Commands

```bash
sqlite3 '/Users/minho/Documents/Dataset/univariate_ts.db' \
  "PRAGMA integrity_check;" \
  "SELECT COUNT(*) AS datasets FROM datasets;" \
  "SELECT COUNT(*) AS instances FROM instances;"
```

## Research Rule

Use `TRAIN` normal instances only to fit transforms, score distributions, thresholds, and alert budgets. Use `TEST` labels only after prediction to measure offline TP, FP, F1, precision, and recall.
