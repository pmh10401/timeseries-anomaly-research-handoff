# SQLite Schema Design for Time Series Datasets

Here are three potential SQLite database structures for storing multiple time series datasets under `Univariate_ts`.

---

## Option A: Hybrid Binary Blob Storage (Recommended)
This approach stores the metadata in relational columns and the time series numerical values as a binary `BLOB` (e.g., float32 array bytes).

### Schema
```sql
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,         -- e.g., 'ArrowHead'
    univariate BOOLEAN NOT NULL,       -- 1 for true, 0 for false
    equal_length BOOLEAN NOT NULL,     -- 1 for true, 0 for false
    series_length INTEGER,             -- Length of series (can be NULL if variable length)
    has_missing BOOLEAN NOT NULL       -- 1 for true, 0 for false
);

CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    split TEXT NOT NULL,                -- 'TRAIN' or 'TEST'
    instance_index INTEGER NOT NULL,    -- 0-based index in the split
    label TEXT NOT NULL,                -- Class label (e.g., '0', '1', 'Avonlea')
    values_blob BLOB NOT NULL,          -- Binary float32 array bytes
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE UNIQUE INDEX idx_instances_dataset_split_index 
ON instances(dataset_id, split, instance_index);
```

### How to use in Python:
* **Saving**: `np.array(values, dtype=np.float32).tobytes()`
* **Loading**: `np.frombuffer(row['values_blob'], dtype=np.float32)`

### Pros & Cons
* **Pros**: 
  * **High Performance**: Extremely fast read/write speeds since Python can directly map bytes to numpy arrays.
  * **Space Efficient**: Uses minimal disk space (4 bytes per float).
* **Cons**:
  * You cannot query individual timesteps directly using SQL (e.g., `WHERE timestep_value > 0.5` is not possible within SQLite).

---

## Option B: Fully Normalized Relational Storage
This approach splits the data down to individual time steps, creating a row for every single value.

### Schema
```sql
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    univariate BOOLEAN NOT NULL,
    equal_length BOOLEAN NOT NULL,
    series_length INTEGER,
    has_missing BOOLEAN NOT NULL
);

CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    instance_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE series_data (
    instance_id INTEGER NOT NULL,
    timestep INTEGER NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (instance_id, timestep),
    FOREIGN KEY (instance_id) REFERENCES instances(id)
);
```

### Pros & Cons
* **Pros**:
  * **Pure SQL Querying**: You can run complex SQL queries, aggregations, and filters directly on the time series (e.g., `AVG(value)`, `MIN/MAX` per window, etc.).
* **Cons**:
  * **Performance Bottleneck**: Inserts and reads will be very slow because of the sheer number of rows (e.g., 100 datasets × 1000 instances × 500 timesteps = 50 million rows).
  * **Huge Database Size**: SQLite overhead for index keys and column metadata for 50M+ rows will make the file size massive.

---

## Option C: JSON / Comma-Separated Text Storage
This stores the sequence of numbers as a plain text string (e.g., `"1.2,3.4,-0.5"`) or a JSON array (`"[1.2, 3.4, -0.5]"`).

### Schema
```sql
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    univariate BOOLEAN NOT NULL,
    equal_length BOOLEAN NOT NULL,
    series_length INTEGER,
    has_missing BOOLEAN NOT NULL
);

CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    instance_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    values_text TEXT NOT NULL,         -- e.g., "1.2,3.4,-0.5"
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
```

### Pros & Cons
* **Pros**:
  * Human-readable when opening the database with a database GUI viewer.
  * Easy to parse in any language.
* **Cons**:
  * Parsing strings to floats in Python is slower than binary numpy conversion.
  * Takes more disk space than binary BLOB format.
