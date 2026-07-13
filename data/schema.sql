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
