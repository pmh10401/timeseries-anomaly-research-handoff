import sys
import csv
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_original_improvement_experiment import (
    cached_classical_embedding_pair,
    classical_embedding_pair,
    completed_dataset_names,
    get_spec,
    load_dataset_names_from_db,
    parse_family,
    rate_for_threshold,
    read_existing_detail_rows,
    summarize,
)


def test_original_specs_keep_experiment_35_policy():
    spec = get_spec("experiment_35_original_threshold_policy_sweep")
    assert spec["source_experiment_id"] == "experiment_35_balanced_threshold_policy_sweep"
    assert "dynamic_1_over_n" in spec["thresholds"]
    assert "family_guard_v1" in spec["thresholds"]
    assert spec["configs"][0]["name"] == "rocket_256_knn3_threshold_probe"


def test_original_rate_policy_uses_original_record_shape():
    record = {
        "family": "Yoga",
        "y_test": [0] * 100,
        "test_actual_len_median": 600,
    }
    assert rate_for_threshold("count_cap_2pct", record) == (0.02, "count_cap_rate")
    assert rate_for_threshold("dynamic_1_over_n", record) == (0.01, "dynamic_count_cap")
    assert rate_for_threshold("adaptive_v0", record) == (0.02, "adaptive_count_cap")
    assert rate_for_threshold("family_guard_v1", record) == (0.03, "family_guard_count_cap")


def test_summary_orders_by_mean_f1_and_family_macro():
    rows = [
        {
            "experiment_id": "experiment_35_original_threshold_policy_sweep",
            "config_name": "a",
            "threshold_method": "count_cap_2pct",
            "family": "F1",
            "f1": 0.2,
            "auc_roc": 0.5,
            "auc_pr": 0.4,
            "predicted_count": 1,
            "anomaly_count": 1,
            "tp": 0,
            "fp": 1,
            "fn": 1,
            "train_exceed_rate": 0.0,
            "oracle_f1": 0.5,
        },
        {
            "experiment_id": "experiment_35_original_threshold_policy_sweep",
            "config_name": "b",
            "threshold_method": "count_cap_2pct",
            "family": "F2",
            "f1": 0.8,
            "auc_roc": 0.8,
            "auc_pr": 0.7,
            "predicted_count": 1,
            "anomaly_count": 1,
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "train_exceed_rate": 0.0,
            "oracle_f1": 0.8,
        },
    ]
    summary = summarize(rows)
    assert summary[0]["config_name"] == "b"
    assert summary[0]["data_variant"] == "original_repeated_normal"
    assert summary[0]["family_macro_f1"] == 0.8


def test_family_parser():
    assert parse_family("ECGFiveDays_normal_2") == "ECGFiveDays"
    assert parse_family("CornellWhaleChallenge") == "CornellWhaleChallenge"


def test_load_dataset_names_uses_explicit_db_path_and_exclusions():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "datasets.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE datasets (name TEXT)")
        conn.executemany(
            "INSERT INTO datasets(name) VALUES (?)",
            [
                ("Z_normal_1",),
                ("A_normal_1",),
                ("CornellWhaleChallenge",),
                ("Wafer_normal_1",),
            ],
        )
        conn.commit()
        conn.close()

        assert load_dataset_names_from_db(db_path) == ["A_normal_1", "Z_normal_1"]


def test_read_existing_detail_rows_supports_resume_skip_list():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "detail.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "dataset_name", "f1"])
            writer.writeheader()
            writer.writerow({"experiment_id": "experiment_43_explanation_space_transforms", "dataset_name": "A_normal_1", "f1": "0.5"})
            writer.writerow({"experiment_id": "other", "dataset_name": "B_normal_1", "f1": "0.1"})

        rows, fieldnames = read_existing_detail_rows(path)

    assert fieldnames == ["experiment_id", "dataset_name", "f1"]
    assert len(rows) == 2
    assert completed_dataset_names(rows, "experiment_43_explanation_space_transforms") == {"A_normal_1"}


def test_cached_classical_embedding_pair_reuses_max_width_features():
    X_train = np.arange(32, dtype=np.float32).reshape(4, 8)
    X_test = np.arange(16, dtype=np.float32).reshape(2, 8)
    config16 = {"embedding": "fft", "n_features": 16}
    config32 = {"embedding": "fft", "n_features": 32}
    cache = {}

    train16, test16 = cached_classical_embedding_pair(
        X_train,
        X_test,
        config16,
        cache=cache,
        cache_key=("time_z", "fft"),
        max_width=32,
    )
    train32, test32 = cached_classical_embedding_pair(
        X_train,
        X_test,
        config32,
        cache=cache,
        cache_key=("time_z", "fft"),
        max_width=32,
    )
    direct16 = classical_embedding_pair(X_train, X_test, config16)

    assert len(cache) == 1
    assert train16.shape[1] == 16
    assert train32.shape[1] == 32
    assert np.allclose(train16, direct16[0])
    assert np.allclose(test16, direct16[1])


if __name__ == "__main__":
    test_original_specs_keep_experiment_35_policy()
    test_original_rate_policy_uses_original_record_shape()
    test_summary_orders_by_mean_f1_and_family_macro()
    test_family_parser()
    test_load_dataset_names_uses_explicit_db_path_and_exclusions()
    test_read_existing_detail_rows_supports_resume_skip_list()
    test_cached_classical_embedding_pair_reuses_max_width_features()
    print("original improvement experiment tests passed")
