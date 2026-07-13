import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_original_improvement_experiment import (
    get_spec,
    min_zero_transform,
    summarize,
    transform_space_arrays,
)
from run_rank_ensemble_calibration import z_normalize


EXP_ID = "experiment_43_explanation_space_transforms"


def test_experiment_43_spec_targets_original_explanation_spaces():
    spec = get_spec(EXP_ID)
    spaces = {config["preprocess_space"] for config in spec["configs"]}

    assert spec["source_experiment_id"] == "paper_explanation_space_minzero"
    assert spec["data_variant"] == "original_repeated_normal"
    assert {"time_z", "minzero_z", "difference_z"} <= spaces
    assert "family_guard_v1" in spec["thresholds"]


def test_min_zero_transform_sets_each_row_minimum_to_zero():
    X = np.array([[-2.0, 1.0, 3.0], [5.0, 4.0, 7.0]], dtype=np.float32)

    transformed = min_zero_transform(X)

    np.testing.assert_allclose(transformed.min(axis=1), np.array([0.0, 0.0]))
    np.testing.assert_allclose(transformed[0], np.array([0.0, 3.0, 5.0]))
    np.testing.assert_allclose(transformed[1], np.array([1.0, 0.0, 3.0]))


def test_transform_space_arrays_preserves_shapes_and_current_default():
    X_train = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
    X_test = np.array([[4.0, 2.0, 1.0, 0.0]], dtype=np.float32)

    train_time, test_time = transform_space_arrays(X_train, X_test, "time_z")
    train_minzero, test_minzero = transform_space_arrays(X_train, X_test, "minzero_z")
    train_diff, test_diff = transform_space_arrays(X_train, X_test, "difference_z")

    np.testing.assert_allclose(train_time, z_normalize(X_train).astype(np.float32))
    assert train_minzero.shape == X_train.shape
    assert test_minzero.shape == X_test.shape
    assert float(train_minzero.min()) >= 0.0
    assert train_diff.shape == X_train.shape
    assert test_diff.shape == X_test.shape
    np.testing.assert_allclose(train_diff[:, 0], np.zeros(X_train.shape[0]))


def test_summary_keeps_preprocess_space_for_experiment_43():
    rows = [
        {
            "experiment_id": EXP_ID,
            "config_name": "minzero_z_rocket_256_knn3",
            "threshold_method": "count_cap_2pct",
            "family": "ElectricDevices",
            "preprocess_space": "minzero_z",
            "f1": 0.5,
            "auc_roc": 0.7,
            "auc_pr": 0.6,
            "predicted_count": 1,
            "anomaly_count": 1,
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "train_exceed_rate": 0.0,
            "oracle_f1": 0.8,
        }
    ]

    summary = summarize(rows)

    assert summary[0]["preprocess_space"] == "minzero_z"


if __name__ == "__main__":
    test_experiment_43_spec_targets_original_explanation_spaces()
    test_min_zero_transform_sets_each_row_minimum_to_zero()
    test_transform_space_arrays_preserves_shapes_and_current_default()
    test_summary_keeps_preprocess_space_for_experiment_43()
    print("experiment 43 explanation space tests passed")
