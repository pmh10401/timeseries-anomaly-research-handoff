import sys
import csv
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_multi_aug_robust_experiment import (
    choose_acceleration_device,
    completed_dataset_names,
    count_cap_threshold,
    get_spec,
    prepare_arrays,
    read_existing_detail_rows,
    sanitize_and_align_series,
    score_metrics,
    threshold_for_method,
)


def test_sanitize_and_align_series_handles_variable_length_and_nan():
    series = [
        np.array([1.0, np.nan, 3.0], dtype=np.float32),
        np.array([1.0, 2.0, np.inf, 4.0, 5.0], dtype=np.float32),
    ]
    aligned = sanitize_and_align_series(series, target_len=4)
    assert aligned.shape == (2, 4)
    assert np.isfinite(aligned).all()


def test_prepare_arrays_uses_common_target_length():
    train = [np.arange(3, dtype=np.float32), np.arange(5, dtype=np.float32)]
    test = [np.arange(4, dtype=np.float32)]
    X_train, X_test = prepare_arrays(train, test, target_len=6)
    assert X_train.shape == (2, 6)
    assert X_test.shape == (1, 6)


def test_experiment_a_and_ab_have_distinct_threshold_sets():
    baseline = get_spec("experiment_41_multi_aug_robust_baseline")
    operational = get_spec("experiment_42_multi_aug_robust_operational")
    assert [item["name"] for item in baseline["thresholds"]] == ["percentile", "adaptive", "evt"]
    assert "count_cap_2pct" in [item["name"] for item in operational["thresholds"]]
    assert operational["include_operational_metrics"] is True


def test_choose_acceleration_device_prefers_mps_when_available():
    device = choose_acceleration_device("auto", mps_available=True, cuda_available=True)
    assert device.type == "mps"


def test_choose_acceleration_device_allows_cpu_override():
    device = choose_acceleration_device("cpu", mps_available=True, cuda_available=True)
    assert device.type == "cpu"


def test_mps_fallback_env_is_enabled_before_torch_work():
    import os

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_count_cap_threshold_limits_train_exceed_budget():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    threshold, effective, cap = count_cap_threshold(scores, 0.2)
    assert threshold == 3.0
    assert cap == 1
    assert effective == 0.2


def test_score_metrics_uses_top_k_oracle_f1_for_ties():
    metrics = score_metrics(np.array([1, 0]), np.array([0.5, 0.5]))
    assert metrics["oracle_f1"] == 2.0 / 3.0


def test_threshold_for_method_reuses_precomputed_originals():
    method = {"name": "adaptive", "kind": "adaptive_distribution"}
    threshold, effective, cap = threshold_for_method(
        method,
        np.array([0.0, 1.0, 2.0]),
        q_target=0.1,
        test_size=10,
        originals={"percentile": 10.0, "adaptive": 11.0, "evt": 12.0},
    )
    assert threshold == 11.0
    assert effective == 0.1
    assert cap == ""


def test_read_existing_detail_rows_supports_resume_skip_list():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "detail.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "dataset_name", "f1"])
            writer.writeheader()
            writer.writerow({"experiment_id": "experiment_42_multi_aug_robust_operational", "dataset_name": "A_normal_1", "f1": "0.5"})
            writer.writerow({"experiment_id": "experiment_41_multi_aug_robust_baseline", "dataset_name": "B_normal_1", "f1": "0.1"})

        rows, fieldnames = read_existing_detail_rows(path)

    assert fieldnames == ["experiment_id", "dataset_name", "f1"]
    assert completed_dataset_names(rows, "experiment_42_multi_aug_robust_operational") == {"A_normal_1"}


if __name__ == "__main__":
    test_sanitize_and_align_series_handles_variable_length_and_nan()
    test_prepare_arrays_uses_common_target_length()
    test_experiment_a_and_ab_have_distinct_threshold_sets()
    test_choose_acceleration_device_prefers_mps_when_available()
    test_choose_acceleration_device_allows_cpu_override()
    test_mps_fallback_env_is_enabled_before_torch_work()
    test_count_cap_threshold_limits_train_exceed_budget()
    test_score_metrics_uses_top_k_oracle_f1_for_ties()
    test_threshold_for_method_reuses_precomputed_originals()
    test_read_existing_detail_rows_supports_resume_skip_list()
    print("multi-aug robust experiment tests passed")
