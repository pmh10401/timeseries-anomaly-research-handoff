import importlib.util
import csv
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_experiment_40_original_score_normalization_sweep.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_40", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configs_include_raw_and_local_gap():
    module = load_module()

    names = [config["name"] for config in module.CONFIGS]

    assert "rocket_256_knn3_raw" in names
    assert "rocket_256_knn3_local_gap" in names


def test_summarize_orders_by_mean_f1_and_counts_zero():
    module = load_module()
    rows = [
        {
            "config_name": "a",
            "threshold_method": "count_cap_2pct",
            "threshold_family": "count_cap_rate",
            "f1": 1.0,
            "auc_roc": 1.0,
            "auc_pr": 0.8,
            "oracle_f1": 1.0,
            "predicted_count": 2,
            "tp": 1,
            "fp": 1,
            "fn": 0,
            "anomaly_count": 1,
            "train_exceed_rate": 0.01,
            "family": "FamA",
        },
        {
            "config_name": "a",
            "threshold_method": "count_cap_2pct",
            "threshold_family": "count_cap_rate",
            "f1": 0.0,
            "auc_roc": 0.5,
            "auc_pr": 0.1,
            "oracle_f1": 0.5,
            "predicted_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 1,
            "anomaly_count": 1,
            "train_exceed_rate": 0.0,
            "family": "FamB",
        },
        {
            "config_name": "b",
            "threshold_method": "count_cap_2pct",
            "threshold_family": "count_cap_rate",
            "f1": 0.75,
            "auc_roc": 0.9,
            "auc_pr": 0.7,
            "oracle_f1": 0.8,
            "predicted_count": 1,
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "anomaly_count": 1,
            "train_exceed_rate": 0.01,
            "family": "FamA",
        },
    ]

    summary = module.summarize(rows)

    assert summary[0]["config_name"] == "b"
    assert summary[1]["config_name"] == "a"
    assert summary[1]["zero_f1_count"] == 1
    assert summary[1]["family_macro_f1"] == 0.5


def test_parse_family_from_one_vs_all_name():
    module = load_module()

    assert module.parse_family("ECGFiveDays_normal_2") == "ECGFiveDays"
    assert module.parse_family("CornellWhaleChallenge") == "CornellWhaleChallenge"


def test_read_existing_detail_rows_supports_resume_skip_list():
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "detail.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_name", "f1"])
            writer.writeheader()
            writer.writerow({"dataset_name": "A_normal_1", "f1": "0.5"})

        rows, fieldnames = module.read_existing_detail_rows(path)

    assert fieldnames == ["dataset_name", "f1"]
    assert module.completed_dataset_names(rows) == {"A_normal_1"}


if __name__ == "__main__":
    test_configs_include_raw_and_local_gap()
    test_summarize_orders_by_mean_f1_and_counts_zero()
    test_parse_family_from_one_vs_all_name()
    test_read_existing_detail_rows_supports_resume_skip_list()
    print("experiment 40 original score normalization tests passed")
