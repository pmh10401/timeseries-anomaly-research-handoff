import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_balanced_improvement_experiment.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location("balanced_improvement", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(path):
    rows = [
        {
            "dataset_name": "Toy_normal_0",
            "family": "Toy",
            "manifest_variant": "balanced_2pct",
            "series_length": "16",
            "actual_len_median": "16",
            "actual_len_max": "16",
            "clean_test_actual_len_median": "16",
            "original_train_count": "4",
            "clean_train_count": "3",
            "original_test_normal_count": "49",
            "original_test_anomaly_count": "1",
            "clean_test_normal_count": "49",
            "clean_test_anomaly_count": "1",
            "clean_test_total_count": "50",
            "removed_train_overlap_normals": "0",
            "removed_test_duplicate_normals": "0",
            "is_eligible": "1",
            "train_instance_ids": "1;2;3;4",
            "clean_train_instance_ids": "1;2;3",
            "clean_test_normal_instance_ids": "5;6;7",
            "clean_test_anomaly_instance_ids": "8",
        },
        {
            "dataset_name": "Toy_normal_1",
            "family": "Toy",
            "manifest_variant": "strict_unbalanced",
            "series_length": "16",
            "actual_len_median": "16",
            "actual_len_max": "16",
            "clean_test_actual_len_median": "16",
            "original_train_count": "4",
            "clean_train_count": "4",
            "original_test_normal_count": "49",
            "original_test_anomaly_count": "1",
            "clean_test_normal_count": "49",
            "clean_test_anomaly_count": "1",
            "clean_test_total_count": "50",
            "removed_train_overlap_normals": "0",
            "removed_test_duplicate_normals": "0",
            "is_eligible": "1",
            "train_instance_ids": "9;10",
            "clean_train_instance_ids": "9;10",
            "clean_test_normal_instance_ids": "11",
            "clean_test_anomaly_instance_ids": "12",
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_loader_filters_balanced_eligible_and_parses_ids(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path)

    rows = module.load_manifest_rows(manifest_path, manifest_variant="balanced_2pct")

    assert len(rows) == 1
    assert rows[0]["dataset_name"] == "Toy_normal_0"
    assert rows[0]["train_instance_ids"] == [1, 2, 3, 4]
    assert rows[0]["clean_train_instance_ids"] == [1, 2, 3]
    assert rows[0]["clean_test_anomaly_instance_ids"] == [8]
    assert rows[0]["clean_test_total_count"] == 50


def test_manifest_loader_can_exclude_oversized_datasets(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path)
    with manifest_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_name",
                "family",
                "manifest_variant",
                "series_length",
                "actual_len_median",
                "actual_len_max",
                "clean_test_actual_len_median",
                "original_train_count",
                "clean_train_count",
                "original_test_normal_count",
                "original_test_anomaly_count",
                "clean_test_normal_count",
                "clean_test_anomaly_count",
                "clean_test_total_count",
                "removed_train_overlap_normals",
                "removed_test_duplicate_normals",
                "is_eligible",
                "train_instance_ids",
                "clean_train_instance_ids",
                "clean_test_normal_instance_ids",
                "clean_test_anomaly_instance_ids",
            ],
        )
        writer.writerow(
            {
                "dataset_name": "CornellWhaleChallenge",
                "family": "CornellWhaleChallenge",
                "manifest_variant": "balanced_2pct",
                "series_length": "4000",
                "actual_len_median": "4000",
                "actual_len_max": "4000",
                "clean_test_actual_len_median": "4000",
                "original_train_count": "18378",
                "clean_train_count": "18378",
                "original_test_normal_count": "4459",
                "original_test_anomaly_count": "91",
                "clean_test_normal_count": "4459",
                "clean_test_anomaly_count": "91",
                "clean_test_total_count": "4550",
                "removed_train_overlap_normals": "0",
                "removed_test_duplicate_normals": "0",
                "is_eligible": "1",
                "train_instance_ids": "101;102",
                "clean_train_instance_ids": "101;102",
                "clean_test_normal_instance_ids": "201;202",
                "clean_test_anomaly_instance_ids": "301",
            }
        )

    rows = module.load_manifest_rows(
        manifest_path,
        manifest_variant="balanced_2pct",
        excluded_dataset_names={"CornellWhaleChallenge"},
    )

    assert [row["dataset_name"] for row in rows] == ["Toy_normal_0"]


def test_rank_normalization_uses_train_reference_distribution():
    module = load_module()

    train_rank, test_rank = module.rank_normalize_scores([1.0, 3.0, 2.0], [0.5, 2.0, 4.0])

    assert train_rank.tolist() == [1 / 3, 1.0, 2 / 3]
    assert test_rank.tolist() == [0.0, 2 / 3, 1.0]


def test_count_cap_threshold_keeps_declared_train_budget():
    module = load_module()

    threshold, effective_rate, cap = module.count_cap_threshold([1, 2, 3, 4, 5], 0.4)

    assert threshold == 3.0
    assert effective_rate == 0.4
    assert cap == 2


def test_summary_includes_family_macro_and_zero_f1():
    module = load_module()
    rows = [
        {
            "experiment_id": "experiment_x",
            "manifest_variant": "balanced_2pct",
            "train_variant": "clean_test_only",
            "config_name": "a",
            "threshold_method": "count_cap_2pct",
            "family": "A",
            "f1": 1.0,
            "auc_roc": 1.0,
            "auc_pr": 1.0,
            "oracle_f1": 1.0,
            "predicted_count": 1,
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "train_exceed_rate": 0.0,
        },
        {
            "experiment_id": "experiment_x",
            "manifest_variant": "balanced_2pct",
            "train_variant": "clean_test_only",
            "config_name": "a",
            "threshold_method": "count_cap_2pct",
            "family": "B",
            "f1": 0.0,
            "auc_roc": 0.5,
            "auc_pr": 0.2,
            "oracle_f1": 0.4,
            "predicted_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 1,
            "train_exceed_rate": 0.0,
        },
    ]

    summary = module.summarize(rows)

    assert len(summary) == 1
    assert summary[0]["num_datasets"] == 2
    assert summary[0]["mean_f1"] == 0.5
    assert summary[0]["family_macro_f1"] == 0.5
    assert summary[0]["zero_f1_count"] == 1


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_manifest_loader_filters_balanced_eligible_and_parses_ids(tmp_path)
        test_manifest_loader_can_exclude_oversized_datasets(tmp_path)
        test_rank_normalization_uses_train_reference_distribution()
        test_count_cap_threshold_keeps_declared_train_budget()
        test_summary_includes_family_macro_and_zero_f1()
    print("balanced improvement experiment tests passed")
