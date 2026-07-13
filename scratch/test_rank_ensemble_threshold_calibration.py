import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_rank_ensemble_threshold_calibration.py"
sys.path.insert(0, str(ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location("rank_ensemble_threshold_calibration", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_keeps_config_and_strategy_grain():
    module = load_module()
    rows = [
        {
            "dataset_name": "A",
            "config_name": "equal",
            "strategy": "prior_q_02",
            "auc_roc": 1.0,
            "auc_pr": 1.0,
            "f1": 1.0,
            "predicted_count": 1,
            "oracle_f1": 1.0,
        },
        {
            "dataset_name": "B",
            "config_name": "equal",
            "strategy": "prior_q_02",
            "auc_roc": 0.5,
            "auc_pr": 0.5,
            "f1": 0.0,
            "predicted_count": 1,
            "oracle_f1": 0.5,
        },
        {
            "dataset_name": "A",
            "config_name": "ma60_fused20_hybrid20",
            "strategy": "prior_q_02",
            "auc_roc": 0.25,
            "auc_pr": 0.25,
            "f1": 0.25,
            "predicted_count": 1,
            "oracle_f1": 0.25,
        },
    ]

    summary = module.summarize_by_config_and_strategy(rows)

    by_key = {(row["config_name"], row["strategy"]): row for row in summary}
    assert set(by_key) == {("equal", "prior_q_02"), ("ma60_fused20_hybrid20", "prior_q_02")}
    assert by_key[("equal", "prior_q_02")]["num_datasets"] == 2
    assert by_key[("equal", "prior_q_02")]["mean_f1"] == 0.5
    assert by_key[("ma60_fused20_hybrid20", "prior_q_02")]["num_datasets"] == 1


if __name__ == "__main__":
    test_summary_keeps_config_and_strategy_grain()
    print("rank ensemble threshold calibration tests passed")
