import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiment_60_62_rocket_imaging_selector_variants import run_experiment


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXP_ID = "experiment_69_operational_train_exceed_calibration"


def cleanup():
    for suffix in ["results.csv", "summary.csv", "stdout.log"]:
        path = DATA_DIR / f"{EXP_ID}_{suffix}"
        if path.exists():
            path.unlink()


def load_rows(suffix):
    return list(csv.DictReader((DATA_DIR / f"{EXP_ID}_{suffix}.csv").open()))


def test_exp69_shape_and_operational_columns():
    cleanup()
    run_experiment(EXP_ID, dataset_limit=2)
    rows = load_rows("results")
    assert len(rows) == 2 * 5
    assert {row["selector_name"] for row in rows} == {
        "strict_05pct_2of3",
        "operational_1pct_2of3",
        "operational_1pct_rocket_fallback",
        "operational_1pct_pair_then_rocket",
        "relaxed_15pct_rocket_fallback",
    }
    assert "calibration_profile" in rows[0]
    assert "max_candidate_train_exceed_rate" in rows[0]
    assert "candidate_threshold_rates" in rows[0]
    summary = load_rows("summary")
    assert summary
    one_pct = [row for row in summary if row["selector_name"].startswith("operational_1pct")]
    assert one_pct
    assert all(float(row["mean_train_exceed_rate"]) <= 0.011 for row in one_pct)
    cleanup()


if __name__ == "__main__":
    test_exp69_shape_and_operational_columns()
    print("experiment_69_operational_calibration_tests_passed")
