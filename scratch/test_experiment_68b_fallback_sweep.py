import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiment_60_62_rocket_imaging_selector_variants import run_experiment


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXP_ID = "experiment_68b_final_operational_fallback_sweep"


def cleanup():
    for suffix in ["results.csv", "summary.csv", "stdout.log"]:
        path = DATA_DIR / f"{EXP_ID}_{suffix}"
        if path.exists():
            path.unlink()


def load_rows(suffix):
    return list(csv.DictReader((DATA_DIR / f"{EXP_ID}_{suffix}.csv").open()))


def test_exp68b_shape():
    cleanup()
    run_experiment(EXP_ID, dataset_limit=2)
    rows = load_rows("results")
    assert len(rows) == 2 * 5
    assert {row["selector_name"] for row in rows} == {
        "exp68_reference_operational_v1",
        "fallback_top2_when_empty",
        "fallback_top3_when_empty",
        "fallback_adaptive_1to3",
        "fallback_rocket_2pct_when_empty",
    }
    assert "selector_reason" in rows[0]
    assert "selected_indices" in rows[0]
    cleanup()


if __name__ == "__main__":
    test_exp68b_shape()
    print("experiment_68b_fallback_sweep_tests_passed")
