import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiment_60_62_rocket_imaging_selector_variants import run_experiment


DATA_DIR = Path("/Users/minho/Documents/Dataset")


def load_rows(exp_id, suffix):
    return list(csv.DictReader((DATA_DIR / f"{exp_id}_{suffix}.csv").open()))


def cleanup(exp_id):
    for suffix in ["results", "summary", "stdout"]:
        path = DATA_DIR / f"{exp_id}_{suffix}.csv"
        if path.exists():
            path.unlink()
    log_path = DATA_DIR / f"{exp_id}_stdout.log"
    if log_path.exists():
        log_path.unlink()


def test_exp60_shape():
    exp_id = "experiment_60_selector_fp_guard_variants"
    cleanup(exp_id)
    run_experiment(exp_id, dataset_limit=5)
    rows = load_rows(exp_id, "results")
    assert len(rows) == 5 * 4
    assert {row["selector_name"] for row in rows} == {
        "rocket_default",
        "agreement_count_v1_reference",
        "agreement_fp_guard_2pct",
        "agreement_fp_guard_3pct",
    }
    cleanup(exp_id)


def test_exp61_index_columns():
    exp_id = "experiment_61_selector_index_agreement"
    cleanup(exp_id)
    run_experiment(exp_id, dataset_limit=2)
    rows = load_rows(exp_id, "results")
    assert len(rows) == 2 * 4
    for key in ["rocket_exp55_overlap", "rocket_exp56_overlap", "exp55_exp56_overlap", "selected_indices"]:
        assert key in rows[0]
    cleanup(exp_id)


def test_exp62_index_columns():
    exp_id = "experiment_62_selector_guarded_index_agreement"
    cleanup(exp_id)


def test_exp63_cap_sweep_shape():
    exp_id = "experiment_63_guarded_cap_sweep"
    cleanup(exp_id)
    run_experiment(exp_id, dataset_limit=2)
    rows = load_rows(exp_id, "results")
    assert len(rows) == 2 * 9
    assert {
        "cap_1pct_2of3",
        "cap_2pct_2of3",
        "cap_3pct_2of3",
        "cap_5pct_2of3",
        "cap_max1_2of3",
        "cap_max3_2of3",
        "cap_max5_2of3",
        "cap_min2pct_max5_2of3",
        "cap_min3pct_max5_2of3",
    } == {row["selector_name"] for row in rows}
    cleanup(exp_id)


def test_exp64_to_68_shapes():
    expected = {
        "experiment_64_guarded_with_fallback": 4,
        "experiment_65_confidence_tier_selector": 4,
        "experiment_66_train_normal_alert_budget": 4,
        "experiment_67_hard_family_fallback_selector": 4,
        "experiment_68_final_operational_selector": 4,
    }
    for exp_id, strategy_count in expected.items():
        cleanup(exp_id)
        run_experiment(exp_id, dataset_limit=2)
        rows = load_rows(exp_id, "results")
        assert len(rows) == 2 * strategy_count
        assert "selector_reason" in rows[0]
        assert "selected_indices" in rows[0]
        cleanup(exp_id)


if __name__ == "__main__":
    test_exp60_shape()
    test_exp61_index_columns()
    test_exp62_index_columns()
    test_exp63_cap_sweep_shape()
    test_exp64_to_68_shapes()
    print("experiment_60_62_selector_variant_tests_passed")
