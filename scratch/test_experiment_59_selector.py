import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_experiment_59_rocket_imaging_selector import (
    EXPERIMENT_ID,
    RESULTS_PATH,
    SUMMARY_PATH,
    family_prior_maps,
    read_candidate_rows,
)


def test_candidate_coverage():
    candidates, datasets = read_candidate_rows()
    assert len(datasets) == 1117
    for rows in candidates.values():
        assert set(rows) == set(datasets)


def test_summary_keeps_oracle_out_of_operational_top():
    rows = list(csv.DictReader(SUMMARY_PATH.open()))
    assert rows[0]["selector_name"] != "oracle_candidate_upper_bound"
    oracle = [row for row in rows if row["selector_name"] == "oracle_candidate_upper_bound"][0]
    assert oracle["selector_validation_mode"] == "labeled_upper_bound"
    assert oracle["operational_candidate"] == "0"


def test_family_prior_leave_one_recipe_shape():
    candidates, datasets = read_candidate_rows()
    prior = family_prior_maps(candidates, datasets, "operational")
    assert set(prior) == set(datasets)
    assert all(selected in {"rocket_exp40", "exp55_best", "exp56_best"} for selected, _ in prior.values())


def test_result_shape():
    rows = list(csv.DictReader(RESULTS_PATH.open()))
    assert {row["experiment_id"] for row in rows} == {EXPERIMENT_ID}
    assert len(rows) == 1117 * 9
    counts = {}
    for row in rows:
        counts[row["selector_name"]] = counts.get(row["selector_name"], 0) + 1
    assert set(counts.values()) == {1117}


if __name__ == "__main__":
    test_candidate_coverage()
    test_summary_keeps_oracle_out_of_operational_top()
    test_family_prior_leave_one_recipe_shape()
    test_result_shape()
    print("experiment_59_selector_tests_passed")
