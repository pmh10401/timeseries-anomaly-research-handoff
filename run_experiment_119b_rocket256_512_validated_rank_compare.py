from __future__ import annotations

import argparse
import csv
from pathlib import Path

import run_experiment_118_rocket512_knn3_exp93_source_probe as probe


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_119b_rocket256_512_validated_rank_compare"
VALIDATION_PATH = DATA_DIR / "experiment_119a_exp93_rank_order_validation_results.csv"


def require_validated_rank_output():
    if not VALIDATION_PATH.exists():
        raise SystemExit("Exp119a rank-order validation result is required before Exp119b")
    with VALIDATION_PATH.open(newline="") as handle:
        validated = {
            row["dataset_name"]
            for row in csv.DictReader(handle)
            if row.get("config_name") == "exp93_rank_order_validated"
        }
    if len(validated) != 1117:
        raise SystemExit(f"Exp119a validated coverage mismatch: {len(validated)}/1117")


def main(limit=None):
    require_validated_rank_output()
    probe.EXPERIMENT_ID = EXPERIMENT_ID
    probe.main(limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int)
    main(parser.parse_args().dataset_limit)
