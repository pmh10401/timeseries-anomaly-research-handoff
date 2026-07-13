from __future__ import annotations

import csv
from pathlib import Path


DATA_DIR = Path("/Users/minho/Documents/Dataset")
VALIDATED_EXP93_PATH = DATA_DIR / "experiment_119a_exp93_rank_order_validation_results.csv"
VALIDATED_EXP93_SELECTOR = "exp93_rank_order_validated"


def require_validated_exp93():
    if not VALIDATED_EXP93_PATH.exists():
        raise SystemExit("Exp119a validated Exp93 output is required")
    with VALIDATED_EXP93_PATH.open(newline="") as handle:
        names = {
            row["dataset_name"]
            for row in csv.DictReader(handle)
            if row.get("config_name") == VALIDATED_EXP93_SELECTOR
        }
    if len(names) != 1117:
        raise SystemExit(f"Validated Exp93 coverage mismatch: {len(names)}/1117")


def configure_exp93(module, experiment_id):
    require_validated_exp93()
    module.EXPERIMENT_ID = experiment_id
    module.STDOUT_LOG = DATA_DIR / f"{experiment_id}_stdout.log"
    module.EXP93_PATH = VALIDATED_EXP93_PATH
    if hasattr(module, "EXP93_SELECTOR"):
        module.EXP93_SELECTOR = VALIDATED_EXP93_SELECTOR
    if hasattr(module, "EXP93_OPERATING_SELECTOR"):
        module.EXP93_OPERATING_SELECTOR = VALIDATED_EXP93_SELECTOR
    return module
