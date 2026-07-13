from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path
from validated_exp93_chain import VALIDATED_EXP93_PATH, VALIDATED_EXP93_SELECTOR, require_validated_exp93


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_120_exp93_reference_rebaseline"
TARGETS = {
    "experiment_113_train_normal_conformal_fusion": (DATA_DIR / "experiment_113_train_normal_conformal_fusion_results.csv", "baseline_exp93_nonpos_weak_alert_replace"),
    "experiment_114_pseudo_anomaly_score_probe": (DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_results.csv", "baseline_exp93_nonpos_weak_alert_replace"),
    "experiment_115_local_normal_state_score": (DATA_DIR / "experiment_115_local_normal_state_score_results.csv", "baseline_exp93_nonpos_weak_alert_replace"),
}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(row, key):
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def main():
    require_validated_exp93()
    validated = {row["dataset_name"]: row for row in read_rows(VALIDATED_EXP93_PATH) if row.get("config_name") == VALIDATED_EXP93_SELECTOR}
    out = []
    for target_id, (path, selector) in TARGETS.items():
        legacy = {row["dataset_name"]: row for row in read_rows(path) if row.get("config_name") == selector}
        if set(legacy) != set(validated):
            raise SystemExit(f"{target_id} baseline coverage mismatch")
        for name, old in legacy.items():
            new = validated[name]
            out.append({
                "experiment_id": EXPERIMENT_ID,
                "source_experiment": target_id,
                "dataset_name": name,
                "legacy_f1": number(old, "f1"),
                "validated_f1": number(new, "f1"),
                "f1_delta": number(new, "f1") - number(old, "f1"),
                "legacy_fp": number(old, "fp"),
                "validated_fp": number(new, "fp"),
                "fp_delta": number(new, "fp") - number(old, "fp"),
                "legacy_tp": number(old, "tp"),
                "validated_tp": number(new, "tp"),
                "tp_delta": number(new, "tp") - number(old, "tp"),
                "legacy_zero_f1": int(number(old, "f1") == 0.0),
                "validated_zero_f1": int(number(new, "f1") == 0.0),
            })
    summary = []
    for source in sorted({row["source_experiment"] for row in out}):
        rows = [row for row in out if row["source_experiment"] == source]
        summary.append({
            "experiment_id": EXPERIMENT_ID,
            "config_name": source,
            "selector_name": "reference_rebaseline_only",
            "threshold_method": "reference_only",
            "num_datasets": len(rows),
            "legacy_mean_f1": float(np.mean([row["legacy_f1"] for row in rows])),
            "validated_mean_f1": float(np.mean([row["validated_f1"] for row in rows])),
            "mean_f1": float(np.mean([row["validated_f1"] for row in rows])),
            "legacy_mean_fp": float(np.mean([row["legacy_fp"] for row in rows])),
            "validated_mean_fp": float(np.mean([row["validated_fp"] for row in rows])),
            "mean_fp": float(np.mean([row["validated_fp"] for row in rows])),
            "legacy_zero_f1_count": int(sum(row["legacy_zero_f1"] for row in rows)),
            "validated_zero_f1_count": int(sum(row["validated_zero_f1"] for row in rows)),
            "zero_f1_count": int(sum(row["validated_zero_f1"] for row in rows)),
        })
    write_csv(results_path(EXPERIMENT_ID), out)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    print(f"{EXPERIMENT_ID} finished rows={len(out)} datasets={len(validated)}", flush=True)


if __name__ == "__main__":
    main()
