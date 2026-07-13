from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path
from run_experiment_89_74d_with_exp84_candidate import as_float
from run_experiment_93_nonpos_candidate_reranker import (
    EXP90_OPERATIONAL_SELECTOR,
    EXP90_RESEARCH_SELECTOR,
    EXP90_PATH,
    EXP87_PATH,
    choose_rows_for_dataset,
    load_exp87_rows,
    read_dict_rows,
)


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_119a_exp93_rank_order_validation"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
WORKERS = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "6"))


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


def load_maps():
    exp90_rows = read_dict_rows(EXP90_PATH)
    operational = {row["dataset_name"]: row for row in exp90_rows if row.get("selector_name") == EXP90_OPERATIONAL_SELECTOR}
    research = {row["dataset_name"]: row for row in exp90_rows if row.get("selector_name") == EXP90_RESEARCH_SELECTOR}
    historical = {row["dataset_name"]: row for row in read_dict_rows(EXP93_PATH) if row.get("selector_name") == EXP93_SELECTOR}
    if len(operational) != 1117 or set(operational) != set(research) or set(operational) != set(historical):
        raise SystemExit("Exp90/Exp93 coverage mismatch")
    return operational, research, historical, load_exp87_rows()


def validated_row(name, operational, research, historical, exp87):
    produced = choose_rows_for_dataset((name, operational, research, exp87))
    row = next(item for item in produced if item.get("selector_name") == EXP93_SELECTOR)
    legacy = historical[name]
    row.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_name": "exp93_rank_order_validated",
            "selector_name": "exp93_rank_order_validated",
            "selector_reason": "validated: stable descending score order for ROCKET, Exp55, and Exp56 candidate ranks",
            "rank_order_validation": "stable_descending_score_order",
            "historical_selected_indices": legacy.get("selected_indices", ""),
            "historical_f1": legacy.get("f1", ""),
            "historical_fp": legacy.get("fp", ""),
            "historical_tp": legacy.get("tp", ""),
            "historical_selector": EXP93_SELECTOR,
            "rank_output_changed": int(str(row.get("selected_indices", "")) != str(legacy.get("selected_indices", ""))),
        }
    )
    return row


def run_one(args):
    name, operational, research, historical, exp87 = args
    legacy = dict(historical[name])
    legacy.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_name": "baseline_exp93_historical",
            "selector_name": "baseline_exp93_historical",
            "selector_reason": "reference: recorded Exp93 output before rank-order validation",
            "rank_order_validation": "historical_unordered_top_set",
            "historical_selected_indices": legacy.get("selected_indices", ""),
            "historical_f1": legacy.get("f1", ""),
            "historical_fp": legacy.get("fp", ""),
            "historical_tp": legacy.get("tp", ""),
            "historical_selector": EXP93_SELECTOR,
            "rank_output_changed": 0,
        }
    )
    return [legacy, validated_row(name, operational, research, historical, exp87)]


def summarize(rows):
    out = []
    for config in sorted({row["config_name"] for row in rows}):
        subset = [row for row in rows if row["config_name"] == config]
        values = lambda key: [as_float(row.get(key)) for row in subset]
        f1s = values("f1")
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "config_name": config,
                "selector_name": config,
                "threshold_method": "selector",
                "num_datasets": len(subset),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(value == 0.0 for value in f1s),
                "mean_fp": float(np.mean(values("fp"))),
                "mean_tp": float(np.mean(values("tp"))),
                "mean_fn": float(np.mean(values("fn"))),
                "mean_auc_pr": float(np.mean(values("auc_pr"))),
                "mean_oracle_f1": float(np.mean(values("oracle_f1"))),
                "rank_output_changed_datasets": int(sum(as_float(row.get("rank_output_changed")) > 0 for row in subset)),
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


def main(limit=None):
    operational, research, historical, exp87 = load_maps()
    names = sorted(operational)[:limit] if limit else sorted(operational)
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, (name, operational[name], research[name], historical, exp87)): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append((name, repr(exc)))
                print(f"ERROR dataset={name} error={exc!r}", flush=True)
            if done % 25 == 0 or done == len(names):
                print(f"Progress: [{done:4d}/{len(names):4d}] rows={len(rows)} last={name} errors={len(errors)}", flush=True)
    if errors or len(rows) != len(names) * 2:
        raise SystemExit(f"coverage failure {len(rows)}/{len(names) * 2} {errors[:5]}")
    write_csv(results_path(EXPERIMENT_ID), rows)
    write_csv(summary_path(EXPERIMENT_ID), summarize(rows))
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int)
    main(parser.parse_args().dataset_limit)
