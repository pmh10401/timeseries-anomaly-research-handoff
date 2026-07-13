#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_experiment_60_62_rocket_imaging_selector_variants import (
    DATA_DIR,
    evaluate_indices,
    load_candidate_predictions,
    read_csv_candidates,
)


EXPERIMENT_ID = "experiment_75_score_source_diagnostic_harness"
DETAIL_PATH = DATA_DIR / f"{EXPERIMENT_ID}_results.csv"
SUMMARY_PATH = DATA_DIR / f"{EXPERIMENT_ID}_summary.csv"
LOG_PATH = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
REFERENCE_EXP = DATA_DIR / "experiment_74d_large_rank_review_tier_split_results.csv"

HARD_FAMILIES = {
    "Phoneme",
    "PigAirwayPressure",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "CricketX",
    "CricketY",
    "CricketZ",
    "InlineSkate",
    "GestureMidAirD3",
    "WordSynonyms",
    "ECG5000",
    "Crop",
    "StarLightCurves",
    "MelbournePedestrian",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "FordA",
    "FreezerRegularTrain",
}


def load_reference_targets(scope):
    _, datasets = read_csv_candidates()
    if scope == "all":
        return datasets
    if not REFERENCE_EXP.exists():
        return datasets
    targets = []
    with REFERENCE_EXP.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("selector_name") != "large_primary_rocket_guard_only":
                continue
            if scope == "zero74d" and float(row.get("f1") or 0.0) == 0.0:
                targets.append(row["dataset_name"])
            elif scope == "hard_families" and row.get("family") in HARD_FAMILIES:
                targets.append(row["dataset_name"])
    return sorted(set(targets))


def rank_diagnostics(y_test, scores):
    scores = np.asarray(scores, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.int64)
    order = np.argsort(-scores)
    anomaly_positions = np.flatnonzero(y_test == 1)
    ranks = []
    for idx in anomaly_positions:
        rank = int(np.flatnonzero(order == idx)[0]) + 1
        ranks.append(rank)
    if not ranks:
        ranks = [len(scores)]
    best_rank = min(ranks)
    median_rank = float(np.median(ranks))
    top_score = float(scores[order[0]]) if len(order) else 0.0
    best_anomaly_score = float(max(scores[anomaly_positions])) if len(anomaly_positions) else 0.0
    train_like_rank_pct = best_rank / max(1, len(scores))
    return {
        "best_anomaly_rank": best_rank,
        "median_anomaly_rank": median_rank,
        "best_anomaly_rank_pct": train_like_rank_pct,
        "anomaly_in_top1": int(best_rank <= 1),
        "anomaly_in_top2": int(best_rank <= 2),
        "anomaly_in_top3": int(best_rank <= 3),
        "anomaly_in_top5": int(best_rank <= 5),
        "anomaly_in_top10pct": int(best_rank <= max(1, math.ceil(len(scores) * 0.10))),
        "top_score": top_score,
        "best_anomaly_score": best_anomaly_score,
        "top_minus_best_anomaly_score": top_score - best_anomaly_score,
    }


def selected_indices_from_bundle(bundle):
    return set(bundle["indices"])


def run_dataset(dataset_name):
    record, y_test, bundles = load_candidate_predictions(dataset_name)
    rows = []
    for source_name, bundle in bundles.items():
        indices = selected_indices_from_bundle(bundle)
        metrics = evaluate_indices(y_test, bundle["test_scores"], indices)
        diag = rank_diagnostics(y_test, bundle["test_scores"])
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_name": dataset_name,
                "family": record["family"],
                "config_name": source_name,
                "selector_name": source_name,
                "score_source_name": source_name,
                "threshold_method": "source_native_threshold",
                "score_family": "score_source_diagnostic",
                "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
                "test_size": len(y_test),
                "anomaly_count": int(np.sum(y_test)),
                "train_score_count": len(bundle["train_scores"]),
                "threshold": bundle["threshold"],
                "q_effective": bundle["q_effective"],
                "cap_target": bundle["cap_target"],
                "train_exceed_count": bundle["train_exceed_count"],
                "train_exceed_rate": bundle["train_exceed_rate"],
                "selected_indices": " ".join(map(str, sorted(indices))),
                **metrics,
                **diag,
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    out = []
    keys = sorted({row["config_name"] for row in rows})
    for key in keys:
        subset = [row for row in rows if row["config_name"] == key]
        by_family = defaultdict(list)
        for row in subset:
            by_family[row["family"]].append(float(row["f1"]))
        family_means = [float(np.mean(values)) for values in by_family.values()]
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "config_name": key,
                "selector_name": key,
                "threshold_method": "source_native_threshold",
                "num_datasets": len(subset),
                "num_families": len(by_family),
                "mean_auc_roc": float(np.mean([float(row["auc_roc"]) for row in subset])),
                "mean_auc_pr": float(np.mean([float(row["auc_pr"]) for row in subset])),
                "mean_f1": float(np.mean([float(row["f1"]) for row in subset])),
                "median_f1": float(np.median([float(row["f1"]) for row in subset])),
                "p25_f1": float(np.percentile([float(row["f1"]) for row in subset], 25)),
                "zero_f1_count": sum(1 for row in subset if float(row["f1"]) == 0.0),
                "ge_0_5_count": sum(1 for row in subset if float(row["f1"]) >= 0.5),
                "family_macro_f1": float(np.mean(family_means)) if family_means else 0.0,
                "mean_predicted_count": float(np.mean([int(row["predicted_count"]) for row in subset])),
                "mean_anomaly_count": float(np.mean([int(row["anomaly_count"]) for row in subset])),
                "mean_tp": float(np.mean([int(row["tp"]) for row in subset])),
                "mean_fp": float(np.mean([int(row["fp"]) for row in subset])),
                "mean_fn": float(np.mean([int(row["fn"]) for row in subset])),
                "mean_train_exceed_rate": float(np.mean([float(row["train_exceed_rate"]) for row in subset])),
                "mean_oracle_f1": float(np.mean([float(row["oracle_f1"]) for row in subset])),
                "mean_best_anomaly_rank": float(np.mean([int(row["best_anomaly_rank"]) for row in subset])),
                "median_best_anomaly_rank": float(np.median([int(row["best_anomaly_rank"]) for row in subset])),
                "mean_best_anomaly_rank_pct": float(np.mean([float(row["best_anomaly_rank_pct"]) for row in subset])),
                "top1_capture_rate": float(np.mean([int(row["anomaly_in_top1"]) for row in subset])),
                "top3_capture_rate": float(np.mean([int(row["anomaly_in_top3"]) for row in subset])),
                "top5_capture_rate": float(np.mean([int(row["anomaly_in_top5"]) for row in subset])),
                "top10pct_capture_rate": float(np.mean([int(row["anomaly_in_top10pct"]) for row in subset])),
            }
        )
    return sorted(out, key=lambda row: (row["top5_capture_rate"], row["mean_oracle_f1"], row["mean_f1"]), reverse=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 75 score-source diagnostic harness")
    parser.add_argument("--scope", choices=["hard_families", "zero74d", "all"], default="hard_families")
    parser.add_argument("--dataset-limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in [DETAIL_PATH, SUMMARY_PATH]:
        if path.exists():
            path.unlink()
    datasets = load_reference_targets(args.scope)
    if args.dataset_limit is not None:
        datasets = datasets[: args.dataset_limit]
    rows = []
    with LOG_PATH.open("w") as log:
        log.write(f"{EXPERIMENT_ID} starting scope={args.scope} datasets={len(datasets)}\n")
        for pos, dataset_name in enumerate(datasets, 1):
            try:
                rows.extend(run_dataset(dataset_name))
            except Exception as exc:
                log.write(f"ERROR {dataset_name}: {exc}\n")
            if pos % 10 == 0 or pos == len(datasets):
                write_csv(DETAIL_PATH, rows)
                write_csv(SUMMARY_PATH, summarize(rows))
                msg = f"{EXPERIMENT_ID} progress {pos}/{len(datasets)} rows={len(rows)}"
                print(msg, flush=True)
                log.write(msg + "\n")
    write_csv(DETAIL_PATH, rows)
    write_csv(SUMMARY_PATH, summarize(rows))
    print(f"{EXPERIMENT_ID} finished. datasets={len(datasets)} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
