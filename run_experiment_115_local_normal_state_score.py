from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_balanced_improvement_experiment import density_knn_score_pair
from run_experiment_40_original_score_normalization_sweep import (
    count_cap_threshold,
    evaluate_threshold,
    parse_family,
    score_metrics,
)
from run_experiment_60_62_rocket_imaging_selector_variants import evaluate_indices, results_path, summary_path
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices
from run_model_hard_research_experiments import (
    gmm_crossfit_score_pair,
    kmeans_crossfit_score_pair,
    rocket_feature_pair,
    scale_feature_pair,
)
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_115_local_normal_state_score"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
WORKERS = int(os.environ.get("EXP115_WORKERS", "4"))
MIN_CLUSTER_TRAIN = 30
THRESHOLDS = (("diagnostic_count_cap_1pct", 0.01), ("diagnostic_count_cap_2pct", 0.02), ("diagnostic_count_cap_3pct", 0.03))
CONFIGS = (
    {"name": "rocket_global_local_gap", "mode": "global_local_gap"},
    {"name": "rocket_local_gmm_bic3_crossfit", "mode": "gmm_diag_crossfit", "score_clusters": 3},
    {"name": "rocket_local_gmm_bic5_crossfit", "mode": "gmm_diag_crossfit", "score_clusters": 5},
    {"name": "rocket_local_kmeans3_crossfit", "mode": "kmeans_crossfit", "score_clusters": 3},
)


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path: Path, rows, fieldnames):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def baseline_row(base_row):
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_name": "baseline_exp93_nonpos_weak_alert_replace",
            "selector_name": "baseline_exp93_nonpos_weak_alert_replace",
            "score_source_name": "exp93",
            "score_family": "baseline",
            "threshold_method": "selector",
            "threshold_family": "existing_exp93_policy",
            "normal_state_mode": "baseline",
        }
    )
    return out


def score_pair(train_features, test_features, config):
    mode = config["mode"]
    if mode == "global_local_gap":
        return density_knn_score_pair(train_features, test_features, neighbors=3, mode="local_gap"), "global_local_gap"
    if len(train_features) < MIN_CLUSTER_TRAIN:
        return density_knn_score_pair(train_features, test_features, neighbors=3, mode="local_gap"), "global_fallback_small_train"
    if mode == "gmm_diag_crossfit":
        return gmm_crossfit_score_pair(train_features, test_features, config), f"gmm_diag_bic_up_to_{config['score_clusters']}_crossfit"
    if mode == "kmeans_crossfit":
        return kmeans_crossfit_score_pair(train_features, test_features, config), f"kmeans_{config['score_clusters']}_crossfit"
    raise ValueError(mode)


def result_rows(dataset_name, base_row):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    y_test = np.asarray(y_test, dtype=np.int64)
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    train_features, test_features = rocket_feature_pair(X_train, X_test, seq_len, num_kernels=256, seed_offset=115)
    train_features, test_features = scale_feature_pair(train_features, test_features)

    rows = [baseline_row(base_row)]
    for config in CONFIGS:
        (train_scores, test_scores), state_mode = score_pair(train_features, test_features, config)
        metrics = score_metrics(y_test, test_scores)
        for threshold_method, rate in THRESHOLDS:
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            evaluated = evaluate_threshold(y_test, test_scores, threshold, metrics)
            selected = set(np.flatnonzero(np.asarray(test_scores) > threshold).astype(int).tolist())
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset_name": dataset_name,
                    "family": parse_family(dataset_name),
                    "config_name": config["name"],
                    "selector_name": config["name"],
                    "score_source_name": "rocket256_features",
                    "score_family": "local_normal_state_score",
                    "threshold_method": threshold_method,
                    "threshold_family": "diagnostic_train_count_cap",
                    "normal_state_mode": state_mode,
                    "sequence_length": seq_len,
                    "train_normal_count": len(X_train),
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "q_effective": q_effective,
                    "cap_target": cap_target,
                    "threshold": threshold,
                    "selected_indices": format_indices(selected),
                    "train_exceed_count": int(np.sum(np.asarray(train_scores) > threshold)),
                    "train_exceed_rate": float(np.mean(np.asarray(train_scores) > threshold)),
                    **evaluated,
                }
            )
    return rows


def summarize(rows):
    out = []
    keys = sorted({(row["config_name"], row["threshold_method"]) for row in rows})
    for config_name, threshold_method in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["threshold_method"] == threshold_method]
        f1s = [as_float(row.get("f1")) for row in subset]
        families = {}
        for row in subset:
            families.setdefault(row["family"], []).append(as_float(row.get("f1")))
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "config_name": config_name,
                "threshold_method": threshold_method,
                "normal_state_mode": subset[0].get("normal_state_mode", ""),
                "num_datasets": len(subset),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": int(sum(value == 0.0 for value in f1s)),
                "mean_fp": float(np.mean([as_float(row.get("fp")) for row in subset])),
                "mean_tp": float(np.mean([as_float(row.get("tp")) for row in subset])),
                "mean_fn": float(np.mean([as_float(row.get("fn")) for row in subset])),
                "mean_auc_pr": float(np.mean([as_float(row.get("auc_pr")) for row in subset])),
                "mean_oracle_f1": float(np.mean([as_float(row.get("oracle_f1")) for row in subset])),
                "family_macro_f1": float(np.mean([np.mean(values) for values in families.values()])),
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


def run_experiment(dataset_limit=None):
    base_rows = {
        row["dataset_name"]: row
        for row in read_rows(EXP93_PATH)
        if row.get("selector_name") == EXP93_SELECTOR
    }
    if len(base_rows) != 1117:
        raise SystemExit(f"Expected 1117 Exp93 rows, got {len(base_rows)}")
    datasets = sorted(base_rows)
    if dataset_limit:
        datasets = datasets[: int(dataset_limit)]
    detail_path = results_path(EXPERIMENT_ID)
    summary_file = summary_path(EXPERIMENT_ID)
    for path in (detail_path, summary_file):
        if path.exists():
            path.unlink()

    rows = []
    fieldnames = None
    errors = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(result_rows, name, base_rows[name]): name for name in datasets}
        for completed, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                dataset_rows = future.result()
            except Exception as exc:
                errors.append((name, repr(exc)))
                print(f"ERROR dataset={name} error={exc!r}", flush=True)
                continue
            rows.extend(dataset_rows)
            if fieldnames is None:
                fieldnames = []
                for row in dataset_rows:
                    for key in row:
                        if key not in fieldnames:
                            fieldnames.append(key)
            append_rows(detail_path, dataset_rows, fieldnames)
            if completed % 25 == 0 or completed == len(datasets):
                write_csv(summary_file, summarize(rows))
                print(f"Progress: [{completed:4d}/{len(datasets):4d}] rows={len(rows)} last={name} errors={len(errors)}", flush=True)

    expected_rows = len(datasets) * (1 + len(CONFIGS) * len(THRESHOLDS))
    if errors or len(rows) != expected_rows:
        raise SystemExit(f"Coverage failure: rows={len(rows)}/{expected_rows} errors={errors[:10]}")
    write_csv(summary_file, summarize(rows))
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(datasets)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 115 local normal-state score probe")
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_limit)
