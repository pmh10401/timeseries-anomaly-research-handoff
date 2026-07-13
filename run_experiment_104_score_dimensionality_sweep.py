from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_26_rocket import load_dataset_names
from run_experiment_29_train_normal_threshold_calibration import train_false_positive_stats
from run_experiment_40_original_score_normalization_sweep import (
    count_cap_threshold,
    evaluate_threshold,
    parse_family,
    score_metrics,
)
from run_experiment_60_62_rocket_imaging_selector_variants import results_path, summary_path
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices
from run_model_hard_research_experiments import (
    prepare_series_pair_for_scale,
    score_pair_for_config as imaging_score_pair_for_config,
)
from run_original_improvement_experiment import DB_PATH, load_original_record, target_len_for_record
from run_rank_ensemble_calibration import align_series_lengths, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_104_score_dimensionality_sweep"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP104_WORKERS", "4"))

DIMENSIONS = (64, 128, 256)
THRESHOLDS = (("count_cap_2pct", 0.02), ("count_cap_3pct", 0.03))

SOURCE_TEMPLATES = {
    "spectrogram": {
        "kind": "imaging_knn",
        "image": "spectrogram",
        "series_scale": "train_global_minmax_clip",
        "size": 32,
        "neighbors": 3,
    },
    "glcm_rp": {
        "kind": "imaging_knn",
        "image": "rp",
        "feature_extractor": "glcm",
        "size": 32,
        "neighbors": 3,
    },
}


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


def top_indices(scores, count=12):
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    return [int(idx) for idx in order[: min(count, len(order))]]


def rank_normalize_against_train(train_scores, test_scores):
    train_scores = np.asarray(train_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    finite_train = train_scores[np.isfinite(train_scores)]
    if len(finite_train) == 0:
        finite_train = np.array([0.0], dtype=np.float64)
    sorted_train = np.sort(finite_train)

    def transform(values):
        values = np.asarray(values, dtype=np.float64)
        safe = np.nan_to_num(values, nan=np.nanmedian(finite_train), posinf=finite_train.max(), neginf=finite_train.min())
        return np.searchsorted(sorted_train, safe, side="right") / max(1, len(sorted_train))

    return transform(train_scores), transform(test_scores)


def combine_score_pairs(score_pairs):
    train_parts = []
    test_parts = []
    for train_scores, test_scores in score_pairs:
        train_rank, test_rank = rank_normalize_against_train(train_scores, test_scores)
        train_parts.append(train_rank)
        test_parts.append(test_rank)
    return np.mean(train_parts, axis=0), np.mean(test_parts, axis=0)


def source_config(source_name, dim):
    config = dict(SOURCE_TEMPLATES[source_name])
    config["pca"] = int(dim)
    config["name"] = f"{source_name}_32_pca{dim}_knn3"
    return config


def prepare_dataset(dataset_name):
    record = load_original_record(dataset_name, DB_PATH)
    target_len = min(max(8, target_len_for_record(record, "actual_median")), 2048)
    X_train_raw = align_series_lengths(record["train_series"], target_len)
    X_test_raw = align_series_lengths(record["test_series"], target_len)
    X_train_z = z_normalize(X_train_raw).astype(np.float32)
    X_test_z = z_normalize(X_test_raw).astype(np.float32)
    return record, target_len, X_train_raw, X_test_raw, X_train_z, X_test_z, np.asarray(record["y_test"], dtype=np.int64)


def score_sources(dataset_name):
    record, target_len, X_train_raw, X_test_raw, X_train_z, X_test_z, y_test = prepare_dataset(dataset_name)
    score_map = {}
    for source_name in SOURCE_TEMPLATES:
        for dim in DIMENSIONS:
            config = source_config(source_name, dim)
            X_train, X_test = prepare_series_pair_for_scale(
                config.get("series_scale", "per_series_z"),
                X_train_raw,
                X_test_raw,
                X_train_z,
                X_test_z,
            )
            train_scores, test_scores = imaging_score_pair_for_config(X_train, X_test, target_len, config, record)
            score_map[f"{source_name}_pca{dim}"] = {
                "source_name": source_name,
                "score_dim": dim,
                "train_scores": train_scores,
                "test_scores": test_scores,
                "component_scores": f"{source_name}_pca{dim}",
            }
    for source_name in SOURCE_TEMPLATES:
        keys = [f"{source_name}_pca{dim}" for dim in DIMENSIONS]
        train_scores, test_scores = combine_score_pairs(
            [(score_map[key]["train_scores"], score_map[key]["test_scores"]) for key in keys]
        )
        score_map[f"{source_name}_pca64_128_256_rank_mean"] = {
            "source_name": source_name,
            "score_dim": "64+128+256",
            "train_scores": train_scores,
            "test_scores": test_scores,
            "component_scores": ";".join(keys),
        }
    all_keys = [f"{source_name}_pca{dim}" for source_name in SOURCE_TEMPLATES for dim in DIMENSIONS]
    train_scores, test_scores = combine_score_pairs(
        [(score_map[key]["train_scores"], score_map[key]["test_scores"]) for key in all_keys]
    )
    score_map["spectrogram_glcm_rp_all_dims_rank_mean"] = {
        "source_name": "spectrogram+glcm_rp",
        "score_dim": "all",
        "train_scores": train_scores,
        "test_scores": test_scores,
        "component_scores": ";".join(all_keys),
    }
    return record, y_test, score_map


def row_for_scores(dataset_name, record, y_test, config_name, spec, threshold_method, rate):
    train_scores = np.asarray(spec["train_scores"], dtype=np.float64)
    test_scores = np.asarray(spec["test_scores"], dtype=np.float64)
    threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
    train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
    metrics = score_metrics(y_test, test_scores)
    evaluated = evaluate_threshold(y_test, test_scores, threshold, metrics)
    selected = np.where(test_scores > threshold)[0]
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset_name": dataset_name,
        "family": parse_family(dataset_name),
        "config_name": config_name,
        "score_source_name": spec["source_name"],
        "score_family": "score_dimensionality_sweep",
        "score_dim": spec["score_dim"],
        "component_scores": spec["component_scores"],
        "threshold_method": threshold_method,
        "threshold_family": "count_cap_rate",
        "sequence_length": len(record["test_series"][0]) if len(record["test_series"]) else "",
        "test_size": len(y_test),
        "anomaly_count": int(np.sum(y_test)),
        "train_score_count": len(train_scores),
        "q_effective": q_effective,
        "cap_target": cap_target,
        "threshold": threshold,
        "train_exceed_count": train_exceed_count,
        "train_exceed_rate": train_exceed_rate,
        "top_score_indices": format_indices(top_indices(test_scores)),
        "selected_indices": format_indices(selected),
        **evaluated,
    }


def baseline_row(dataset_name, base_row):
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_name": "baseline_exp93_nonpos_weak_alert_replace",
            "score_source_name": "exp93",
            "score_family": "baseline",
            "score_dim": "baseline",
            "component_scores": "exp93",
            "threshold_method": "selector",
            "threshold_family": "selector",
        }
    )
    return out


def run_dataset(args):
    dataset_name, base_row = args
    record, y_test, score_map = score_sources(dataset_name)
    rows = [baseline_row(dataset_name, base_row)]
    for config_name, spec in score_map.items():
        for threshold_method, rate in THRESHOLDS:
            rows.append(row_for_scores(dataset_name, record, y_test, config_name, spec, threshold_method, rate))
    return rows


def summarize(rows):
    out = []
    keys = sorted({(row["config_name"], row["threshold_method"]) for row in rows})
    for config_name, threshold_method in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["threshold_method"] == threshold_method]
        f1s = [as_float(row.get("f1")) for row in subset]
        tp_sum = sum(as_float(row.get("tp")) for row in subset)
        fp_sum = sum(as_float(row.get("fp")) for row in subset)
        families = {}
        for row in subset:
            families.setdefault(row["family"], []).append(as_float(row.get("f1")))
        family_means = [float(np.mean(values)) for values in families.values()]
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "config_name": config_name,
                "threshold_method": threshold_method,
                "score_source_name": subset[0].get("score_source_name"),
                "score_dim": subset[0].get("score_dim"),
                "component_scores": subset[0].get("component_scores"),
                "num_datasets": len(subset),
                "num_families": len(families),
                "mean_auc_roc": float(np.mean([as_float(row.get("auc_roc")) for row in subset])),
                "mean_auc_pr": float(np.mean([as_float(row.get("auc_pr")) for row in subset])),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(1 for value in f1s if value == 0.0),
                "family_macro_f1": float(np.mean(family_means)) if family_means else 0.0,
                "mean_predicted_count": float(np.mean([as_float(row.get("predicted_count")) for row in subset])),
                "mean_tp": float(np.mean([as_float(row.get("tp")) for row in subset])),
                "mean_fp": float(np.mean([as_float(row.get("fp")) for row in subset])),
                "mean_fn": float(np.mean([as_float(row.get("fn")) for row in subset])),
                "alert_precision": tp_sum / max(1.0, tp_sum + fp_sum),
                "mean_train_exceed_rate": float(np.mean([as_float(row.get("train_exceed_rate")) for row in subset])),
                "mean_oracle_f1": float(np.mean([as_float(row.get("oracle_f1")) for row in subset])),
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], row["mean_auc_pr"]), reverse=True)


def run_experiment(dataset_limit=None):
    base_rows = {
        row["dataset_name"]: row
        for row in read_rows(EXP93_PATH)
        if row.get("selector_name") == EXP93_SELECTOR
    }
    names = [name for name in load_dataset_names() if name in base_rows]
    if dataset_limit:
        names = names[: int(dataset_limit)]
    tasks = [(name, base_rows[name]) for name in names]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_dataset, task): task[0] for task in tasks}
        for idx, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            rows.extend(future.result())
            if idx % 25 == 0 or idx == len(tasks):
                print(f"{EXPERIMENT_ID} progress {idx}/{len(tasks)} last={name}", flush=True)
    write_csv(results_path(EXPERIMENT_ID), rows)
    summary = summarize(rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_limit)
