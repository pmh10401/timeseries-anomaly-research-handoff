from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment_104_score_dimensionality_sweep import (
    DIMENSIONS,
    EXP93_PATH,
    EXP93_SELECTOR,
    SOURCE_TEMPLATES,
    prepare_dataset,
    rank_normalize_against_train,
    read_rows,
    score_sources,
    source_config,
    top_indices,
    write_csv,
)
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


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_105_score_combination_methods"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP105_WORKERS", "4"))

COUNT_CAP_THRESHOLDS = (("count_cap_2pct", 0.02), ("count_cap_3pct", 0.03))
AGREEMENT_RATE = 0.02


def component_score_map(dataset_name):
    record, target_len, X_train_raw, X_test_raw, X_train_z, X_test_z, y_test = prepare_dataset(dataset_name)
    scores = {}
    ranks = {}
    exceeds = {}
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
            key = f"{source_name}_pca{dim}"
            scores[key] = (np.asarray(train_scores, dtype=np.float64), np.asarray(test_scores, dtype=np.float64))
            ranks[key] = rank_normalize_against_train(train_scores, test_scores)
            threshold, _, _ = count_cap_threshold(train_scores, AGREEMENT_RATE)
            exceeds[key] = (
                (np.asarray(train_scores) > threshold).astype(np.float64),
                (np.asarray(test_scores) > threshold).astype(np.float64),
            )
    return record, y_test, scores, ranks, exceeds


def weighted_rank(keys, ranks, weights):
    train_parts = []
    test_parts = []
    total = float(sum(weights))
    for key, weight in zip(keys, weights):
        train_rank, test_rank = ranks[key]
        train_parts.append(train_rank * weight / total)
        test_parts.append(test_rank * weight / total)
    return np.sum(train_parts, axis=0), np.sum(test_parts, axis=0)


def combine_for_keys(prefix, keys, ranks, exceeds):
    rank_train = [ranks[key][0] for key in keys]
    rank_test = [ranks[key][1] for key in keys]
    exceed_train = [exceeds[key][0] for key in keys]
    exceed_test = [exceeds[key][1] for key in keys]
    out = {}
    out[f"{prefix}_rank_mean"] = {
        "combination_method": "rank_mean",
        "train_scores": np.mean(rank_train, axis=0),
        "test_scores": np.mean(rank_test, axis=0),
    }
    out[f"{prefix}_rank_max"] = {
        "combination_method": "rank_max",
        "train_scores": np.max(rank_train, axis=0),
        "test_scores": np.max(rank_test, axis=0),
    }
    out[f"{prefix}_rank_min"] = {
        "combination_method": "rank_min",
        "train_scores": np.min(rank_train, axis=0),
        "test_scores": np.min(rank_test, axis=0),
    }
    if len(keys) == 3:
        out[f"{prefix}_weighted_50_35_15"] = {
            "combination_method": "weighted_50_35_15",
            "train_scores": weighted_rank(keys, ranks, [0.50, 0.35, 0.15])[0],
            "test_scores": weighted_rank(keys, ranks, [0.50, 0.35, 0.15])[1],
        }
        out[f"{prefix}_weighted_20_30_50"] = {
            "combination_method": "weighted_20_30_50",
            "train_scores": weighted_rank(keys, ranks, [0.20, 0.30, 0.50])[0],
            "test_scores": weighted_rank(keys, ranks, [0.20, 0.30, 0.50])[1],
        }
        agreement_train = np.sum(exceed_train, axis=0)
        agreement_test = np.sum(exceed_test, axis=0)
        mean_train = np.mean(rank_train, axis=0)
        mean_test = np.mean(rank_test, axis=0)
        out[f"{prefix}_agreement_2of3"] = {
            "combination_method": "agreement_2of3",
            "fixed_threshold": 1.5,
            "train_scores": agreement_train + 0.01 * mean_train,
            "test_scores": agreement_test + 0.01 * mean_test,
        }
        out[f"{prefix}_agreement_3of3"] = {
            "combination_method": "agreement_3of3",
            "fixed_threshold": 2.5,
            "train_scores": agreement_train + 0.01 * mean_train,
            "test_scores": agreement_test + 0.01 * mean_test,
        }
    return out


def combination_specs(ranks, exceeds):
    out = {}
    for source_name in SOURCE_TEMPLATES:
        keys = [f"{source_name}_pca{dim}" for dim in DIMENSIONS]
        for name, spec in combine_for_keys(source_name, keys, ranks, exceeds).items():
            spec.update(
                {
                    "score_source_name": source_name,
                    "score_dim": "64+128+256",
                    "component_scores": ";".join(keys),
                }
            )
            out[name] = spec
    all_keys = [f"{source_name}_pca{dim}" for source_name in SOURCE_TEMPLATES for dim in DIMENSIONS]
    for name, spec in combine_for_keys("spectrogram_glcm_rp_all_dims", all_keys, ranks, exceeds).items():
        spec.update(
            {
                "score_source_name": "spectrogram+glcm_rp",
                "score_dim": "all",
                "component_scores": ";".join(all_keys),
            }
        )
        out[name] = spec
    return out


def row_for_scores(dataset_name, record, y_test, config_name, spec, threshold_method, rate=None):
    train_scores = np.asarray(spec["train_scores"], dtype=np.float64)
    test_scores = np.asarray(spec["test_scores"], dtype=np.float64)
    if threshold_method == "fixed_agreement":
        threshold = float(spec["fixed_threshold"])
        q_effective = float((train_scores > threshold).mean()) if len(train_scores) else 0.0
        cap_target = int((train_scores > threshold).sum())
    else:
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
        "score_source_name": spec["score_source_name"],
        "score_family": "score_combination_methods",
        "score_dim": spec["score_dim"],
        "combination_method": spec["combination_method"],
        "component_scores": spec["component_scores"],
        "threshold_method": threshold_method,
        "threshold_family": "fixed_agreement" if threshold_method == "fixed_agreement" else "count_cap_rate",
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
            "combination_method": "baseline",
            "component_scores": "exp93",
            "threshold_method": "selector",
            "threshold_family": "selector",
        }
    )
    return out


def run_dataset(args):
    dataset_name, base_row = args
    record, y_test, _, ranks, exceeds = component_score_map(dataset_name)
    combos = combination_specs(ranks, exceeds)
    rows = [baseline_row(dataset_name, base_row)]
    for config_name, spec in combos.items():
        if "fixed_threshold" in spec:
            rows.append(row_for_scores(dataset_name, record, y_test, config_name, spec, "fixed_agreement"))
        else:
            for threshold_method, rate in COUNT_CAP_THRESHOLDS:
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
                "combination_method": subset[0].get("combination_method"),
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
