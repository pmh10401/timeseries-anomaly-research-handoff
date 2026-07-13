from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler

from run_experiment_60_62_rocket_imaging_selector_variants import (
    evaluate_indices,
    results_path,
    summary_path,
)
from run_experiment_89_74d_with_exp84_candidate import as_float, format_indices, parse_indices
from run_model_hard_research_experiments import choose_shapelets, resample_rows, shapelet_distance_features
from run_rank_ensemble_calibration import load_dataset_data, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
EXPERIMENT_ID = "experiment_101_shapelet_normal_prototype"
EXP93_PATH = DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv"
EXP97_PATH = DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_results.csv"
EXP93_SELECTOR = "nonpos_weak_alert_replace"
STDOUT_LOG = DATA_DIR / f"{EXPERIMENT_ID}_stdout.log"
WORKERS = int(os.environ.get("EXP101_WORKERS", "4"))

SHAPE_FAMILIES = {
    "Adiac",
    "ArrowHead",
    "BeetleFly",
    "BirdChicken",
    "Fish",
    "HandOutlines",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "PhalangesOutlinesCorrect",
    "ShapesAll",
    "ShapeletSim",
    "SwedishLeaf",
    "OSULeaf",
    "Plane",
    "Trace",
    "Worms",
    "WormsTwoClass",
    "WordSynonyms",
    "FiftyWords",
}


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(name: str) -> int:
    return 20260709 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def load_research_targets():
    return {
        row["dataset_name"]
        for row in read_rows(EXP97_PATH)
        if row.get("feature_need_primary") == "D_shapelet_prototype_feature"
    }


def robust_score_pair(train_features, test_features, top_k=None):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    center = np.median(train_scaled, axis=0)
    spread = np.median(np.abs(train_scaled - center), axis=0) + 1e-6
    train_dev = np.abs(train_scaled - center) / spread
    test_dev = np.abs(test_scaled - center) / spread
    if top_k is None:
        return train_dev.mean(axis=1), test_dev.mean(axis=1)
    top_k = max(1, min(int(top_k), train_dev.shape[1]))
    split_at = train_dev.shape[1] - top_k
    train_scores = np.mean(np.partition(train_dev, split_at, axis=1)[:, -top_k:], axis=1)
    test_scores = np.mean(np.partition(test_dev, split_at, axis=1)[:, -top_k:], axis=1)
    return train_scores, test_scores


def select(scores, train_scores, q, cap):
    threshold = float(np.quantile(train_scores, q))
    idxs = np.flatnonzero(np.asarray(scores) > threshold)
    order = sorted(idxs, key=lambda idx: scores[idx], reverse=True)
    return set(int(i) for i in order[:cap]), threshold


def shapelet_feature_pair(dataset_name, X_train, X_test):
    target_len = min(512, max(16, X_train.shape[1]))
    X_train_s = z_normalize(resample_rows(X_train, target_len)).astype(np.float32)
    X_test_s = z_normalize(resample_rows(X_test, target_len)).astype(np.float32)
    lengths = [length for length in [8, 16, 32, 64] if length <= target_len]
    shapelets = choose_shapelets(X_train_s, 32, lengths, stable_seed(dataset_name))
    return (
        shapelet_distance_features(X_train_s, shapelets),
        shapelet_distance_features(X_test_s, shapelets),
    )


def passthrough(base_row, selector, reason, extras=None):
    out = dict(base_row)
    out["experiment_id"] = EXPERIMENT_ID
    out["selector_name"] = selector
    out["config_name"] = selector
    out["selector_reason"] = reason
    out["threshold_method"] = "selector"
    if extras:
        out.update(extras)
    return out


def row_with_metrics(name, family, y_test, base_row, selector, indices, scores, reason, extras=None):
    metrics = evaluate_indices(y_test, scores, indices)
    out = dict(base_row)
    out.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset_name": name,
            "family": family,
            "config_name": selector,
            "selector_name": selector,
            "selector_reason": reason,
            "score_source_name": "shapelet_normal_prototype",
            "threshold_method": "selector",
            "score_family": "shapelet_normal_prototype",
            "test_size": len(y_test),
            "anomaly_count": int(np.sum(y_test)),
            "selected_indices": format_indices(indices),
            "predicted_count": metrics["predicted_count"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "oracle_f1": metrics["oracle_f1"],
        }
    )
    if extras:
        out.update(extras)
    return out


def run_dataset(args):
    name, base_row, research_target = args
    family = base_row["family"]
    family_target = family in SHAPE_FAMILIES
    selectors = [
        "baseline_exp93_operating",
        "research_shapelet_q98_cap2",
        "train_family_shapelet_q99_cap1",
        "train_family_shapelet_q98_cap2",
        "train_family_shapelet_topk4_q98_cap2",
    ]
    if not (research_target or family_target):
        return [passthrough(base_row, s, "non-target passthrough", {"shapelet_target": 0}) for s in selectors]

    X_train, X_test, y_test = load_dataset_data(name)
    train_features, test_features = shapelet_feature_pair(name, X_train, X_test)
    train_mean, test_mean = robust_score_pair(train_features, test_features, None)
    train_top4, test_top4 = robust_score_pair(train_features, test_features, 4)
    base_indices = parse_indices(base_row.get("selected_indices"))
    rows = [
        passthrough(
            base_row,
            "baseline_exp93_operating",
            "control: Exp93 operating default",
            {"shapelet_target": int(family_target), "research_target": int(research_target)},
        )
    ]
    configs = [
        ("research_shapelet_q98_cap2", research_target, train_mean, test_mean, 0.98, 2),
        ("train_family_shapelet_q99_cap1", family_target, train_mean, test_mean, 0.99, 1),
        ("train_family_shapelet_q98_cap2", family_target, train_mean, test_mean, 0.98, 2),
        ("train_family_shapelet_topk4_q98_cap2", family_target, train_top4, test_top4, 0.98, 2),
    ]
    for selector, enabled, train_scores, test_scores, q, cap in configs:
        if enabled:
            idxs, threshold = select(test_scores, train_scores, q, cap)
        else:
            idxs, threshold = base_indices, float("nan")
        rows.append(
            row_with_metrics(
                name,
                family,
                y_test,
                base_row,
                selector,
                idxs,
                test_scores,
                "shapelet normal prototype robust score candidate",
                {
                    "shapelet_target": int(family_target),
                    "research_target": int(research_target),
                    "shapelet_threshold": threshold,
                },
            )
        )
    return rows


def summarize(rows):
    out = []
    for selector in sorted({row["selector_name"] for row in rows}):
        subset = [row for row in rows if row["selector_name"] == selector]
        target = [row for row in subset if as_float(row.get("shapelet_target")) > 0]
        research_target = [row for row in subset if as_float(row.get("research_target")) > 0]
        vals = lambda key, rows_: [as_float(row.get(key)) for row in rows_]
        f1s = vals("f1", subset)
        target_f1s = vals("f1", target)
        research_f1s = vals("f1", research_target)
        out.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "selector_name": selector,
                "config_name": selector,
                "threshold_method": "selector",
                "num_datasets": len(subset),
                "num_families": len({row["family"] for row in subset}),
                "mean_auc_roc": float(np.mean(vals("auc_roc", subset))),
                "mean_auc_pr": float(np.mean(vals("auc_pr", subset))),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "p25_f1": float(np.percentile(f1s, 25)),
                "zero_f1_count": sum(1 for value in f1s if value == 0.0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "mean_predicted_count": float(np.mean(vals("predicted_count", subset))),
                "mean_tp": float(np.mean(vals("tp", subset))),
                "mean_fp": float(np.mean(vals("fp", subset))),
                "mean_fn": float(np.mean(vals("fn", subset))),
                "mean_oracle_f1": float(np.mean(vals("oracle_f1", subset))),
                "target_datasets": len(target),
                "target_mean_f1": float(np.mean(target_f1s)) if target_f1s else 0.0,
                "target_zero_f1_count": sum(1 for value in target_f1s if value == 0.0) if target_f1s else 0,
                "target_mean_fp": float(np.mean(vals("fp", target))) if target else 0.0,
                "target_mean_tp": float(np.mean(vals("tp", target))) if target else 0.0,
                "research_target_datasets": len(research_target),
                "research_target_mean_f1": float(np.mean(research_f1s)) if research_f1s else 0.0,
                "research_target_zero_f1_count": sum(1 for value in research_f1s if value == 0.0) if research_f1s else 0,
                "research_target_mean_fp": float(np.mean(vals("fp", research_target))) if research_target else 0.0,
                "research_target_mean_tp": float(np.mean(vals("tp", research_target))) if research_target else 0.0,
            }
        )
    return sorted(out, key=lambda row: (row["mean_f1"], -row["mean_fp"]), reverse=True)


def run_experiment(dataset_limit=None):
    exp93 = {row["dataset_name"]: row for row in read_rows(EXP93_PATH) if row.get("selector_name") == EXP93_SELECTOR}
    research_targets = load_research_targets()
    names = sorted(exp93)
    if dataset_limit:
        names = names[: int(dataset_limit)]
    tasks = [(name, exp93[name], name in research_targets) for name in names]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_dataset, task): task[0] for task in tasks}
        for idx, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if idx % 50 == 0 or idx == len(tasks):
                print(f"{EXPERIMENT_ID} progress {idx}/{len(tasks)}", flush=True)
    write_csv(results_path(EXPERIMENT_ID), rows)
    summary = summarize(rows)
    write_csv(summary_path(EXPERIMENT_ID), summary)
    STDOUT_LOG.write_text(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}\n{summary[0] if summary else ''}\n")
    print(f"{EXPERIMENT_ID} finished rows={len(rows)} datasets={len(names)}")
    if summary:
        print(summary[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-limit", type=int, default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_limit)
