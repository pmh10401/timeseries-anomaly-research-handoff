import argparse
import csv
import logging
import os
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from run_experiment_26_rocket import make_kernels, rocket_transform
from run_experiment_29_train_normal_threshold_calibration import (
    knn_score_pair,
    train_false_positive_stats,
)
from run_rank_ensemble_calibration import align_series_lengths, sanitize_series, z_normalize
from run_rank_threshold_calibration import top_k_oracle_f1


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
MANIFEST_PATH = DATA_DIR / "experiment_33_evalset_reconstruction_manifest.csv"
RNG_SEED = 20260707
DEFAULT_WORKERS = int(os.environ.get("BALANCED_IMPROVEMENT_WORKERS", "4"))
MAX_TRAIN_REFERENCE = int(os.environ.get("BALANCED_IMPROVEMENT_MAX_TRAIN_REFERENCE", "1000"))
DEFAULT_EXCLUDED_DATASETS = {"CornellWhaleChallenge"}
WEAK_FAMILIES = {
    "ChlorineConcentration",
    "Computers",
    "EthanolLevel",
    "HandOutlines",
    "LargeKitchenAppliances",
    "MiddlePhalanxOutlineCorrect",
    "MoteStrain",
    "RefrigerationDevices",
    "ScreenType",
    "Yoga",
}


def experiment_paths(exp_id):
    return {
        "detail": DATA_DIR / f"{exp_id}_results.csv",
        "summary": DATA_DIR / f"{exp_id}_summary.csv",
        "log": DATA_DIR / f"{exp_id}.log",
    }


EXPERIMENT_SPECS = {
    "experiment_34_balanced_feature_capacity_sweep": {
        "label": "Exp 34 - Clean-balanced ROCKET capacity sweep",
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "adaptive_v0"],
        "configs": [
            {"name": "rocket_128_knn3", "kind": "rocket_knn", "num_kernels": 128, "neighbors": 3},
            {"name": "rocket_256_knn3", "kind": "rocket_knn", "num_kernels": 256, "neighbors": 3},
            {"name": "rocket_256_knn5", "kind": "rocket_knn", "num_kernels": 256, "neighbors": 5},
            {"name": "rocket_512_knn3", "kind": "rocket_knn", "num_kernels": 512, "neighbors": 3},
            {"name": "rocket_512_knn5", "kind": "rocket_knn", "num_kernels": 512, "neighbors": 5},
        ],
    },
    "experiment_35_balanced_threshold_policy_sweep": {
        "label": "Exp 35 - Clean-balanced threshold policy diagnosis",
        "thresholds": [
            "count_cap_1pct",
            "count_cap_2pct",
            "count_cap_3pct",
            "count_cap_4pct",
            "count_cap_5pct",
            "dynamic_1_over_n",
            "adaptive_v0",
            "family_guard_v1",
        ],
        "configs": [
            {"name": "rocket_256_knn3_threshold_probe", "kind": "rocket_knn", "num_kernels": 256, "neighbors": 3},
        ],
    },
    "experiment_36_balanced_score_normalization_sweep": {
        "label": "Exp 36 - Clean-balanced density-normalized KNN scores",
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "adaptive_v0"],
        "configs": [
            {"name": "rocket_256_knn3_raw", "kind": "rocket_knn", "num_kernels": 256, "neighbors": 3},
            {
                "name": "rocket_256_knn3_density_ratio",
                "kind": "density_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "mode": "ratio",
            },
            {
                "name": "rocket_512_knn5_density_ratio",
                "kind": "density_knn",
                "num_kernels": 512,
                "neighbors": 5,
                "mode": "ratio",
            },
            {
                "name": "rocket_256_knn3_local_gap",
                "kind": "density_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "mode": "local_gap",
            },
        ],
    },
    "experiment_37_balanced_bagged_rocket_ensemble": {
        "label": "Exp 37 - Clean-balanced bagged ROCKET ensemble",
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "adaptive_v0"],
        "configs": [
            {
                "name": "bagged_128_256_512_knn3_mean_rank",
                "kind": "bagged_rocket",
                "aggregation": "mean",
                "components": [
                    {"num_kernels": 128, "neighbors": 3, "seed_offset": 11},
                    {"num_kernels": 256, "neighbors": 3, "seed_offset": 17},
                    {"num_kernels": 512, "neighbors": 3, "seed_offset": 23},
                ],
            },
            {
                "name": "bagged_128_256_512_knn3_median_rank",
                "kind": "bagged_rocket",
                "aggregation": "median",
                "components": [
                    {"num_kernels": 128, "neighbors": 3, "seed_offset": 31},
                    {"num_kernels": 256, "neighbors": 3, "seed_offset": 37},
                    {"num_kernels": 512, "neighbors": 3, "seed_offset": 41},
                ],
            },
            {
                "name": "bagged_256_knn3_5_mean_rank",
                "kind": "bagged_rocket",
                "aggregation": "mean",
                "components": [
                    {"num_kernels": 256, "neighbors": 3, "seed_offset": 43},
                    {"num_kernels": 256, "neighbors": 5, "seed_offset": 47},
                    {"num_kernels": 256, "neighbors": 7, "seed_offset": 53},
                ],
            },
        ],
    },
    "experiment_38_balanced_actual_length_handling": {
        "label": "Exp 38 - Clean-balanced actual length handling",
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "adaptive_v0"],
        "configs": [
            {
                "name": "metadata_len_rocket_256_knn3",
                "kind": "rocket_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "target_len_strategy": "metadata",
            },
            {
                "name": "actual_median_len_rocket_256_knn3",
                "kind": "rocket_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "target_len_strategy": "actual_median",
            },
            {
                "name": "clean_test_median_len_rocket_256_knn3",
                "kind": "rocket_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "target_len_strategy": "clean_test_median",
            },
            {
                "name": "actual_max_cap2048_rocket_256_knn3",
                "kind": "rocket_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "target_len_strategy": "actual_max_cap2048",
            },
        ],
    },
    "experiment_39_balanced_candidate_retest": {
        "label": "Exp 39 - Clean-balanced candidate retest and family guard",
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "adaptive_v0", "family_guard_v1"],
        "configs": [
            {"name": "retest_rocket_512_knn5", "kind": "rocket_knn", "num_kernels": 512, "neighbors": 5},
            {
                "name": "retest_density_ratio_256_knn3",
                "kind": "density_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "mode": "ratio",
            },
            {
                "name": "retest_bagged_mean_rank",
                "kind": "bagged_rocket",
                "aggregation": "mean",
                "components": [
                    {"num_kernels": 128, "neighbors": 3, "seed_offset": 61},
                    {"num_kernels": 256, "neighbors": 3, "seed_offset": 67},
                    {"num_kernels": 512, "neighbors": 3, "seed_offset": 71},
                ],
            },
            {
                "name": "retest_patch_global_max_rank",
                "kind": "patch_rocket",
                "num_kernels": 128,
                "neighbors": 3,
                "aggregation": "global_max",
            },
            {
                "name": "retest_actual_median_len_rocket_256_knn3",
                "kind": "rocket_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "target_len_strategy": "actual_median",
            },
        ],
    },
}


def get_spec(exp_id):
    if exp_id not in EXPERIMENT_SPECS:
        raise SystemExit(f"Unknown balanced improvement experiment: {exp_id}")
    spec = dict(EXPERIMENT_SPECS[exp_id])
    spec["id"] = exp_id
    spec["paths"] = experiment_paths(exp_id)
    return spec


def parse_id_list(value):
    if value is None:
        return []
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []
    return [int(part) for part in text.split(";") if part]


def coerce_int(row, key):
    value = row.get(key)
    if value in (None, ""):
        return 0
    return int(float(value))


def coerce_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def load_manifest_rows(path=MANIFEST_PATH, manifest_variant="balanced_2pct", limit=None, excluded_dataset_names=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    excluded_dataset_names = set(excluded_dataset_names or [])
    rows = []
    id_fields = [
        "train_instance_ids",
        "clean_train_instance_ids",
        "clean_test_normal_instance_ids",
        "clean_test_anomaly_instance_ids",
    ]
    int_fields = [
        "series_length",
        "original_train_count",
        "clean_train_count",
        "original_test_normal_count",
        "original_test_anomaly_count",
        "clean_test_normal_count",
        "clean_test_anomaly_count",
        "clean_test_total_count",
        "removed_train_overlap_normals",
        "removed_test_duplicate_normals",
        "is_eligible",
    ]
    float_fields = ["actual_len_median", "actual_len_max", "clean_test_actual_len_median"]
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("manifest_variant") != manifest_variant:
                continue
            if row.get("dataset_name") in excluded_dataset_names:
                continue
            if coerce_int(row, "is_eligible") != 1:
                continue
            converted = dict(row)
            for field in id_fields:
                converted[field] = parse_id_list(row.get(field))
            for field in int_fields:
                converted[field] = coerce_int(row, field)
            for field in float_fields:
                converted[field] = coerce_float(row, field)
            rows.append(converted)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def target_len_for_manifest(manifest, strategy):
    if strategy == "metadata":
        value = manifest["series_length"]
    elif strategy == "actual_median":
        value = manifest["actual_len_median"]
    elif strategy == "clean_test_median":
        value = manifest["clean_test_actual_len_median"]
    elif strategy == "actual_max_cap2048":
        value = min(2048, manifest["actual_len_max"])
    else:
        raise ValueError(f"Unknown target length strategy: {strategy}")
    value = int(round(float(value)))
    if value <= 0:
        value = int(round(float(manifest["series_length"] or manifest["actual_len_median"] or 16)))
    return max(8, value)


def load_arrays_by_ids(db_path, instance_ids, target_len):
    if not instance_ids:
        return np.empty((0, target_len), dtype=np.float32)
    placeholders = ",".join("?" for _ in instance_ids)
    query = f"SELECT id, values_blob FROM instances WHERE id IN ({placeholders})"
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(query, instance_ids).fetchall()
    conn.close()
    by_id = {int(row[0]): sanitize_series(np.frombuffer(row[1], dtype=np.float32)) for row in rows}
    series = [by_id[instance_id] for instance_id in instance_ids if instance_id in by_id]
    return align_series_lengths(series, target_len).astype(np.float32)


def reference_train(X_train, seq_len, seed_offset):
    if len(X_train) <= MAX_TRAIN_REFERENCE:
        return X_train
    rng = np.random.default_rng(RNG_SEED + seed_offset + seq_len + len(X_train))
    idx = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
    return X_train[idx]


def rocket_feature_pair_seeded(X_train, X_test, seq_len, num_kernels, seed_offset=0):
    X_ref = reference_train(X_train, seq_len, seed_offset + num_kernels)
    kernels = make_kernels(seq_len, num_kernels=num_kernels, seed=RNG_SEED + seed_offset)
    train_features = rocket_transform(X_ref, kernels)
    test_features = rocket_transform(X_test, kernels)
    return train_features, test_features


def rank_normalize_scores(train_scores, test_scores):
    train_scores = np.asarray(train_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    finite_train = train_scores[np.isfinite(train_scores)]
    if len(finite_train) == 0:
        finite_train = np.array([0.0], dtype=np.float64)
    sorted_train = np.sort(finite_train)
    denom = float(len(sorted_train))
    train_rank = np.searchsorted(sorted_train, train_scores, side="right") / denom
    test_rank = np.searchsorted(sorted_train, test_scores, side="right") / denom
    return train_rank.astype(np.float64), test_rank.astype(np.float64)


def density_knn_score_pair(train_features, test_features, neighbors, mode):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    n_train = len(train_scaled)
    k_self = max(1, min(neighbors + 1, n_train))
    nn_train = NearestNeighbors(n_neighbors=k_self, metric="euclidean")
    nn_train.fit(train_scaled)
    train_distances, train_indices = nn_train.kneighbors(train_scaled)
    if train_distances.shape[1] > 1:
        base_train = train_distances[:, 1:].mean(axis=1)
        neighbor_indices = train_indices[:, 1:]
    else:
        base_train = train_distances[:, 0]
        neighbor_indices = train_indices
    local_radius = base_train + 1e-9
    local_train = np.mean(local_radius[neighbor_indices], axis=1) + 1e-9

    k_test = max(1, min(neighbors, n_train))
    nn_test = NearestNeighbors(n_neighbors=k_test, metric="euclidean")
    nn_test.fit(train_scaled)
    test_distances, test_indices = nn_test.kneighbors(test_scaled)
    base_test = test_distances.mean(axis=1)
    local_test = np.mean(local_radius[test_indices], axis=1) + 1e-9

    if mode == "ratio":
        return base_train / local_train, base_test / local_test
    if mode == "local_gap":
        scale = np.median(local_radius) + 1e-9
        return (base_train - local_train) / scale, (base_test - local_test) / scale
    raise ValueError(f"Unknown density KNN mode: {mode}")


def stack_and_aggregate(train_components, test_components, aggregation):
    train_matrix = np.vstack(train_components)
    test_matrix = np.vstack(test_components)
    if aggregation == "mean":
        return train_matrix.mean(axis=0), test_matrix.mean(axis=0)
    if aggregation == "median":
        return np.median(train_matrix, axis=0), np.median(test_matrix, axis=0)
    if aggregation == "max" or aggregation == "global_max":
        return train_matrix.max(axis=0), test_matrix.max(axis=0)
    raise ValueError(f"Unknown score aggregation: {aggregation}")


def bagged_rocket_score_pair(X_train, X_test, seq_len, config):
    train_components = []
    test_components = []
    for component in config["components"]:
        train_features, test_features = rocket_feature_pair_seeded(
            X_train,
            X_test,
            seq_len,
            component["num_kernels"],
            seed_offset=component.get("seed_offset", 0),
        )
        train_scores, test_scores = knn_score_pair(train_features, test_features, component["neighbors"])
        train_rank, test_rank = rank_normalize_scores(train_scores, test_scores)
        train_components.append(train_rank)
        test_components.append(test_rank)
    return stack_and_aggregate(train_components, test_components, config["aggregation"])


def crop_window(X, start_frac, width_frac):
    seq_len = X.shape[1]
    width = max(8, int(round(seq_len * width_frac)))
    width = min(seq_len, width)
    start = int(round((seq_len - width) * start_frac))
    start = max(0, min(start, seq_len - width))
    return X[:, start : start + width]


def patch_rocket_score_pair(X_train, X_test, seq_len, config):
    windows = [(0.0, 1.0)]
    if seq_len >= 32:
        windows.extend([(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)])
    train_components = []
    test_components = []
    for idx, (start_frac, width_frac) in enumerate(windows):
        X_train_patch = z_normalize(crop_window(X_train, start_frac, width_frac)).astype(np.float32)
        X_test_patch = z_normalize(crop_window(X_test, start_frac, width_frac)).astype(np.float32)
        patch_len = X_train_patch.shape[1]
        train_features, test_features = rocket_feature_pair_seeded(
            X_train_patch,
            X_test_patch,
            patch_len,
            config["num_kernels"],
            seed_offset=500 + idx,
        )
        train_scores, test_scores = knn_score_pair(train_features, test_features, config["neighbors"])
        train_rank, test_rank = rank_normalize_scores(train_scores, test_scores)
        train_components.append(train_rank)
        test_components.append(test_rank)
    return stack_and_aggregate(train_components, test_components, config["aggregation"])


def score_pair_for_config(X_train, X_test, target_len, config):
    kind = config["kind"]
    if kind == "rocket_knn":
        train_features, test_features = rocket_feature_pair_seeded(
            X_train,
            X_test,
            target_len,
            config["num_kernels"],
            seed_offset=config.get("seed_offset", 0),
        )
        return knn_score_pair(train_features, test_features, config["neighbors"])
    if kind == "density_knn":
        train_features, test_features = rocket_feature_pair_seeded(
            X_train,
            X_test,
            target_len,
            config["num_kernels"],
            seed_offset=config.get("seed_offset", 0),
        )
        return density_knn_score_pair(train_features, test_features, config["neighbors"], config["mode"])
    if kind == "bagged_rocket":
        return bagged_rocket_score_pair(X_train, X_test, target_len, config)
    if kind == "patch_rocket":
        return patch_rocket_score_pair(X_train, X_test, target_len, config)
    raise ValueError(f"Unknown config kind: {kind}")


def clean_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([0.0], dtype=np.float64)
    return scores


def count_cap_threshold(train_scores, rate):
    train_scores = clean_scores(train_scores)
    cap = int(np.floor(float(rate) * len(train_scores)))
    cap = max(0, min(cap, len(train_scores) - 1))
    threshold = float(np.sort(train_scores)[len(train_scores) - cap - 1])
    return threshold, cap / max(1, len(train_scores)), cap


def rate_for_threshold(method, manifest):
    if method.startswith("count_cap_") and method.endswith("pct"):
        pct = float(method.removeprefix("count_cap_").removesuffix("pct"))
        return pct / 100.0, "count_cap_rate"
    if method == "dynamic_1_over_n":
        return max(0.001, 1.0 / max(1, manifest["clean_test_total_count"])), "dynamic_count_cap"
    if method == "adaptive_v0":
        median_len = float(manifest["clean_test_actual_len_median"])
        guarded_mid_length = 513 <= median_len <= 1024
        rate = 0.03 if manifest["clean_test_total_count"] > 50 and not guarded_mid_length else 0.02
        return rate, "adaptive_count_cap"
    if method == "family_guard_v1":
        long_series = float(manifest["clean_test_actual_len_median"]) >= 512
        rate = 0.03 if manifest["family"] in WEAK_FAMILIES or long_series else 0.02
        return rate, "family_guard_count_cap"
    raise ValueError(f"Unknown threshold method: {method}")


def score_metrics(y_true, test_scores):
    precision, recall, _ = precision_recall_curve(y_true, test_scores)
    return {
        "auc_roc": roc_auc_score(y_true, test_scores),
        "auc_pr": auc(recall, precision),
        "oracle_f1": top_k_oracle_f1(y_true, test_scores),
    }


def evaluate_threshold(y_true, test_scores, threshold, metrics):
    preds = (test_scores > threshold).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return {
        "predicted_count": int(preds.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "f1": f1_score(y_true, preds, zero_division=0),
        "oracle_f1": metrics["oracle_f1"],
    }


def run_manifest_task(task):
    exp_id, db_path, manifest, train_variant = task
    spec = get_spec(exp_id)
    train_ids = (
        manifest["train_instance_ids"]
        if train_variant == "clean_test_only"
        else manifest["clean_train_instance_ids"]
    )
    test_normal_ids = manifest["clean_test_normal_instance_ids"]
    test_anomaly_ids = manifest["clean_test_anomaly_instance_ids"]
    test_ids = test_normal_ids + test_anomaly_ids
    y_test = np.array([0] * len(test_normal_ids) + [1] * len(test_anomaly_ids), dtype=np.int64)
    if len(train_ids) == 0 or len(test_ids) == 0 or len(np.unique(y_test)) < 2:
        return []

    arrays_by_len = {}
    rows = []
    for config in spec["configs"]:
        target_strategy = config.get("target_len_strategy", "metadata")
        target_len = target_len_for_manifest(manifest, target_strategy)
        if target_len not in arrays_by_len:
            X_train = load_arrays_by_ids(db_path, train_ids, target_len)
            X_test = load_arrays_by_ids(db_path, test_ids, target_len)
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            arrays_by_len[target_len] = (
                z_normalize(X_train).astype(np.float32),
                z_normalize(X_test).astype(np.float32),
            )
        X_train, X_test = arrays_by_len[target_len]
        train_scores, test_scores = score_pair_for_config(X_train, X_test, target_len, config)
        metrics = score_metrics(y_test, test_scores)
        for method in spec["thresholds"]:
            rate, threshold_family = rate_for_threshold(method, manifest)
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "dataset_name": manifest["dataset_name"],
                    "family": manifest["family"],
                    "manifest_variant": manifest["manifest_variant"],
                    "train_variant": train_variant,
                    "config_name": config["name"],
                    "score_family": config["kind"],
                    "num_kernels": config.get("num_kernels", ""),
                    "knn_neighbors": config.get("neighbors", ""),
                    "target_len_strategy": target_strategy,
                    "threshold_method": method,
                    "threshold_family": threshold_family,
                    "sequence_length": target_len,
                    "actual_len_median": manifest["actual_len_median"],
                    "actual_len_max": manifest["actual_len_max"],
                    "clean_test_actual_len_median": manifest["clean_test_actual_len_median"],
                    "original_train_count": manifest["original_train_count"],
                    "clean_train_count": manifest["clean_train_count"],
                    "train_score_count": len(train_scores),
                    "original_test_normal_count": manifest["original_test_normal_count"],
                    "original_test_anomaly_count": manifest["original_test_anomaly_count"],
                    "clean_test_normal_count": len(test_normal_ids),
                    "clean_test_anomaly_count": len(test_anomaly_ids),
                    "clean_test_total_count": len(test_ids),
                    "removed_train_overlap_normals": manifest["removed_train_overlap_normals"],
                    "removed_test_duplicate_normals": manifest["removed_test_duplicate_normals"],
                    "q_effective": q_effective,
                    "cap_target": cap_target,
                    "threshold": threshold,
                    "train_exceed_count": train_exceed_count,
                    "train_exceed_rate": train_exceed_rate,
                    **evaluate_threshold(y_test, test_scores, threshold, metrics),
                }
            )
    return rows


def append_rows(path, rows, fieldnames):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    keys = sorted(
        {
            (
                row["experiment_id"],
                row["manifest_variant"],
                row["train_variant"],
                row["config_name"],
                row["threshold_method"],
            )
            for row in rows
        }
    )
    for exp_id, manifest_variant, train_variant, config_name, method in keys:
        subset = [
            row
            for row in rows
            if row["experiment_id"] == exp_id
            and row["manifest_variant"] == manifest_variant
            and row["train_variant"] == train_variant
            and row["config_name"] == config_name
            and row["threshold_method"] == method
        ]
        f1s = [float(row["f1"]) for row in subset]
        by_family = defaultdict(list)
        for row in subset:
            by_family[row["family"]].append(float(row["f1"]))
        family_means = [float(np.mean(values)) for values in by_family.values()]
        summary.append(
            {
                "experiment_id": exp_id,
                "manifest_variant": manifest_variant,
                "train_variant": train_variant,
                "config_name": config_name,
                "threshold_method": method,
                "num_datasets": len(subset),
                "num_families": len(by_family),
                "mean_auc_roc": float(np.mean([float(row["auc_roc"]) for row in subset])),
                "mean_auc_pr": float(np.mean([float(row["auc_pr"]) for row in subset])),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "p25_f1": float(np.percentile(f1s, 25)),
                "zero_f1_count": sum(1 for value in f1s if value == 0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "family_macro_f1": float(np.mean(family_means)) if family_means else 0.0,
                "family_p25_f1": float(np.percentile(family_means, 25)) if family_means else 0.0,
                "mean_predicted_count": float(np.mean([int(row["predicted_count"]) for row in subset])),
                "mean_tp": float(np.mean([int(row["tp"]) for row in subset])),
                "mean_fp": float(np.mean([int(row["fp"]) for row in subset])),
                "mean_fn": float(np.mean([int(row["fn"]) for row in subset])),
                "mean_train_exceed_rate": float(np.mean([float(row["train_exceed_rate"]) for row in subset])),
                "mean_oracle_f1": float(np.mean([float(row["oracle_f1"]) for row in subset])),
            }
        )
    return sorted(summary, key=lambda row: (row["mean_f1"], row["mean_auc_pr"]), reverse=True)


def make_logger(exp_id, log_path):
    logger = logging.getLogger(exp_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler())
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    return logger


def run_experiment(exp_id, args):
    spec = get_spec(exp_id)
    paths = spec["paths"]
    logger = make_logger(exp_id, paths["log"])
    if not args.keep_existing:
        for path in [paths["detail"], paths["summary"]]:
            if path.exists():
                path.unlink()
    excluded_dataset_names = set(args.exclude_dataset or [])
    if not args.include_oversized:
        excluded_dataset_names.update(DEFAULT_EXCLUDED_DATASETS)
    manifests = load_manifest_rows(
        args.manifest_path,
        manifest_variant=args.manifest_variant,
        limit=args.dataset_limit,
        excluded_dataset_names=excluded_dataset_names,
    )
    train_variants = args.train_variant or ["clean_test_only"]
    tasks = []
    for manifest in manifests:
        for train_variant in train_variants:
            tasks.append((exp_id, str(args.db_path), manifest, train_variant))
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]

    logger.info(
        "Starting %s on %d balanced-clean tasks with %d workers. excluded=%s",
        exp_id,
        len(tasks),
        args.workers,
        sorted(excluded_dataset_names),
    )
    detail_rows = []
    fieldnames = None
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_manifest_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            manifest = task[2]
            try:
                rows = future.result()
            except Exception as exc:
                logger.error("Error evaluating %s: %s", manifest["dataset_name"], exc, exc_info=True)
                rows = []
            completed += 1
            if rows:
                detail_rows.extend(rows)
                fieldnames = fieldnames or list(rows[0].keys())
                append_rows(paths["detail"], rows, fieldnames)
            if completed % 10 == 0 or completed == len(tasks):
                summary_rows = summarize(detail_rows) if detail_rows else []
                write_csv(paths["summary"], summary_rows)
                best = summary_rows[0] if summary_rows else None
                if best:
                    logger.info(
                        "Progress: [%4d/%4d] rows=%d | best=%s/%s meanF1=%.4f medianF1=%.4f aucPR=%.4f oracle=%.4f zero=%d fp=%.2f",
                        completed,
                        len(tasks),
                        len(detail_rows),
                        best["config_name"],
                        best["threshold_method"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["mean_auc_pr"],
                        best["mean_oracle_f1"],
                        best["zero_f1_count"],
                        best["mean_fp"],
                    )
    logger.info("%s finished.", exp_id)
    return detail_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run clean-balanced improvement experiments.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--manifest-variant", default="balanced_2pct")
    parser.add_argument("--include-oversized", action="store_true", help="Include oversized datasets such as CornellWhaleChallenge.")
    parser.add_argument("--exclude-dataset", action="append", help="Additional dataset name to exclude. Repeatable.")
    parser.add_argument(
        "--train-variant",
        action="append",
        choices=["clean_test_only", "clean_train_test"],
        help="Default is clean_test_only.",
    )
    return parser.parse_args(argv)


def main_for_experiment(exp_id, argv=None):
    args = parse_args(argv)
    run_experiment(exp_id, args)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one clean-balanced improvement experiment by id.")
    parser.add_argument("experiment_id", choices=sorted(EXPERIMENT_SPECS))
    args, rest = parser.parse_known_args(argv)
    main_for_experiment(args.experiment_id, rest)


if __name__ == "__main__":
    main()
