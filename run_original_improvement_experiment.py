import argparse
import copy
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

from run_balanced_improvement_experiment import (
    EXPERIMENT_SPECS as BALANCED_SPECS,
    WEAK_FAMILIES,
    count_cap_threshold,
    evaluate_threshold,
    score_metrics,
    score_pair_for_config as base_score_pair_for_config,
)
from run_experiment_29_train_normal_threshold_calibration import knn_score_pair, train_false_positive_stats
from run_rank_ensemble_calibration import align_series_lengths, sanitize_series, z_normalize


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
DEFAULT_WORKERS = int(os.environ.get("ORIGINAL_IMPROVEMENT_WORKERS", "4"))


ORIGINAL_EXPERIMENT_MAP = {
    "experiment_35_original_threshold_policy_sweep": "experiment_35_balanced_threshold_policy_sweep",
    "experiment_36_original_score_normalization_sweep": "experiment_36_balanced_score_normalization_sweep",
    "experiment_37_original_bagged_rocket_ensemble": "experiment_37_balanced_bagged_rocket_ensemble",
    "experiment_38_original_actual_length_handling": "experiment_38_balanced_actual_length_handling",
    "experiment_39_original_candidate_retest": "experiment_39_balanced_candidate_retest",
}


LABEL_OVERRIDES = {
    "experiment_35_original_threshold_policy_sweep": "Exp 35 - Original threshold policy diagnosis",
    "experiment_36_original_score_normalization_sweep": "Exp 36 - Original density-normalized KNN scores",
    "experiment_37_original_bagged_rocket_ensemble": "Exp 37 - Original bagged ROCKET ensemble",
    "experiment_38_original_actual_length_handling": "Exp 38 - Original actual length handling",
    "experiment_39_original_candidate_retest": "Exp 39 - Original candidate retest and family guard",
}


ORIGINAL_CUSTOM_SPECS = {
    "experiment_43_explanation_space_transforms": {
        "label": "Exp 43 - Original explanation-space transform diagnostic",
        "source_experiment_id": "paper_explanation_space_minzero",
        "data_variant": "original_repeated_normal",
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "adaptive_v0", "family_guard_v1"],
        "configs": [
            {
                "name": "time_z_rocket_256_knn3",
                "kind": "rocket_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "preprocess_space": "time_z",
            },
            {
                "name": "minzero_z_rocket_256_knn3",
                "kind": "rocket_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "preprocess_space": "minzero_z",
            },
            {
                "name": "difference_z_rocket_256_knn3",
                "kind": "rocket_knn",
                "num_kernels": 256,
                "neighbors": 3,
                "preprocess_space": "difference_z",
            },
        ],
    },
    "experiment_44_classical_embedding_baselines": {
        "label": "Exp 44 - Original classical embedding baselines",
        "source_experiment_id": "paper_time_series_embedding_review",
        "data_variant": "original_repeated_normal",
        "thresholds": ["count_cap_2pct", "count_cap_3pct", "adaptive_v0", "family_guard_v1"],
        "configs": [
            {
                "name": "fft_mag_16_knn3",
                "kind": "classical_embedding_knn",
                "embedding": "fft",
                "n_features": 16,
                "neighbors": 3,
            },
            {
                "name": "fft_mag_32_knn3",
                "kind": "classical_embedding_knn",
                "embedding": "fft",
                "n_features": 32,
                "neighbors": 3,
            },
            {
                "name": "pca_8_knn3",
                "kind": "classical_embedding_knn",
                "embedding": "pca",
                "n_components": 8,
                "neighbors": 3,
            },
            {
                "name": "pca_16_knn3",
                "kind": "classical_embedding_knn",
                "embedding": "pca",
                "n_components": 16,
                "neighbors": 3,
            },
            {
                "name": "haar_wavelet_16_knn3",
                "kind": "classical_embedding_knn",
                "embedding": "haar_wavelet",
                "n_features": 16,
                "neighbors": 3,
            },
            {
                "name": "haar_wavelet_32_knn3",
                "kind": "classical_embedding_knn",
                "embedding": "haar_wavelet",
                "n_features": 32,
                "neighbors": 3,
            },
        ],
    },
}


def experiment_paths(exp_id):
    return {
        "detail": DATA_DIR / f"{exp_id}_results.csv",
        "summary": DATA_DIR / f"{exp_id}_summary.csv",
        "log": DATA_DIR / f"{exp_id}.log",
    }


def get_spec(exp_id):
    if exp_id in ORIGINAL_CUSTOM_SPECS:
        spec = copy.deepcopy(ORIGINAL_CUSTOM_SPECS[exp_id])
        spec["id"] = exp_id
        spec["paths"] = experiment_paths(exp_id)
        return spec
    if exp_id not in ORIGINAL_EXPERIMENT_MAP:
        raise SystemExit(f"Unknown original improvement experiment: {exp_id}")
    spec = dict(BALANCED_SPECS[ORIGINAL_EXPERIMENT_MAP[exp_id]])
    spec["id"] = exp_id
    spec["source_experiment_id"] = ORIGINAL_EXPERIMENT_MAP[exp_id]
    spec["label"] = LABEL_OVERRIDES[exp_id]
    spec["data_variant"] = "original_repeated_normal"
    spec["paths"] = experiment_paths(exp_id)
    return spec


def available_original_experiment_ids():
    return sorted(set(ORIGINAL_EXPERIMENT_MAP) | set(ORIGINAL_CUSTOM_SPECS))


def parse_family(dataset_name):
    if dataset_name == "CornellWhaleChallenge":
        return "CornellWhaleChallenge"
    if "_normal_" in dataset_name:
        return dataset_name.rsplit("_normal_", 1)[0]
    return dataset_name


def load_dataset_names_from_db(db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT name
            FROM datasets
            WHERE name NOT IN ('CornellWhaleChallenge', 'Wafer_normal_1')
            ORDER BY name
            """
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def min_zero_transform(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("min_zero_transform expects a 2D array")
    row_min = X.min(axis=1, keepdims=True)
    return (X - row_min).astype(np.float32)


def first_difference_transform(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("first_difference_transform expects a 2D array")
    if X.shape[1] == 0:
        return X.astype(np.float32)
    return np.diff(X, axis=1, prepend=X[:, :1]).astype(np.float32)


def transform_space_arrays(X_train_raw, X_test_raw, preprocess_space):
    if preprocess_space == "time_z":
        return (
            z_normalize(X_train_raw).astype(np.float32),
            z_normalize(X_test_raw).astype(np.float32),
        )
    if preprocess_space == "minzero_z":
        return (
            min_zero_transform(z_normalize(X_train_raw)),
            min_zero_transform(z_normalize(X_test_raw)),
        )
    if preprocess_space == "difference_z":
        return (
            first_difference_transform(z_normalize(X_train_raw)),
            first_difference_transform(z_normalize(X_test_raw)),
        )
    raise ValueError(f"Unknown preprocess space: {preprocess_space}")


def fixed_width_features(features, n_features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("fixed_width_features expects a 2D array")
    n_features = max(1, int(n_features))
    if features.shape[1] == n_features:
        return features.astype(np.float32, copy=False)
    if features.shape[1] > n_features:
        return features[:, :n_features].astype(np.float32, copy=False)
    pad = np.zeros((features.shape[0], n_features - features.shape[1]), dtype=np.float32)
    return np.hstack([features, pad]).astype(np.float32)


def fft_magnitude_features(X, n_features):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("fft_magnitude_features expects a 2D array")
    magnitude = np.abs(np.fft.rfft(X, axis=1)).astype(np.float32)
    if magnitude.shape[1] > 1:
        magnitude = magnitude[:, 1:]
    magnitude = np.log1p(magnitude)
    return fixed_width_features(magnitude, n_features)


def pca_embedding_pair(X_train, X_test, n_components):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("pca_embedding_pair expects 2D arrays")
    n_components = max(1, int(n_components))
    if X_train.shape[0] == 0:
        return np.empty((0, n_components), dtype=np.float32), np.empty((len(X_test), n_components), dtype=np.float32)
    center = X_train.mean(axis=0, keepdims=True)
    train_centered = X_train - center
    test_centered = X_test - center
    effective_components = max(1, min(n_components, X_train.shape[0], X_train.shape[1]))
    try:
        _, _, vt = np.linalg.svd(train_centered, full_matrices=False)
        components = vt[:effective_components].T
    except np.linalg.LinAlgError:
        components = np.eye(X_train.shape[1], effective_components, dtype=np.float32)
    train_features = train_centered @ components
    test_features = test_centered @ components
    return (
        fixed_width_features(train_features, n_components),
        fixed_width_features(test_features, n_components),
    )


def haar_wavelet_features(X, n_features):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("haar_wavelet_features expects a 2D array")
    rows = []
    scale = np.float32(np.sqrt(2.0))
    for row in X:
        current = row.astype(np.float32, copy=True)
        coeffs = []
        while len(current) > 1 and sum(len(part) for part in coeffs) < n_features:
            if len(current) % 2:
                current = np.append(current, current[-1]).astype(np.float32)
            left = current[0::2]
            right = current[1::2]
            avg = (left + right) / scale
            detail = (left - right) / scale
            coeffs.append(detail.astype(np.float32))
            current = avg.astype(np.float32)
        coeffs.append(current.astype(np.float32))
        rows.append(np.concatenate(coeffs) if coeffs else np.zeros(1, dtype=np.float32))
    return fixed_width_features(np.vstack(rows), n_features)


def classical_embedding_pair(X_train, X_test, config):
    embedding = config.get("embedding")
    if embedding == "fft":
        n_features = int(config.get("n_features", 32))
        return fft_magnitude_features(X_train, n_features), fft_magnitude_features(X_test, n_features)
    if embedding == "pca":
        return pca_embedding_pair(X_train, X_test, int(config.get("n_components", config.get("n_features", 16))))
    if embedding == "haar_wavelet":
        n_features = int(config.get("n_features", 32))
        return haar_wavelet_features(X_train, n_features), haar_wavelet_features(X_test, n_features)
    raise ValueError(f"Unknown classical embedding: {embedding}")


def embedding_width(config):
    if config.get("embedding") == "pca":
        return int(config.get("n_components", config.get("n_features", 16)))
    return int(config.get("n_features", config.get("n_components", 32)))


def config_with_embedding_width(config, width):
    widened = dict(config)
    if config.get("embedding") == "pca":
        widened["n_components"] = int(width)
    else:
        widened["n_features"] = int(width)
    return widened


def cached_classical_embedding_pair(X_train, X_test, config, cache=None, cache_key=None, max_width=None):
    requested_width = embedding_width(config)
    if cache is None or cache_key is None:
        return classical_embedding_pair(X_train, X_test, config)
    max_width = max(requested_width, int(max_width or requested_width))
    full_key = tuple(cache_key) + (config.get("embedding"), max_width)
    if full_key not in cache:
        cache[full_key] = classical_embedding_pair(X_train, X_test, config_with_embedding_width(config, max_width))
    train_features, test_features = cache[full_key]
    return fixed_width_features(train_features, requested_width), fixed_width_features(test_features, requested_width)


def original_score_pair_for_config(X_train, X_test, target_len, config, feature_cache=None, feature_cache_key=None, max_width=None):
    if config["kind"] == "classical_embedding_knn":
        train_features, test_features = cached_classical_embedding_pair(
            X_train,
            X_test,
            config,
            cache=feature_cache,
            cache_key=feature_cache_key,
            max_width=max_width,
        )
        return knn_score_pair(train_features, test_features, int(config.get("neighbors", 3)))
    return base_score_pair_for_config(X_train, X_test, target_len, config)


def load_original_record(dataset_name, db_path=DB_PATH):
    conn = sqlite3.connect(str(db_path))
    meta = conn.execute("SELECT series_length FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
    metadata_len = int(meta[0]) if meta and meta[0] else 0
    train_rows = conn.execute(
        """
        SELECT i.values_blob
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TRAIN'
        ORDER BY i.instance_index
        """,
        (dataset_name,),
    ).fetchall()
    test_rows = conn.execute(
        """
        SELECT i.values_blob, i.label
        FROM instances i
        JOIN datasets d ON i.dataset_id = d.id
        WHERE d.name = ? AND i.split = 'TEST'
        ORDER BY i.instance_index
        """,
        (dataset_name,),
    ).fetchall()
    conn.close()
    train_series = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in train_rows]
    test_series = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in test_rows]
    y_test = np.array([int(row[1]) for row in test_rows], dtype=np.int64)
    lengths = [len(x) for x in train_series + test_series]
    actual_median = float(np.median(lengths)) if lengths else float(metadata_len or 0)
    actual_max = float(np.max(lengths)) if lengths else float(metadata_len or 0)
    test_lengths = [len(x) for x in test_series]
    test_median = float(np.median(test_lengths)) if test_lengths else actual_median
    return {
        "dataset_name": dataset_name,
        "family": parse_family(dataset_name),
        "metadata_len": metadata_len,
        "actual_len_median": actual_median,
        "actual_len_max": actual_max,
        "test_actual_len_median": test_median,
        "train_series": train_series,
        "test_series": test_series,
        "y_test": y_test,
    }


def target_len_for_record(record, strategy):
    if strategy == "metadata":
        value = record["metadata_len"]
    elif strategy == "actual_median":
        value = record["actual_len_median"]
    elif strategy == "clean_test_median":
        value = record["test_actual_len_median"]
    elif strategy == "actual_max_cap2048":
        value = min(2048, record["actual_len_max"])
    else:
        raise ValueError(f"Unknown target length strategy: {strategy}")
    value = int(round(float(value)))
    if value <= 0:
        value = int(round(float(record["actual_len_median"] or 16)))
    return max(8, value)


def rate_for_threshold(method, record):
    if method.startswith("count_cap_") and method.endswith("pct"):
        pct = float(method.removeprefix("count_cap_").removesuffix("pct"))
        return pct / 100.0, "count_cap_rate"
    if method == "dynamic_1_over_n":
        return max(0.001, 1.0 / max(1, len(record["y_test"]))), "dynamic_count_cap"
    if method == "adaptive_v0":
        median_len = float(record["test_actual_len_median"])
        guarded_mid_length = 513 <= median_len <= 1024
        rate = 0.03 if len(record["y_test"]) > 50 and not guarded_mid_length else 0.02
        return rate, "adaptive_count_cap"
    if method == "family_guard_v1":
        long_series = float(record["test_actual_len_median"]) >= 512
        rate = 0.03 if record["family"] in WEAK_FAMILIES or long_series else 0.02
        return rate, "family_guard_count_cap"
    raise ValueError(f"Unknown threshold method: {method}")


def append_rows(path, rows, fieldnames):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_existing_detail_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return [], None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def completed_dataset_names(rows, exp_id=None):
    completed = set()
    for row in rows:
        if exp_id is not None and row.get("experiment_id", exp_id) != exp_id:
            continue
        dataset_name = row.get("dataset_name")
        if dataset_name:
            completed.add(dataset_name)
    return completed


def assert_detail_dataset_coverage(path, expected_dataset_names, exp_id, logger):
    disk_rows, _ = read_existing_detail_rows(path)
    disk_dataset_names = completed_dataset_names(disk_rows, exp_id)
    missing = [name for name in expected_dataset_names if name not in disk_dataset_names]
    missing_path = path.with_name(f"{path.stem}_missing_datasets.csv")
    if missing:
        with missing_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "dataset_name"])
            writer.writeheader()
            writer.writerows({"experiment_id": exp_id, "dataset_name": name} for name in missing)
        logger.warning(
            "%s detail CSV is missing %d/%d datasets after run; wrote %s. Queue will continue.",
            exp_id,
            len(missing),
            len(expected_dataset_names),
            missing_path,
        )
    elif missing_path.exists():
        missing_path.unlink()
    return disk_rows


def repair_missing_datasets(exp_id, db_path, detail_path, expected_dataset_names, fieldnames, logger, max_attempts=1):
    detail_rows, existing_fieldnames = read_existing_detail_rows(detail_path)
    fieldnames = fieldnames or existing_fieldnames
    for attempt in range(1, int(max_attempts) + 1):
        completed = completed_dataset_names(detail_rows, exp_id)
        missing = [name for name in expected_dataset_names if name not in completed]
        if not missing:
            break
        logger.warning(
            "%s repairing %d missing datasets before queue continues. attempt=%d",
            exp_id,
            len(missing),
            attempt,
        )
        for dataset_name in missing:
            try:
                rows = run_dataset_task((exp_id, str(db_path), dataset_name))
            except Exception as exc:
                logger.error("Repair failed for %s: %s", dataset_name, exc, exc_info=True)
                rows = []
            if not rows:
                logger.warning("Repair produced no rows for %s", dataset_name)
                continue
            detail_rows.extend(rows)
            fieldnames = fieldnames or list(rows[0].keys())
            append_rows(detail_path, rows, fieldnames)
    return read_existing_detail_rows(detail_path)[0]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_dataset_task(task):
    exp_id, db_path, dataset_name = task
    spec = get_spec(exp_id)
    record = load_original_record(dataset_name, db_path)
    y_test = record["y_test"]
    if len(record["train_series"]) == 0 or len(record["test_series"]) == 0 or len(np.unique(y_test)) < 2:
        return []

    arrays_by_len = {}
    raw_arrays_by_len = {}
    feature_cache = {}
    max_embedding_widths = {}
    for config in spec["configs"]:
        if config.get("kind") != "classical_embedding_knn":
            continue
        target_strategy = config.get("target_len_strategy", "metadata")
        target_len = target_len_for_record(record, target_strategy)
        preprocess_space = config.get("preprocess_space", "time_z")
        key = (target_len, preprocess_space, config.get("embedding"))
        max_embedding_widths[key] = max(max_embedding_widths.get(key, 0), embedding_width(config))
    rows = []
    for config in spec["configs"]:
        target_strategy = config.get("target_len_strategy", "metadata")
        target_len = target_len_for_record(record, target_strategy)
        preprocess_space = config.get("preprocess_space", "time_z")
        if target_len not in raw_arrays_by_len:
            raw_arrays_by_len[target_len] = (
                align_series_lengths(record["train_series"], target_len),
                align_series_lengths(record["test_series"], target_len),
            )
        array_key = (target_len, preprocess_space)
        if array_key not in arrays_by_len:
            X_train_raw, X_test_raw = raw_arrays_by_len[target_len]
            arrays_by_len[array_key] = transform_space_arrays(X_train_raw, X_test_raw, preprocess_space)
        X_train, X_test = arrays_by_len[array_key]
        max_width = max_embedding_widths.get((target_len, preprocess_space, config.get("embedding")))
        train_scores, test_scores = original_score_pair_for_config(
            X_train,
            X_test,
            target_len,
            config,
            feature_cache=feature_cache,
            feature_cache_key=array_key,
            max_width=max_width,
        )
        metrics = score_metrics(y_test, test_scores)
        for method in spec["thresholds"]:
            rate, threshold_family = rate_for_threshold(method, record)
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "source_experiment_id": spec["source_experiment_id"],
                    "dataset_name": dataset_name,
                    "family": record["family"],
                    "data_variant": spec.get("data_variant", "original_repeated_normal"),
                    "config_name": config["name"],
                    "preprocess_space": preprocess_space,
                    "score_family": config["kind"],
                    "embedding": config.get("embedding", ""),
                    "feature_count": config.get("n_features", config.get("n_components", "")),
                    "num_kernels": config.get("num_kernels", ""),
                    "knn_neighbors": config.get("neighbors", ""),
                    "target_len_strategy": target_strategy,
                    "threshold_method": method,
                    "threshold_family": threshold_family,
                    "sequence_length": target_len,
                    "metadata_len": record["metadata_len"],
                    "actual_len_median": record["actual_len_median"],
                    "actual_len_max": record["actual_len_max"],
                    "test_actual_len_median": record["test_actual_len_median"],
                    "train_score_count": len(train_scores),
                    "train_count": len(record["train_series"]),
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "q_effective": q_effective,
                    "cap_target": cap_target,
                    "threshold": threshold,
                    "train_exceed_count": train_exceed_count,
                    "train_exceed_rate": train_exceed_rate,
                    **evaluate_threshold(y_test, test_scores, threshold, metrics),
                }
            )
    return rows


def summarize(rows):
    summary = []
    keys = sorted({(row["experiment_id"], row["config_name"], row["threshold_method"]) for row in rows})
    for exp_id, config_name, method in keys:
        subset = [
            row
            for row in rows
            if row["experiment_id"] == exp_id
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
                "data_variant": subset[0].get("data_variant", "original_repeated_normal"),
                "config_name": config_name,
                "preprocess_space": subset[0].get("preprocess_space", "time_z"),
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
                "mean_anomaly_count": float(np.mean([int(row["anomaly_count"]) for row in subset])),
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

    detail_rows = []
    fieldnames = None
    if args.keep_existing:
        detail_rows, fieldnames = read_existing_detail_rows(paths["detail"])

    dataset_names = load_dataset_names_from_db(args.db_path)
    if args.dataset_limit is not None:
        dataset_names = dataset_names[: args.dataset_limit]
    expected_dataset_names = list(dataset_names)
    if args.keep_existing and detail_rows:
        completed_names = completed_dataset_names(detail_rows, exp_id)
        dataset_names = [name for name in dataset_names if name not in completed_names]
    tasks = [(exp_id, str(args.db_path), name) for name in dataset_names]
    logger.info(
        "Starting %s on %d original datasets with %d workers. existing_rows=%d",
        exp_id,
        len(tasks),
        args.workers,
        len(detail_rows),
    )
    if not tasks and detail_rows:
        write_csv(paths["summary"], summarize(detail_rows))
        logger.info("%s resume found no remaining datasets.", exp_id)
        return
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_dataset_task, task): task for task in tasks}
        for future in as_completed(futures):
            _, _, dataset_name = futures[future]
            try:
                rows = future.result()
            except Exception as exc:
                logger.error("Error evaluating %s: %s", dataset_name, exc, exc_info=True)
                rows = []
            completed += 1
            if not rows:
                logger.warning("No rows produced for %s", dataset_name)
            if rows:
                detail_rows.extend(rows)
                fieldnames = fieldnames or list(rows[0].keys())
                append_rows(paths["detail"], rows, fieldnames)
            if completed % 25 == 0 or completed == len(tasks):
                summary_rows = summarize(detail_rows) if detail_rows else []
                write_csv(paths["summary"], summary_rows)
                best = summary_rows[0] if summary_rows else None
                if best:
                    logger.info(
                        "Progress: [%4d/%4d] rows=%d | best=%s/%s meanF1=%.4f medianF1=%.4f aucPR=%.4f zero=%d fp=%.2f",
                        completed,
                        len(tasks),
                        len(detail_rows),
                        best["config_name"],
                        best["threshold_method"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["mean_auc_pr"],
                        best["zero_f1_count"],
                        best["mean_fp"],
                    )
    final_rows = repair_missing_datasets(exp_id, args.db_path, paths["detail"], expected_dataset_names, fieldnames, logger)
    final_rows = assert_detail_dataset_coverage(paths["detail"], expected_dataset_names, exp_id, logger)
    write_csv(paths["summary"], summarize(final_rows))
    logger.info("%s finished.", exp_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run original-data improvement experiments.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args(argv)


def main_for_experiment(exp_id, argv=None):
    run_experiment(exp_id, parse_args(argv))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one original-data improvement experiment by id.")
    parser.add_argument("experiment_id", choices=available_original_experiment_ids())
    args, rest = parser.parse_known_args(argv)
    main_for_experiment(args.experiment_id, rest)


if __name__ == "__main__":
    main()
