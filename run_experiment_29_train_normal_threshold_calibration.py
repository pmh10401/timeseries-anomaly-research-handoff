import csv
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import scipy.stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from run_experiment_26_rocket import make_kernels, rocket_transform, load_dataset_names
from run_experiment_27f_rstsf_interval import interval_features, make_intervals, representation_sets
from run_experiment_28_minirocket_multirocket_features import (
    build_feature_sets as build_mini_multi_feature_sets,
)
from run_rank_ensemble_calibration import load_dataset_data, z_normalize
from run_rank_threshold_calibration import top_k_oracle_f1


DATA_DIR = "/Users/minho/Documents/Dataset"
DETAIL_OUT_PATH = f"{DATA_DIR}/experiment_29_train_normal_threshold_calibration_results.csv"
SUMMARY_OUT_PATH = f"{DATA_DIR}/experiment_29_train_normal_threshold_calibration_summary.csv"
LOG_PATH = f"{DATA_DIR}/experiment_29_train_normal_threshold_calibration.log"


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


RNG_SEED = 20260706
WORKERS = env_int("TRAIN_THRESHOLD_WORKERS", 4)
MAX_TRAIN_REFERENCE = env_int("TRAIN_THRESHOLD_MAX_TRAIN_REFERENCE", 1000)

CONFIGS = [
    ("rocket_256_robust_top16", "rocket", {"num_kernels": 256, "score_mode": "robust_topk", "top_k": 16}),
    ("rocket_256_robust_top32", "rocket", {"num_kernels": 256, "score_mode": "robust_topk", "top_k": 32}),
    ("rocket_256_robust_top64", "rocket", {"num_kernels": 256, "score_mode": "robust_topk", "top_k": 64}),
    ("rocket_256_knn5", "rocket", {"num_kernels": 256, "score_mode": "knn", "neighbors": 5}),
    ("rocket_256_iforest", "rocket", {"num_kernels": 256, "score_mode": "iforest"}),
    ("rocket_1024_robust_top32", "rocket", {"num_kernels": 1024, "score_mode": "robust_topk", "top_k": 32}),
    ("rocket_1024_robust_top64", "rocket", {"num_kernels": 1024, "score_mode": "robust_topk", "top_k": 64}),
    ("rstsf_interval_top32", "rstsf", {"num_intervals": 256, "top_k": 32}),
    ("rstsf_interval_top64", "rstsf", {"num_intervals": 256, "top_k": 64}),
    ("minirocket_ppv_raw_top32", "mini_multi", {"feature_key": "mini", "top_k": 32}),
    ("minirocket_ppv_raw_top64", "mini_multi", {"feature_key": "mini", "top_k": 64}),
    ("multirocket_stats_raw_diff_top32", "mini_multi", {"feature_key": "multi", "top_k": 32}),
    ("multirocket_stats_raw_diff_top64", "mini_multi", {"feature_key": "multi", "top_k": 64}),
]

THRESHOLD_METHODS = [
    "empirical_dynamic_quantile",
    "skew_adaptive",
    "evt_gpd_p90_dynamic",
    "evt_gpd_p95_dynamic",
    "fixed_q02_empirical",
]


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment29TrainNormalThresholdCalibration")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


def reference_train(X_train, seq_len, seed_offset=0):
    if len(X_train) <= MAX_TRAIN_REFERENCE:
        return X_train
    rng = np.random.default_rng(RNG_SEED + seed_offset + seq_len + len(X_train))
    idx = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
    return X_train[idx]


def robust_deviation_pair(train_features, test_features, top_k):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    center = np.median(train_scaled, axis=0)
    spread = np.median(np.abs(train_scaled - center), axis=0) + 1e-6
    train_deviations = np.abs(train_scaled - center) / spread
    test_deviations = np.abs(test_scaled - center) / spread
    top_k = max(1, min(top_k, train_deviations.shape[1]))
    split_at = train_deviations.shape[1] - top_k
    train_scores = np.mean(np.partition(train_deviations, split_at, axis=1)[:, -top_k:], axis=1)
    test_scores = np.mean(np.partition(test_deviations, split_at, axis=1)[:, -top_k:], axis=1)
    return train_scores, test_scores


def knn_score_pair(train_features, test_features, neighbors):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    n_neighbors = max(1, min(neighbors + 1, len(train_scaled)))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(train_scaled)
    train_distances, _ = nn.kneighbors(train_scaled)
    test_distances, _ = nn.kneighbors(test_scaled)
    if train_distances.shape[1] > 1:
        train_scores = train_distances[:, 1:].mean(axis=1)
    else:
        train_scores = train_distances[:, 0]
    test_scores = test_distances[:, : max(1, min(neighbors, test_distances.shape[1]))].mean(axis=1)
    return train_scores, test_scores


def iforest_score_pair(train_features, test_features):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=RNG_SEED, n_jobs=1)
    model.fit(train_scaled)
    return -model.score_samples(train_scaled), -model.score_samples(test_scaled)


def rocket_feature_pair(X_train, X_test, seq_len, num_kernels):
    X_ref = reference_train(X_train, seq_len, seed_offset=num_kernels)
    kernels = make_kernels(seq_len, num_kernels=num_kernels)
    train_features = rocket_transform(X_ref, kernels)
    test_features = rocket_transform(X_test, kernels)
    return train_features, test_features


def score_pair_from_features(train_features, test_features, params):
    mode = params["score_mode"]
    if mode == "robust_topk":
        return robust_deviation_pair(train_features, test_features, params["top_k"])
    if mode == "knn":
        return knn_score_pair(train_features, test_features, params["neighbors"])
    if mode == "iforest":
        return iforest_score_pair(train_features, test_features)
    raise ValueError(f"Unknown rocket score mode: {mode}")


def rocket_score_pair(X_train, X_test, seq_len, params):
    train_features, test_features = rocket_feature_pair(X_train, X_test, seq_len, params["num_kernels"])
    return score_pair_from_features(train_features, test_features, params)


def mini_multi_score_pair(X_train, X_test, seq_len, params):
    feature_sets = build_mini_multi_feature_sets(X_train, X_test, seq_len)
    train_features, test_features = feature_sets[params["feature_key"]]
    return robust_deviation_pair(train_features, test_features, params["top_k"])


def rstsf_feature_pair(X_train, X_test, seq_len, num_intervals):
    X_ref = reference_train(X_train, seq_len, seed_offset=4000)
    train_reps = representation_sets(X_ref)
    test_reps = representation_sets(X_test)
    per_rep_intervals = max(8, num_intervals // len(train_reps))
    train_blocks = []
    test_blocks = []
    for rep_index, (train_rep, test_rep) in enumerate(zip(train_reps, test_reps)):
        intervals = make_intervals(train_rep.shape[1], len(X_train), rep_index, per_rep_intervals)
        train_blocks.append(interval_features(train_rep, intervals))
        test_blocks.append(interval_features(test_rep, intervals))
    train_features = np.concatenate(train_blocks, axis=1)
    test_features = np.concatenate(test_blocks, axis=1)
    return train_features, test_features


def rstsf_score_pair(X_train, X_test, seq_len, params):
    train_features, test_features = rstsf_feature_pair(X_train, X_test, seq_len, params["num_intervals"])
    return robust_deviation_pair(train_features, test_features, params["top_k"])


def clean_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([0.0], dtype=np.float64)
    return scores


def dynamic_q(n_test):
    return max(0.001, 1.0 / max(1, n_test))


def empirical_threshold(train_scores, q_target):
    return float(np.percentile(train_scores, 100.0 * (1.0 - q_target)))


def skew_adaptive_threshold(train_scores, q_target):
    fallback = empirical_threshold(train_scores, q_target)
    shifted = train_scores - np.min(train_scores) + 1e-6
    skew = scipy.stats.skew(shifted)
    try:
        if skew > 1.2:
            shape, loc, scale = scipy.stats.lognorm.fit(shifted, floc=0)
            threshold = scipy.stats.lognorm.ppf(1.0 - q_target, shape, loc=0, scale=scale) + np.min(train_scores) - 1e-6
        elif skew < 0.2:
            mu, std = scipy.stats.norm.fit(train_scores)
            threshold = scipy.stats.norm.ppf(1.0 - q_target, mu, std)
        else:
            a_fit, loc_fit, scale_fit = scipy.stats.gamma.fit(shifted, floc=0)
            threshold = scipy.stats.gamma.ppf(1.0 - q_target, a_fit, loc=0, scale=scale_fit) + np.min(train_scores) - 1e-6
    except Exception:
        threshold = fallback
    if not np.isfinite(threshold):
        threshold = fallback
    return float(threshold)


def evt_gpd_threshold(train_scores, q_target, tail_percentile):
    adaptive = skew_adaptive_threshold(train_scores, q_target)
    t = np.percentile(train_scores, tail_percentile)
    excesses = train_scores[train_scores > t] - t
    n = len(train_scores)
    nt = len(excesses)
    if nt <= 10:
        return adaptive
    try:
        c_fit, loc_fit, scale_fit = scipy.stats.genpareto.fit(excesses, floc=0)
        prob_excess = 1.0 - (q_target * n / nt)
        prob_excess = np.clip(prob_excess, 0.90, 0.9999)
        threshold = t + scipy.stats.genpareto.ppf(prob_excess, c_fit, loc=0, scale=scale_fit)
    except Exception:
        threshold = adaptive
    if not np.isfinite(threshold):
        threshold = adaptive
    return float(threshold)


def threshold_for_method(train_scores, n_test, method):
    train_scores = clean_scores(train_scores)
    q_target = dynamic_q(n_test)
    if method == "empirical_dynamic_quantile":
        return empirical_threshold(train_scores, q_target), q_target
    if method == "skew_adaptive":
        return skew_adaptive_threshold(train_scores, q_target), q_target
    if method == "evt_gpd_p90_dynamic":
        return evt_gpd_threshold(train_scores, q_target, 90), q_target
    if method == "evt_gpd_p95_dynamic":
        return evt_gpd_threshold(train_scores, q_target, 95), q_target
    if method == "fixed_q02_empirical":
        return empirical_threshold(train_scores, 0.02), 0.02
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
    return {
        "predicted_count": int(preds.sum()),
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "f1": f1_score(y_true, preds, zero_division=0),
        "oracle_f1": metrics["oracle_f1"],
    }


def train_false_positive_stats(train_scores, threshold):
    train_scores = np.asarray(train_scores, dtype=np.float64)
    exceed = train_scores > threshold
    return int(exceed.sum()), float(exceed.mean()) if len(exceed) else 0.0


def run_dataset(dataset_name):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    if len(X_train) == 0 or len(X_test) == 0:
        return []
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    rows = []
    rocket_feature_cache = {}
    mini_multi_feature_sets = None
    rstsf_feature_cache = {}
    for config_name, family, params in CONFIGS:
        if family == "rocket":
            num_kernels = params["num_kernels"]
            if num_kernels not in rocket_feature_cache:
                rocket_feature_cache[num_kernels] = rocket_feature_pair(X_train, X_test, seq_len, num_kernels)
            train_features, test_features = rocket_feature_cache[num_kernels]
            train_scores, test_scores = score_pair_from_features(train_features, test_features, params)
        elif family == "mini_multi":
            if mini_multi_feature_sets is None:
                mini_multi_feature_sets = build_mini_multi_feature_sets(X_train, X_test, seq_len)
            train_features, test_features = mini_multi_feature_sets[params["feature_key"]]
            train_scores, test_scores = robust_deviation_pair(train_features, test_features, params["top_k"])
        elif family == "rstsf":
            num_intervals = params["num_intervals"]
            if num_intervals not in rstsf_feature_cache:
                rstsf_feature_cache[num_intervals] = rstsf_feature_pair(X_train, X_test, seq_len, num_intervals)
            train_features, test_features = rstsf_feature_cache[num_intervals]
            train_scores, test_scores = robust_deviation_pair(train_features, test_features, params["top_k"])
        else:
            raise ValueError(f"Unknown family: {family}")
        metrics = score_metrics(y_test, test_scores)
        for method in THRESHOLD_METHODS:
            threshold, q_target = threshold_for_method(train_scores, len(y_test), method)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "config_name": config_name,
                    "threshold_method": method,
                    "sequence_length": seq_len,
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "train_score_count": len(train_scores),
                    "q_target": q_target,
                    "threshold": threshold,
                    "train_exceed_count": train_exceed_count,
                    "train_exceed_rate": train_exceed_rate,
                    **evaluate_threshold(y_test, test_scores, threshold, metrics),
                }
            )
    return rows


def append_rows(path, rows, fieldnames):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    keys = sorted({(row["config_name"], row["threshold_method"]) for row in rows})
    for config_name, method in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["threshold_method"] == method]
        f1s = sorted(float(row["f1"]) for row in subset)
        summary.append(
            {
                "config_name": config_name,
                "threshold_method": method,
                "num_datasets": len(subset),
                "mean_auc_roc": np.mean([float(row["auc_roc"]) for row in subset]),
                "mean_auc_pr": np.mean([float(row["auc_pr"]) for row in subset]),
                "mean_f1": np.mean(f1s),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(1 for value in f1s if value == 0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "mean_predicted_count": np.mean([int(row["predicted_count"]) for row in subset]),
                "mean_train_exceed_rate": np.mean([float(row["train_exceed_rate"]) for row in subset]),
                "mean_oracle_f1": np.mean([float(row["oracle_f1"]) for row in subset]),
            }
        )
    return sorted(summary, key=lambda row: row["mean_f1"], reverse=True)


def run_one(dataset_name):
    return dataset_name, run_dataset(dataset_name)


def main():
    for path in [DETAIL_OUT_PATH, SUMMARY_OUT_PATH]:
        if os.path.exists(path):
            os.remove(path)
    dataset_names = load_dataset_names()
    logger.info(
        "Starting Experiment 29 train-normal threshold calibration on %d datasets with %d workers.",
        len(dataset_names),
        WORKERS,
    )
    detail_rows = []
    fieldnames = None
    completed = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, name): name for name in dataset_names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                dataset_name, rows = future.result()
            except Exception as exc:
                logger.error("Error evaluating dataset %s: %s", name, exc, exc_info=True)
                dataset_name, rows = name, []
            completed += 1
            if rows:
                detail_rows.extend(rows)
                fieldnames = fieldnames or list(rows[0].keys())
                append_rows(DETAIL_OUT_PATH, rows, fieldnames)
            if completed % 25 == 0 or completed == len(dataset_names):
                summary_rows = summarize(detail_rows)
                write_csv(SUMMARY_OUT_PATH, summary_rows)
                best = summary_rows[0] if summary_rows else None
                if best:
                    logger.info(
                        "Progress: [%4d/%4d] rows=%d | best=%s/%s meanF1=%.4f medianF1=%.4f zero=%d",
                        completed,
                        len(dataset_names),
                        len(detail_rows),
                        best["config_name"],
                        best["threshold_method"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["zero_f1_count"],
                    )
    logger.info("Experiment 29 train-normal threshold calibration finished.")


if __name__ == "__main__":
    main()
