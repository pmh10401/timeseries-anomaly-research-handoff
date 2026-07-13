import csv
import itertools
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
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import RobustScaler

from run_experiment_26_rocket import dilated_convolution, load_dataset_names
from run_rank_ensemble_calibration import load_dataset_data, z_normalize
from run_rank_threshold_calibration import STRATEGIES, predict_by_strategy, top_k_oracle_f1


DATA_DIR = "/Users/minho/Documents/Dataset"
DETAIL_OUT_PATH = f"{DATA_DIR}/experiment_28_minirocket_multirocket_features_results.csv"
SUMMARY_OUT_PATH = f"{DATA_DIR}/experiment_28_minirocket_multirocket_features_summary.csv"
LOG_PATH = f"{DATA_DIR}/experiment_28_minirocket_multirocket_features.log"


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


RNG_SEED = 20260706
WORKERS = env_int("MINIROCKET_WORKERS", 4)
MINI_PARAMS = env_int("MINIROCKET_FEATURE_PARAMS", 256)
MULTI_PARAMS_PER_REP = env_int("MULTIROCKET_PARAMS_PER_REP", 128)
MAX_TRAIN_REFERENCE = env_int("MINIROCKET_MAX_TRAIN_REFERENCE", 1000)
BIAS_QUANTILE_LOW = 0.10
BIAS_QUANTILE_HIGH = 0.90

CONFIGS = [
    ("minirocket_ppv_raw_top32", "mini", 32),
    ("minirocket_ppv_raw_top64", "mini", 64),
    ("multirocket_stats_raw_diff_top32", "multi", 32),
    ("multirocket_stats_raw_diff_top64", "multi", 64),
]


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment28MiniMultiRocketFeatures")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


def reference_train(X_train, seq_len):
    if len(X_train) <= MAX_TRAIN_REFERENCE:
        return X_train
    rng = np.random.default_rng(RNG_SEED + seq_len + len(X_train))
    idx = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
    return X_train[idx]


def base_kernel_bank():
    kernels = []
    for positive_indices in itertools.combinations(range(9), 3):
        weights = np.full(9, -1.0, dtype=np.float32)
        weights[list(positive_indices)] = 2.0
        kernels.append(weights)
    return kernels


def make_feature_params(series_length, num_params, seed_offset):
    rng = np.random.default_rng(RNG_SEED + seed_offset + series_length)
    kernels = base_kernel_bank()
    max_dilation_power = max(0, int(math.log2(max(1, (series_length - 1) / 8))))
    params = []
    for idx in range(num_params):
        weights = kernels[idx % len(kernels)]
        dilation = int(2 ** rng.integers(0, max_dilation_power + 1)) if max_dilation_power else 1
        padding = bool(rng.integers(0, 2))
        quantile = float(rng.uniform(BIAS_QUANTILE_LOW, BIAS_QUANTILE_HIGH))
        params.append((weights, dilation, padding, quantile))
    return params


def conv_values(X, weights, dilation, padding):
    return [dilated_convolution(x, weights, dilation, padding, 0.0) for x in X]


def fit_bias(train_convs, quantile):
    values = np.concatenate(train_convs) if train_convs else np.array([0.0], dtype=np.float32)
    return float(np.quantile(values, quantile))


def ppv_feature(conv, bias):
    return float(np.mean(conv > bias))


def multipool_features(conv, bias):
    above = conv > bias
    ppv = float(np.mean(above))
    if not np.any(above):
        return ppv, 0.0, 0.0, 0.0
    excess = conv[above] - bias
    mpv = float(np.mean(excess))
    mipv = float(np.mean(np.flatnonzero(above) / max(1, len(conv) - 1)))
    longest = 0
    current = 0
    for flag in above:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    lspv = float(longest / len(conv))
    return ppv, mpv, mipv, lspv


def minirocket_ppv_transform(X_train, X_test, params):
    train_features = np.empty((len(X_train), len(params)), dtype=np.float32)
    test_features = np.empty((len(X_test), len(params)), dtype=np.float32)
    for col, (weights, dilation, padding, quantile) in enumerate(params):
        train_convs = conv_values(X_train, weights, dilation, padding)
        bias = fit_bias(train_convs, quantile)
        for row_idx, conv in enumerate(train_convs):
            train_features[row_idx, col] = ppv_feature(conv, bias)
        for row_idx, conv in enumerate(conv_values(X_test, weights, dilation, padding)):
            test_features[row_idx, col] = ppv_feature(conv, bias)
    return train_features, test_features


def multirocket_stats_transform(X_train, X_test, params):
    train_features = np.empty((len(X_train), len(params) * 4), dtype=np.float32)
    test_features = np.empty((len(X_test), len(params) * 4), dtype=np.float32)
    for idx, (weights, dilation, padding, quantile) in enumerate(params):
        train_convs = conv_values(X_train, weights, dilation, padding)
        bias = fit_bias(train_convs, quantile)
        base = idx * 4
        for row_idx, conv in enumerate(train_convs):
            train_features[row_idx, base : base + 4] = multipool_features(conv, bias)
        for row_idx, conv in enumerate(conv_values(X_test, weights, dilation, padding)):
            test_features[row_idx, base : base + 4] = multipool_features(conv, bias)
    return train_features, test_features


def diff_representation(X):
    if X.shape[1] <= 1:
        return X
    return np.diff(X, n=1, axis=1).astype(np.float32)


def build_feature_sets(X_train, X_test, seq_len):
    X_ref = reference_train(X_train, seq_len)

    mini_params = make_feature_params(seq_len, MINI_PARAMS, seed_offset=1000)
    mini_train, mini_test = minirocket_ppv_transform(X_ref, X_test, mini_params)

    raw_params = make_feature_params(seq_len, MULTI_PARAMS_PER_REP, seed_offset=2000)
    raw_train, raw_test = multirocket_stats_transform(X_ref, X_test, raw_params)

    X_ref_diff = diff_representation(X_ref)
    X_test_diff = diff_representation(X_test)
    diff_params = make_feature_params(X_ref_diff.shape[1], MULTI_PARAMS_PER_REP, seed_offset=3000)
    diff_train, diff_test = multirocket_stats_transform(X_ref_diff, X_test_diff, diff_params)
    multi_train = np.concatenate([raw_train, diff_train], axis=1)
    multi_test = np.concatenate([raw_test, diff_test], axis=1)

    return {
        "mini": (mini_train, mini_test),
        "multi": (multi_train, multi_test),
    }


def robust_deviation_scores(train_features, test_features, top_k):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    center = np.median(train_scaled, axis=0)
    spread = np.median(np.abs(train_scaled - center), axis=0) + 1e-6
    deviations = np.abs(test_scaled - center) / spread
    top_k = max(1, min(top_k, deviations.shape[1]))
    split_at = deviations.shape[1] - top_k
    return np.mean(np.partition(deviations, split_at, axis=1)[:, -top_k:], axis=1)


def score_metrics(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {
        "auc_roc": roc_auc_score(y_true, scores),
        "auc_pr": auc(recall, precision),
        "oracle_f1": top_k_oracle_f1(y_true, scores),
    }


def evaluate_scores(y_true, scores, strategy, metrics):
    preds = predict_by_strategy(scores, strategy)
    return {
        "predicted_count": int(preds.sum()),
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "f1": f1_score(y_true, preds, zero_division=0),
        "oracle_f1": metrics["oracle_f1"],
    }


def run_dataset(dataset_name):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    if len(X_train) == 0 or len(X_test) == 0:
        return []
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    feature_sets = build_feature_sets(X_train, X_test, seq_len)
    rows = []
    for config_name, feature_key, top_k in CONFIGS:
        train_features, test_features = feature_sets[feature_key]
        scores = robust_deviation_scores(train_features, test_features, top_k)
        metrics = score_metrics(y_test, scores)
        for strategy in STRATEGIES:
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "config_name": config_name,
                    "strategy": strategy,
                    "sequence_length": seq_len,
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "num_features": train_features.shape[1],
                    **evaluate_scores(y_test, scores, strategy, metrics),
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
    keys = sorted({(row["config_name"], row["strategy"]) for row in rows})
    for config_name, strategy in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["strategy"] == strategy]
        f1s = sorted(float(row["f1"]) for row in subset)
        summary.append(
            {
                "config_name": config_name,
                "strategy": strategy,
                "num_datasets": len(subset),
                "num_features": int(float(subset[0]["num_features"])) if subset else 0,
                "mean_auc_roc": np.mean([float(row["auc_roc"]) for row in subset]),
                "mean_auc_pr": np.mean([float(row["auc_pr"]) for row in subset]),
                "mean_f1": np.mean(f1s),
                "median_f1": f1s[len(f1s) // 2],
                "zero_f1_count": sum(1 for value in f1s if value == 0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "mean_predicted_count": np.mean([int(row["predicted_count"]) for row in subset]),
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
        "Starting Experiment 28 MiniROCKET/MultiROCKET-style features on %d datasets with %d workers.",
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
                        best["strategy"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["zero_f1_count"],
                    )
    logger.info("Experiment 28 MiniROCKET/MultiROCKET-style features finished.")


if __name__ == "__main__":
    main()
