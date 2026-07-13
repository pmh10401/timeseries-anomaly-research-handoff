import csv
import logging
import math
import sqlite3

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import RobustScaler

from run_rank_ensemble_calibration import DB_PATH, load_dataset_data, z_normalize
from run_rank_threshold_calibration import STRATEGIES, predict_by_strategy, top_k_predictions


DATA_DIR = "/Users/minho/Documents/Dataset"
DETAIL_OUT_PATH = f"{DATA_DIR}/experiment_26_rocket_results.csv"
SUMMARY_OUT_PATH = f"{DATA_DIR}/experiment_26_rocket_summary.csv"
LOG_PATH = f"{DATA_DIR}/experiment_26_rocket.log"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment26Rocket")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


RNG_SEED = 20260706
NUM_KERNELS = 256
MAX_TRAIN_REFERENCE = 1000


def make_kernels(series_length, num_kernels=NUM_KERNELS, seed=RNG_SEED):
    rng = np.random.default_rng(seed + series_length)
    lengths = np.array([7, 9, 11], dtype=np.int64)
    kernels = []
    max_dilation_power = max(0, int(math.log2(max(1, (series_length - 1) / 8))))
    for _ in range(num_kernels):
        length = int(rng.choice(lengths))
        weights = rng.normal(0, 1, size=length).astype(np.float32)
        weights -= weights.mean()
        dilation = int(2 ** rng.integers(0, max_dilation_power + 1)) if max_dilation_power else 1
        padding = bool(rng.integers(0, 2))
        bias = float(rng.uniform(-1, 1))
        kernels.append((weights, dilation, padding, bias))
    return kernels


def dilated_convolution(x, weights, dilation, padding, bias):
    if dilation > 1:
        dilated = np.zeros((len(weights) - 1) * dilation + 1, dtype=np.float32)
        dilated[::dilation] = weights
        weights = dilated
    if padding:
        pad = len(weights) // 2
        x = np.pad(x, (pad, pad), mode="constant")
    if len(x) < len(weights):
        x = np.pad(x, (0, len(weights) - len(x)), mode="constant")
    conv = np.convolve(x, weights[::-1], mode="valid") + bias
    return conv


def rocket_transform(X, kernels):
    features = np.empty((len(X), len(kernels) * 2), dtype=np.float32)
    for row_idx, x in enumerate(X):
        col = 0
        for weights, dilation, padding, bias in kernels:
            conv = dilated_convolution(x, weights, dilation, padding, bias)
            features[row_idx, col] = np.max(conv)
            features[row_idx, col + 1] = np.mean(conv > 0)
            col += 2
    return features


def rocket_scores(X_train, X_test, seq_len):
    if len(X_train) > MAX_TRAIN_REFERENCE:
        rng = np.random.default_rng(RNG_SEED + seq_len + len(X_train))
        ref_idx = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
        X_ref = X_train[ref_idx]
    else:
        X_ref = X_train

    kernels = make_kernels(seq_len)
    train_features = rocket_transform(X_ref, kernels)
    test_features = rocket_transform(X_test, kernels)

    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    center = np.median(train_scaled, axis=0)
    spread = np.median(np.abs(train_scaled - center), axis=0) + 1e-6
    deviations = np.abs(test_scaled - center) / spread
    return np.mean(np.sort(deviations, axis=1)[:, -32:], axis=1)


def evaluate_scores(y_true, scores, strategy):
    preds = predict_by_strategy(scores, strategy)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    oracle = 0.0
    for k in range(1, len(scores) + 1):
        oracle = max(oracle, f1_score(y_true, top_k_predictions(scores, k), zero_division=0))
    return {
        "predicted_count": int(preds.sum()),
        "auc_roc": roc_auc_score(y_true, scores),
        "auc_pr": auc(recall, precision),
        "f1": f1_score(y_true, preds, zero_division=0),
        "oracle_f1": oracle,
    }


def run_dataset(dataset_name):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    if len(X_train) == 0 or len(X_test) == 0:
        return []
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)
    logger.info("Dataset: %s | Length: %s | Train size: %s", dataset_name, seq_len, len(X_train))
    scores = rocket_scores(X_train, X_test, seq_len)
    rows = []
    for strategy in STRATEGIES:
        metrics = evaluate_scores(y_test, scores, strategy)
        rows.append(
            {
                "dataset_name": dataset_name,
                "config_name": "rocket_random_kernels",
                "strategy": strategy,
                "sequence_length": seq_len,
                "test_size": len(y_test),
                "anomaly_count": int(np.sum(y_test)),
                **metrics,
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path, rows, fieldnames):
    exists = False
    try:
        with open(path):
            exists = True
    except FileNotFoundError:
        pass
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for strategy in STRATEGIES:
        subset = [row for row in rows if row["strategy"] == strategy]
        if not subset:
            continue
        f1s = sorted(float(row["f1"]) for row in subset)
        summary.append(
            {
                "strategy": strategy,
                "num_datasets": len(subset),
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


def load_dataset_names():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT name FROM datasets
        WHERE name NOT IN ('CornellWhaleChallenge', 'Wafer_normal_1')
        ORDER BY name
        """
    )
    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names


def main():
    detail_rows = []
    fieldnames = None
    dataset_names = load_dataset_names()
    logger.info("Starting Experiment 26 ROCKET on %d datasets.", len(dataset_names))
    for idx, name in enumerate(dataset_names, 1):
        try:
            rows = run_dataset(name)
            detail_rows.extend(rows)
            if rows:
                fieldnames = fieldnames or list(rows[0].keys())
                append_rows(DETAIL_OUT_PATH, rows, fieldnames)
        except Exception as exc:
            logger.error("Error evaluating dataset %s: %s", name, exc, exc_info=True)
        if idx % 25 == 0 or idx == len(dataset_names):
            summary_rows = summarize(detail_rows)
            write_csv(SUMMARY_OUT_PATH, summary_rows)
            best = summary_rows[0] if summary_rows else None
            if best:
                logger.info(
                    "Progress: [%4d/%4d] rows=%d | best=%s meanF1=%.4f medianF1=%.4f zero=%d",
                    idx,
                    len(dataset_names),
                    len(detail_rows),
                    best["strategy"],
                    best["mean_f1"],
                    best["median_f1"],
                    best["zero_f1_count"],
                )
    logger.info("Experiment 26 ROCKET finished.")


if __name__ == "__main__":
    main()
