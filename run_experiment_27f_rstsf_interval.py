import csv
import logging
import os
import sqlite3

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import RobustScaler

from run_rank_ensemble_calibration import DB_PATH, load_dataset_data, z_normalize
from run_rank_threshold_calibration import STRATEGIES, predict_by_strategy, top_k_oracle_f1
from run_experiment_26_rocket import summarize, write_csv


DATA_DIR = "/Users/minho/Documents/Dataset"
DETAIL_OUT_PATH = f"{DATA_DIR}/experiment_27f_rstsf_interval_results.csv"
SUMMARY_OUT_PATH = f"{DATA_DIR}/experiment_27f_rstsf_interval_summary.csv"
LOG_PATH = f"{DATA_DIR}/experiment_27f_rstsf_interval.log"


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


WORKERS = env_int("RSTSF_WORKERS", 4)
NUM_INTERVALS = env_int("RSTSF_INTERVALS", 256)
MAX_TRAIN_REFERENCE = env_int("RSTSF_MAX_TRAIN_REFERENCE", 1000)
CONFIG_NAME = "rstsf_random_interval_robust"


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment27fRSTSFInterval")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


def make_intervals(series_length, dataset_size, rep_index, num_intervals, seed=20260706):
    rng = np.random.default_rng(seed + series_length + dataset_size + rep_index * 1009)
    intervals = []
    min_len = max(4, series_length // 40)
    max_len = max(min_len + 1, series_length)
    for _ in range(num_intervals):
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, max(1, series_length - length + 1)))
        end = min(series_length, start + length)
        intervals.append((start, end))
    return intervals


def slope_feature(segment):
    n = segment.shape[1]
    if n <= 1:
        return np.zeros(segment.shape[0], dtype=np.float32)
    x = np.arange(n, dtype=np.float32)
    x = x - x.mean()
    denom = float(np.sum(x * x)) + 1e-6
    centered = segment - segment.mean(axis=1, keepdims=True)
    return centered @ x / denom


def representation_sets(X):
    reps = [X.astype(np.float32)]
    reps.append(np.diff(X, n=1, axis=1).astype(np.float32) if X.shape[1] > 1 else X.astype(np.float32))
    reps.append(np.diff(X, n=2, axis=1).astype(np.float32) if X.shape[1] > 2 else reps[-1])
    spectrum = np.abs(np.fft.rfft(X, axis=1)).astype(np.float32)
    reps.append(spectrum if spectrum.shape[1] > 0 else X.astype(np.float32))
    return reps


def moment_features(segment):
    mean = segment.mean(axis=1)
    std = segment.std(axis=1)
    centered = segment - mean[:, None]
    scaled = centered / (std[:, None] + 1e-6)
    skew = np.mean(scaled**3, axis=1)
    kurtosis = np.mean(scaled**4, axis=1)
    return mean, std, skew, kurtosis


def interval_features(X, intervals):
    features = np.empty((len(X), len(intervals) * 9), dtype=np.float32)
    for idx, (start, end) in enumerate(intervals):
        seg = X[:, start:end]
        base = idx * 9
        mean, std, skew, kurtosis = moment_features(seg)
        q25 = np.percentile(seg, 25, axis=1)
        q75 = np.percentile(seg, 75, axis=1)
        features[:, base] = mean
        features[:, base + 1] = std
        features[:, base + 2] = seg.min(axis=1)
        features[:, base + 3] = seg.max(axis=1)
        features[:, base + 4] = np.median(seg, axis=1)
        features[:, base + 5] = q75 - q25
        features[:, base + 6] = slope_feature(seg)
        features[:, base + 7] = skew
        features[:, base + 8] = kurtosis
    return features


def reference_train(X_train, seq_len):
    if len(X_train) <= MAX_TRAIN_REFERENCE:
        return X_train
    rng = np.random.default_rng(20260706 + seq_len + len(X_train))
    idx = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
    return X_train[idx]


def rstsf_scores(X_train, X_test, seq_len):
    X_ref = reference_train(X_train, seq_len)
    train_reps = representation_sets(X_ref)
    test_reps = representation_sets(X_test)
    per_rep_intervals = max(8, NUM_INTERVALS // len(train_reps))
    train_feature_blocks = []
    test_feature_blocks = []
    for rep_index, (train_rep, test_rep) in enumerate(zip(train_reps, test_reps)):
        intervals = make_intervals(train_rep.shape[1], len(X_train), rep_index, per_rep_intervals)
        train_feature_blocks.append(interval_features(train_rep, intervals))
        test_feature_blocks.append(interval_features(test_rep, intervals))
    train_features = np.concatenate(train_feature_blocks, axis=1)
    test_features = np.concatenate(test_feature_blocks, axis=1)
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    center = np.median(train_scaled, axis=0)
    spread = np.median(np.abs(train_scaled - center), axis=0) + 1e-6
    deviations = np.abs(test_scaled - center) / spread
    top_k = min(32, deviations.shape[1])
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
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)
    scores = rstsf_scores(X_train, X_test, seq_len)
    metrics = score_metrics(y_test, scores)
    rows = []
    for strategy in STRATEGIES:
        rows.append(
            {
                "dataset_name": dataset_name,
                "config_name": CONFIG_NAME,
                "strategy": strategy,
                "sequence_length": seq_len,
                "test_size": len(y_test),
                "anomaly_count": int(np.sum(y_test)),
                **evaluate_scores(y_test, scores, strategy, metrics),
            }
        )
    return rows


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


def append_rows(path, rows, fieldnames):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_one(dataset_name):
    return dataset_name, run_dataset(dataset_name)


def main():
    for path in [DETAIL_OUT_PATH, SUMMARY_OUT_PATH]:
        if os.path.exists(path):
            os.remove(path)
    names = load_dataset_names()
    logger.info("Starting r-STSF interval anomaly experiment on %d datasets with %d workers.", len(names), WORKERS)
    detail_rows = []
    fieldnames = None
    completed = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_one, name): name for name in names}
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
            if completed % 25 == 0 or completed == len(names):
                summary_rows = summarize(detail_rows)
                write_csv(SUMMARY_OUT_PATH, summary_rows)
                best = summary_rows[0] if summary_rows else None
                if best:
                    logger.info(
                        "Progress: [%4d/%4d] rows=%d | best=%s meanF1=%.4f medianF1=%.4f zero=%d",
                        completed,
                        len(names),
                        len(detail_rows),
                        best["strategy"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["zero_f1_count"],
                    )
    logger.info("r-STSF interval anomaly experiment finished.")


if __name__ == "__main__":
    main()
