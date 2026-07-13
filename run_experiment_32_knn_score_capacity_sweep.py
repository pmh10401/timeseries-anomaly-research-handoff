import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

from run_experiment_26_rocket import load_dataset_names
from run_experiment_29_train_normal_threshold_calibration import (
    knn_score_pair,
    rocket_feature_pair,
    train_false_positive_stats,
)
from run_rank_ensemble_calibration import load_dataset_data, z_normalize
from run_rank_threshold_calibration import top_k_oracle_f1


DATA_DIR = "/Users/minho/Documents/Dataset"
DETAIL_OUT_PATH = f"{DATA_DIR}/experiment_32_knn_score_capacity_sweep_results.csv"
SUMMARY_OUT_PATH = f"{DATA_DIR}/experiment_32_knn_score_capacity_sweep_summary.csv"
LOG_PATH = f"{DATA_DIR}/experiment_32_knn_score_capacity_sweep.log"


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


WORKERS = env_int("KNN_SCORE_CAPACITY_WORKERS", 4)

CONFIGS = [
    ("rocket_128_knn3", 128, 3),
    ("rocket_128_knn5", 128, 5),
    ("rocket_256_knn3", 256, 3),
    ("rocket_256_knn5", 256, 5),
    ("rocket_256_knn10", 256, 10),
    ("rocket_512_knn5", 512, 5),
    ("rocket_512_knn10", 512, 10),
]

THRESHOLD_METHODS = [
    ("count_cap_1pct", 0.01),
    ("count_cap_2pct", 0.02),
    ("count_cap_3pct", 0.03),
]


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment32KnnScoreCapacitySweep")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


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


def run_dataset(dataset_name):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    if len(X_train) == 0 or len(X_test) == 0:
        return []
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    rows = []
    feature_cache = {}
    for config_name, num_kernels, neighbors in CONFIGS:
        if num_kernels not in feature_cache:
            feature_cache[num_kernels] = rocket_feature_pair(X_train, X_test, seq_len, num_kernels)
        train_features, test_features = feature_cache[num_kernels]
        train_scores, test_scores = knn_score_pair(train_features, test_features, neighbors)
        metrics = score_metrics(y_test, test_scores)
        for threshold_method, rate in THRESHOLD_METHODS:
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "config_name": config_name,
                    "num_kernels": num_kernels,
                    "knn_neighbors": neighbors,
                    "threshold_method": threshold_method,
                    "threshold_family": "count_cap_rate",
                    "sequence_length": seq_len,
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    "train_score_count": len(train_scores),
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
        train_exceed_rates = [float(row["train_exceed_rate"]) for row in subset]
        predicted_counts = [int(row["predicted_count"]) for row in subset]
        anomaly_counts = [int(row["anomaly_count"]) for row in subset]
        summary.append(
            {
                "config_name": config_name,
                "threshold_method": method,
                "threshold_family": "count_cap_rate",
                "num_datasets": len(subset),
                "mean_auc_roc": np.mean([float(row["auc_roc"]) for row in subset]),
                "mean_auc_pr": np.mean([float(row["auc_pr"]) for row in subset]),
                "mean_f1": np.mean(f1s),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(1 for value in f1s if value == 0),
                "ge_0_5_count": sum(1 for value in f1s if value >= 0.5),
                "mean_predicted_count": np.mean(predicted_counts),
                "mean_anomaly_count": np.mean(anomaly_counts),
                "mean_train_exceed_rate": np.mean(train_exceed_rates),
                "median_train_exceed_rate": float(np.median(train_exceed_rates)),
                "max_train_exceed_rate": np.max(train_exceed_rates),
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
        "Starting Experiment 32 KNN score capacity sweep on %d datasets with %d workers.",
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
                _, rows = future.result()
            except Exception as exc:
                logger.error("Error evaluating dataset %s: %s", name, exc, exc_info=True)
                rows = []
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
                        "Progress: [%4d/%4d] rows=%d | best=%s/%s meanF1=%.4f medianF1=%.4f trainEx=%.4f pred=%.2f",
                        completed,
                        len(dataset_names),
                        len(detail_rows),
                        best["config_name"],
                        best["threshold_method"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["mean_train_exceed_rate"],
                        best["mean_predicted_count"],
                    )
    logger.info("Experiment 32 KNN score capacity sweep finished.")


if __name__ == "__main__":
    main()
