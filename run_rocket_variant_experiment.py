import csv
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from run_experiment_26_rocket import make_kernels, rocket_transform, summarize, write_csv
from run_rank_ensemble_calibration import load_dataset_data, z_normalize
from run_rank_threshold_calibration import STRATEGIES, predict_by_strategy, top_k_oracle_f1


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


DATA_DIR = "/Users/minho/Documents/Dataset"
EXPERIMENT_ID = os.environ["ROCKET_EXPERIMENT_ID"]
CONFIG_NAME = os.environ["ROCKET_CONFIG_NAME"]
DETAIL_OUT_PATH = f"{DATA_DIR}/{EXPERIMENT_ID}_results.csv"
SUMMARY_OUT_PATH = f"{DATA_DIR}/{EXPERIMENT_ID}_summary.csv"
LOG_PATH = f"{DATA_DIR}/{EXPERIMENT_ID}.log"
WORKERS = env_int("ROCKET_WORKERS", 4)
NUM_KERNELS = env_int("ROCKET_NUM_KERNELS", 256)
SCORE_MODE = os.environ.get("ROCKET_SCORE_MODE", "robust_topk")
TOP_DEVIATIONS = env_int("ROCKET_TOP_DEVIATIONS", 32)
KNN_NEIGHBORS = env_int("ROCKET_KNN_NEIGHBORS", 5)
MAX_TRAIN_REFERENCE = env_int("ROCKET_MAX_TRAIN_REFERENCE", 1000)


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger(EXPERIMENT_ID)
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
    rng = np.random.default_rng(20260706 + seq_len + len(X_train) + NUM_KERNELS)
    idx = rng.choice(len(X_train), size=MAX_TRAIN_REFERENCE, replace=False)
    return X_train[idx]


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


def knn_scores(train_features, test_features):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    n_neighbors = max(1, min(KNN_NEIGHBORS, len(train_scaled)))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(train_scaled)
    distances, _ = nn.kneighbors(test_scaled)
    return distances.mean(axis=1)


def iforest_scores(train_features, test_features):
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=20260706, n_jobs=1)
    model.fit(train_scaled)
    return -model.score_samples(test_scaled)


def rocket_variant_scores(X_train, X_test, seq_len):
    X_ref = reference_train(X_train, seq_len)
    kernels = make_kernels(seq_len, num_kernels=NUM_KERNELS)
    train_features = rocket_transform(X_ref, kernels)
    test_features = rocket_transform(X_test, kernels)
    if SCORE_MODE == "robust_topk":
        return robust_deviation_scores(train_features, test_features, TOP_DEVIATIONS)
    if SCORE_MODE == "knn":
        return knn_scores(train_features, test_features)
    if SCORE_MODE == "iforest":
        return iforest_scores(train_features, test_features)
    raise ValueError(f"Unknown ROCKET_SCORE_MODE: {SCORE_MODE}")


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
    scores = rocket_variant_scores(X_train, X_test, seq_len)
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
    from run_experiment_26_rocket import load_dataset_names as load_names

    return load_names()


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
    dataset_names = load_dataset_names()
    logger.info(
        "Starting %s on %d datasets | workers=%d kernels=%d mode=%s top=%d knn=%d",
        EXPERIMENT_ID,
        len(dataset_names),
        WORKERS,
        NUM_KERNELS,
        SCORE_MODE,
        TOP_DEVIATIONS,
        KNN_NEIGHBORS,
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
                        "Progress: [%4d/%4d] rows=%d | best=%s meanF1=%.4f medianF1=%.4f zero=%d",
                        completed,
                        len(dataset_names),
                        len(detail_rows),
                        best["strategy"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["zero_f1_count"],
                    )
    logger.info("%s finished.", EXPERIMENT_ID)


if __name__ == "__main__":
    main()
