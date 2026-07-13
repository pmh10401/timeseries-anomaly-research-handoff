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

from run_experiment_26_rocket import make_kernels, rocket_transform
from run_rank_ensemble_calibration import load_dataset_data, z_normalize
from run_rank_threshold_calibration import STRATEGIES, predict_by_strategy, top_k_oracle_f1


DATA_DIR = "/Users/minho/Documents/Dataset"
DETAIL_OUT_PATH = f"{DATA_DIR}/experiment_27_rocket_score_variants_results.csv"
SUMMARY_OUT_PATH = f"{DATA_DIR}/experiment_27_rocket_score_variants_summary.csv"
LOG_PATH = f"{DATA_DIR}/experiment_27_rocket_score_variants.log"
WORKERS = int(os.environ.get("ROCKET_WORKERS", "4") or "4")
NUM_KERNELS = 256
MAX_TRAIN_REFERENCE = int(os.environ.get("ROCKET_MAX_TRAIN_REFERENCE", "1000") or "1000")

CONFIGS = [
    ("rocket_256_robust_top16", "robust_topk", {"top_k": 16}),
    ("rocket_256_robust_top64", "robust_topk", {"top_k": 64}),
    ("rocket_256_knn5", "knn", {"neighbors": 5}),
    ("rocket_256_iforest", "iforest", {}),
]


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment27RocketScoreVariants")
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


def scaled_features(train_features, test_features):
    scaler = RobustScaler(quantile_range=(10, 90))
    return scaler.fit_transform(train_features), scaler.transform(test_features)


def robust_deviation_scores(train_scaled, test_scaled, top_k):
    center = np.median(train_scaled, axis=0)
    spread = np.median(np.abs(train_scaled - center), axis=0) + 1e-6
    deviations = np.abs(test_scaled - center) / spread
    top_k = max(1, min(top_k, deviations.shape[1]))
    split_at = deviations.shape[1] - top_k
    return np.mean(np.partition(deviations, split_at, axis=1)[:, -top_k:], axis=1)


def knn_scores(train_scaled, test_scaled, neighbors):
    n_neighbors = max(1, min(neighbors, len(train_scaled)))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(train_scaled)
    distances, _ = nn.kneighbors(test_scaled)
    return distances.mean(axis=1)


def iforest_scores(train_scaled, test_scaled):
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=20260706, n_jobs=1)
    model.fit(train_scaled)
    return -model.score_samples(test_scaled)


def build_scores(X_train, X_test, seq_len):
    X_ref = reference_train(X_train, seq_len)
    kernels = make_kernels(seq_len, num_kernels=NUM_KERNELS)
    train_features = rocket_transform(X_ref, kernels)
    test_features = rocket_transform(X_test, kernels)
    train_scaled, test_scaled = scaled_features(train_features, test_features)
    scores = {}
    for config_name, mode, params in CONFIGS:
        if mode == "robust_topk":
            scores[config_name] = robust_deviation_scores(train_scaled, test_scaled, params["top_k"])
        elif mode == "knn":
            scores[config_name] = knn_scores(train_scaled, test_scaled, params["neighbors"])
        elif mode == "iforest":
            scores[config_name] = iforest_scores(train_scaled, test_scaled)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return scores


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
    score_map = build_scores(X_train, X_test, seq_len)
    rows = []
    for config_name, scores in score_map.items():
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
        "Starting Experiment 27 ROCKET score variants on %d datasets with %d workers.",
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
    logger.info("Experiment 27 ROCKET score variants finished.")


if __name__ == "__main__":
    main()
