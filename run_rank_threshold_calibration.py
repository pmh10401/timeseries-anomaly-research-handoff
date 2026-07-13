import csv
import math
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
DETAIL_OUT_PATH = DATA_DIR / "rank_threshold_calibration_results.csv"
SUMMARY_OUT_PATH = DATA_DIR / "rank_threshold_calibration_summary.csv"

STRATEGIES = [
    "top_1",
    "prior_q_01",
    "prior_q_02",
    "prior_q_03",
    "prior_q_05",
    "prior_q_02_min_k_2",
    "prior_q_02_min_k_3",
    "sqrt_n_tail",
    "tail_gap",
]


def load_test_metadata():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT d.name, COUNT(i.id), SUM(i.label)
        FROM datasets d
        JOIN instances i ON d.id = i.dataset_id
        WHERE i.split = 'TEST'
        GROUP BY d.name
        """
    )
    meta = {name: {"n_test": int(n), "n_anomaly": int(n_anomaly or 0)} for name, n, n_anomaly in cur.fetchall()}
    conn.close()
    return meta


def top_k_predictions(scores, k):
    scores = np.asarray(scores, dtype=np.float64)
    preds = np.zeros(len(scores), dtype=np.int64)
    if len(scores) == 0:
        return preds
    k = int(max(1, min(len(scores), k)))
    order = np.argsort(scores, kind="mergesort")
    preds[order[-k:]] = 1
    return preds


def top_k_oracle_f1(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return 0.0
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")[::-1]
    true_positives = np.cumsum(y_true[order])
    k_values = np.arange(1, len(scores) + 1)
    f1_values = (2.0 * true_positives) / (k_values + positives)
    return float(np.max(f1_values))


def predict_by_strategy(scores, strategy):
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if strategy == "top_1":
        return top_k_predictions(scores, 1)
    if strategy.startswith("prior_q_"):
        parts = strategy.split("_")
        q = int(parts[2]) / 100
        k = int(math.ceil(n * q))
        if len(parts) == 6 and parts[3] == "min" and parts[4] == "k":
            k = max(k, int(parts[5]))
        return top_k_predictions(scores, k)
    if strategy == "sqrt_n_tail":
        return top_k_predictions(scores, int(math.ceil(math.sqrt(n))))
    if strategy == "tail_gap":
        return predict_by_tail_gap(scores)
    raise ValueError(f"Unknown strategy: {strategy}")


def predict_by_tail_gap(scores, min_k=1, max_fraction=0.10):
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n <= 2:
        return top_k_predictions(scores, 1)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    max_k = max(min_k, 3, int(math.ceil(n * max_fraction)))
    upper_start = max(1, n - max_k - 1)
    gaps = np.diff(sorted_scores)
    candidate_indices = np.arange(upper_start, n - min_k)
    if len(candidate_indices) == 0:
        return top_k_predictions(scores, min_k)
    best_gap_index = candidate_indices[np.argmax(gaps[candidate_indices])]
    k = n - best_gap_index - 1
    return top_k_predictions(scores, max(min_k, k))


def evaluate_strategy(y_true, scores, strategy):
    preds = predict_by_strategy(scores, strategy)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return {
        "strategy": strategy,
        "predicted_count": int(preds.sum()),
        "auc_roc": roc_auc_score(y_true, scores),
        "auc_pr": auc(recall, precision),
        "f1": f1_score(y_true, preds, zero_division=0),
        "oracle_f1": top_k_oracle_f1(y_true, scores),
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for strategy in STRATEGIES:
        subset = [r for r in rows if r["strategy"] == strategy]
        if not subset:
            continue
        f1_values = sorted(float(r["f1"]) for r in subset)
        summary.append(
            {
                "strategy": strategy,
                "num_datasets": len(subset),
                "mean_auc_roc": np.mean([float(r["auc_roc"]) for r in subset]),
                "mean_auc_pr": np.mean([float(r["auc_pr"]) for r in subset]),
                "mean_f1": np.mean(f1_values),
                "median_f1": float(np.median(f1_values)),
                "zero_f1_count": sum(v == 0.0 for v in f1_values),
                "ge_0_5_count": sum(v >= 0.5 for v in f1_values),
                "mean_predicted_count": np.mean([int(r["predicted_count"]) for r in subset]),
                "mean_oracle_f1": np.mean([float(r["oracle_f1"]) for r in subset]),
            }
        )
    return summary


def main():
    print(
        "This script defines rank threshold strategies. Full calibration requires per-sample rank scores "
        "from a rank ensemble scoring run."
    )


if __name__ == "__main__":
    main()
