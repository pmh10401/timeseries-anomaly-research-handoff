import csv
import logging
import sqlite3
from pathlib import Path

import numpy as np

from run_rank_ensemble_calibration import (
    DB_PATH,
    WEIGHT_CONFIGS,
    device,
    load_dataset_data,
    train_contrastive_scores,
    train_mse_scores,
    weighted_reference_rank_ensemble,
    z_normalize,
)
from run_rank_threshold_calibration import STRATEGIES, evaluate_strategy, write_csv


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DETAIL_OUT_PATH = DATA_DIR / "rank_ensemble_threshold_calibration.csv"
SUMMARY_OUT_PATH = DATA_DIR / "rank_ensemble_threshold_calibration_summary.csv"
LOG_PATH = DATA_DIR / "rank_ensemble_threshold_calibration.log"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("RankEnsembleThresholdCalibration")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


def append_rows(path, rows, fieldnames):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize_by_config_and_strategy(rows):
    summary = []
    keys = sorted({(row["config_name"], row["strategy"]) for row in rows})
    for config_name, strategy in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["strategy"] == strategy]
        if not subset:
            continue
        f1_values = sorted(float(row["f1"]) for row in subset)
        summary.append(
            {
                "config_name": config_name,
                "strategy": strategy,
                "num_datasets": len(subset),
                "mean_auc_roc": np.mean([float(row["auc_roc"]) for row in subset]),
                "mean_auc_pr": np.mean([float(row["auc_pr"]) for row in subset]),
                "mean_f1": np.mean(f1_values),
                "median_f1": float(np.median(f1_values)),
                "zero_f1_count": sum(value == 0.0 for value in f1_values),
                "ge_0_5_count": sum(value >= 0.5 for value in f1_values),
                "mean_predicted_count": np.mean([int(row["predicted_count"]) for row in subset]),
                "mean_oracle_f1": np.mean([float(row["oracle_f1"]) for row in subset]),
            }
        )
    return sorted(summary, key=lambda row: (row["mean_f1"], row["mean_auc_pr"]), reverse=True)


def run_dataset(dataset_name, epochs=10):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    if len(X_train) == 0 or len(X_test) == 0:
        return []
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train)
    X_test = z_normalize(X_test)
    logger.info("Dataset: %s | Length: %s | Train size: %s", dataset_name, seq_len, len(X_train))

    multi_train, multi_test = train_contrastive_scores(X_train, X_test, mode="multi_aug", epochs=epochs)
    fused_train, fused_test = train_contrastive_scores(X_train, X_test, mode="fused", epochs=epochs)
    hybrid_train, hybrid_test = train_mse_scores(X_train, X_test, epochs=epochs)
    train_score_map = {"multi_aug": multi_train, "fused": fused_train, "hybrid": hybrid_train}
    test_score_map = {"multi_aug": multi_test, "fused": fused_test, "hybrid": hybrid_test}

    rows = []
    for config_name, weights in WEIGHT_CONFIGS.items():
        _, test_scores = weighted_reference_rank_ensemble(train_score_map, test_score_map, weights)
        for strategy in STRATEGIES:
            metrics = evaluate_strategy(y_test, test_scores, strategy)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "config_name": config_name,
                    "strategy": strategy,
                    "sequence_length": seq_len,
                    "test_size": len(y_test),
                    "anomaly_count": int(np.sum(y_test)),
                    **metrics,
                }
            )
    return rows


def main():
    if DETAIL_OUT_PATH.exists():
        DETAIL_OUT_PATH.unlink()
    if SUMMARY_OUT_PATH.exists():
        SUMMARY_OUT_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT name FROM datasets
        WHERE name NOT IN ('CornellWhaleChallenge', 'Wafer_normal_1')
        ORDER BY name
        """
    )
    dataset_names = [row[0] for row in cursor.fetchall()]
    conn.close()

    logger.info("Using acceleration device: %s", device)
    logger.info("Starting rank threshold calibration on %d datasets.", len(dataset_names))
    detail_rows = []
    fieldnames = None
    for idx, name in enumerate(dataset_names, 1):
        try:
            rows = run_dataset(name, epochs=10)
            detail_rows.extend(rows)
            if rows:
                fieldnames = fieldnames or list(rows[0].keys())
                append_rows(DETAIL_OUT_PATH, rows, fieldnames)
        except Exception as exc:
            logger.error("Error evaluating dataset %s: %s", name, exc, exc_info=True)
        if idx % 25 == 0 or idx == len(dataset_names):
            summary_rows = summarize_by_config_and_strategy(detail_rows)
            write_csv(SUMMARY_OUT_PATH, summary_rows)
            best = max(summary_rows, key=lambda r: r["mean_f1"]) if summary_rows else None
            if best:
                logger.info(
                    "Progress: [%4d/%4d] rows=%d | best=%s/%s meanF1=%.4f medianF1=%.4f zero=%d",
                    idx,
                    len(dataset_names),
                    len(detail_rows),
                    best["strategy"],
                    "all_configs",
                    best["mean_f1"],
                    best["median_f1"],
                    best["zero_f1_count"],
                )

    summary_rows = summarize_by_config_and_strategy(detail_rows)
    write_csv(SUMMARY_OUT_PATH, summary_rows)
    for row in sorted(summary_rows, key=lambda r: r["mean_f1"], reverse=True)[:10]:
        logger.info(
            "%s | meanF1 %.4f | medianF1 %.4f | zero %d | meanPred %.2f | AUC %.4f | PR %.4f",
            row["strategy"],
            row["mean_f1"],
            row["median_f1"],
            row["zero_f1_count"],
            row["mean_predicted_count"],
            row["mean_auc_roc"],
            row["mean_auc_pr"],
        )
    logger.info("Rank threshold calibration finished.")


if __name__ == "__main__":
    main()
