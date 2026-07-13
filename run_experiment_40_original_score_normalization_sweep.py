import argparse
import csv
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

from run_balanced_improvement_experiment import density_knn_score_pair
from run_experiment_26_rocket import load_dataset_names
from run_experiment_29_train_normal_threshold_calibration import (
    knn_score_pair,
    rocket_feature_pair,
    train_false_positive_stats,
)
from run_rank_ensemble_calibration import load_dataset_data, z_normalize
from run_rank_threshold_calibration import top_k_oracle_f1


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DETAIL_OUT_PATH = DATA_DIR / "experiment_40_original_score_normalization_sweep_results.csv"
SUMMARY_OUT_PATH = DATA_DIR / "experiment_40_original_score_normalization_sweep_summary.csv"
LOG_PATH = DATA_DIR / "experiment_40_original_score_normalization_sweep.log"


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


WORKERS = env_int("ORIGINAL_SCORE_NORMALIZATION_WORKERS", 4)

CONFIGS = [
    {"name": "rocket_256_knn3_raw", "kind": "rocket_knn", "num_kernels": 256, "neighbors": 3},
    {
        "name": "rocket_256_knn3_local_gap",
        "kind": "density_knn",
        "num_kernels": 256,
        "neighbors": 3,
        "mode": "local_gap",
    },
    {
        "name": "rocket_256_knn3_density_ratio",
        "kind": "density_knn",
        "num_kernels": 256,
        "neighbors": 3,
        "mode": "ratio",
    },
    {
        "name": "rocket_512_knn5_local_gap",
        "kind": "density_knn",
        "num_kernels": 512,
        "neighbors": 5,
        "mode": "local_gap",
    },
]

THRESHOLD_METHODS = [
    ("count_cap_1pct", 0.01),
    ("count_cap_2pct", 0.02),
    ("count_cap_3pct", 0.03),
]


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment40OriginalScoreNormalizationSweep")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


def parse_family(dataset_name):
    if dataset_name == "CornellWhaleChallenge":
        return "CornellWhaleChallenge"
    if "_normal_" in dataset_name:
        return dataset_name.rsplit("_normal_", 1)[0]
    return dataset_name


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
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return {
        "predicted_count": int(preds.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "f1": f1_score(y_true, preds, zero_division=0),
        "oracle_f1": metrics["oracle_f1"],
    }


def score_pair_for_config(X_train, X_test, seq_len, config, feature_cache):
    num_kernels = config["num_kernels"]
    if num_kernels not in feature_cache:
        feature_cache[num_kernels] = rocket_feature_pair(X_train, X_test, seq_len, num_kernels)
    train_features, test_features = feature_cache[num_kernels]
    if config["kind"] == "rocket_knn":
        return knn_score_pair(train_features, test_features, config["neighbors"])
    if config["kind"] == "density_knn":
        return density_knn_score_pair(train_features, test_features, config["neighbors"], config["mode"])
    raise ValueError(f"Unknown config kind: {config['kind']}")


def run_dataset(dataset_name):
    X_train, X_test, y_test = load_dataset_data(dataset_name)
    if len(X_train) == 0 or len(X_test) == 0 or len(np.unique(y_test)) < 2:
        return []
    seq_len = X_train.shape[1]
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)
    rows = []
    feature_cache = {}
    for config in CONFIGS:
        train_scores, test_scores = score_pair_for_config(X_train, X_test, seq_len, config, feature_cache)
        metrics = score_metrics(y_test, test_scores)
        for threshold_method, rate in THRESHOLD_METHODS:
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "family": parse_family(dataset_name),
                    "config_name": config["name"],
                    "score_family": config["kind"],
                    "num_kernels": config["num_kernels"],
                    "knn_neighbors": config["neighbors"],
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


def completed_dataset_names(rows):
    return {row["dataset_name"] for row in rows if row.get("dataset_name")}


def assert_detail_dataset_coverage(path, expected_dataset_names):
    disk_rows, _ = read_existing_detail_rows(path)
    disk_dataset_names = completed_dataset_names(disk_rows)
    missing = [name for name in expected_dataset_names if name not in disk_dataset_names]
    missing_path = path.with_name(f"{path.stem}_missing_datasets.csv")
    if missing:
        with missing_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment_id", "dataset_name"])
            writer.writeheader()
            writer.writerows(
                {"experiment_id": "experiment_40_original_score_normalization_sweep", "dataset_name": name}
                for name in missing
            )
        logger.warning(
            "Experiment 40 detail CSV is missing %d/%d datasets after run; wrote %s. Queue will continue.",
            len(missing),
            len(expected_dataset_names),
            missing_path,
        )
    elif missing_path.exists():
        missing_path.unlink()
    return disk_rows


def repair_missing_datasets(expected_dataset_names, fieldnames, max_attempts=1):
    detail_rows, existing_fieldnames = read_existing_detail_rows(DETAIL_OUT_PATH)
    fieldnames = fieldnames or existing_fieldnames
    for attempt in range(1, int(max_attempts) + 1):
        completed = completed_dataset_names(detail_rows)
        missing = [name for name in expected_dataset_names if name not in completed]
        if not missing:
            break
        logger.warning(
            "Experiment 40 repairing %d missing datasets before queue continues. attempt=%d",
            len(missing),
            attempt,
        )
        for dataset_name in missing:
            try:
                _, rows = run_one(dataset_name)
            except Exception as exc:
                logger.error("Repair failed for dataset %s: %s", dataset_name, exc, exc_info=True)
                rows = []
            if not rows:
                logger.warning("Repair produced no rows for dataset %s", dataset_name)
                continue
            detail_rows.extend(rows)
            fieldnames = fieldnames or list(rows[0].keys())
            append_rows(DETAIL_OUT_PATH, rows, fieldnames)
    return read_existing_detail_rows(DETAIL_OUT_PATH)[0]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    keys = sorted({(row["config_name"], row["threshold_method"]) for row in rows})
    for config_name, method in keys:
        subset = [row for row in rows if row["config_name"] == config_name and row["threshold_method"] == method]
        f1s = [float(row["f1"]) for row in subset]
        by_family = defaultdict(list)
        for row in subset:
            by_family[row["family"]].append(float(row["f1"]))
        family_means = [float(np.mean(values)) for values in by_family.values()]
        summary.append(
            {
                "config_name": config_name,
                "threshold_method": method,
                "threshold_family": "count_cap_rate",
                "num_datasets": len(subset),
                "num_families": len(by_family),
                "mean_auc_roc": float(np.mean([float(row["auc_roc"]) for row in subset])),
                "mean_auc_pr": float(np.mean([float(row["auc_pr"]) for row in subset])),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
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


def run_one(dataset_name):
    return dataset_name, run_dataset(dataset_name)


def run_experiment(args):
    for path in [DETAIL_OUT_PATH, SUMMARY_OUT_PATH]:
        if path.exists() and not args.keep_existing:
            path.unlink()
    detail_rows = []
    fieldnames = None
    if args.keep_existing:
        detail_rows, fieldnames = read_existing_detail_rows(DETAIL_OUT_PATH)

    dataset_names = load_dataset_names()
    if args.dataset_limit is not None:
        dataset_names = dataset_names[: args.dataset_limit]
    expected_dataset_names = list(dataset_names)
    if args.keep_existing and detail_rows:
        completed_names = completed_dataset_names(detail_rows)
        dataset_names = [name for name in dataset_names if name not in completed_names]
    logger.info(
        "Starting Experiment 40 original score normalization sweep on %d datasets with %d workers. existing_rows=%d",
        len(dataset_names),
        args.workers,
        len(detail_rows),
    )
    if not dataset_names and detail_rows:
        write_csv(SUMMARY_OUT_PATH, summarize(detail_rows))
        logger.info("Experiment 40 resume found no remaining datasets.")
        return
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, name): name for name in dataset_names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, rows = future.result()
            except Exception as exc:
                logger.error("Error evaluating dataset %s: %s", name, exc, exc_info=True)
                rows = []
            completed += 1
            if not rows:
                logger.warning("No rows produced for dataset %s", name)
            if rows:
                detail_rows.extend(rows)
                fieldnames = fieldnames or list(rows[0].keys())
                append_rows(DETAIL_OUT_PATH, rows, fieldnames)
            if completed % 25 == 0 or completed == len(dataset_names):
                summary_rows = summarize(detail_rows) if detail_rows else []
                write_csv(SUMMARY_OUT_PATH, summary_rows)
                best = summary_rows[0] if summary_rows else None
                if best:
                    logger.info(
                        "Progress: [%4d/%4d] rows=%d | best=%s/%s meanF1=%.4f medianF1=%.4f aucPR=%.4f fp=%.2f",
                        completed,
                        len(dataset_names),
                        len(detail_rows),
                        best["config_name"],
                        best["threshold_method"],
                        best["mean_f1"],
                        best["median_f1"],
                        best["mean_auc_pr"],
                        best["mean_fp"],
                    )
    final_rows = repair_missing_datasets(expected_dataset_names, fieldnames)
    final_rows = assert_detail_dataset_coverage(DETAIL_OUT_PATH, expected_dataset_names)
    write_csv(SUMMARY_OUT_PATH, summarize(final_rows))
    logger.info("Experiment 40 original score normalization sweep finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 40 original score normalization sweep")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
