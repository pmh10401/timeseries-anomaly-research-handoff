import argparse
import csv
import hashlib
import logging
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

from run_experiment_29_train_normal_threshold_calibration import (
    knn_score_pair,
    rocket_feature_pair,
    train_false_positive_stats,
)
from run_rank_ensemble_calibration import align_series_lengths, sanitize_series, z_normalize
from run_rank_threshold_calibration import top_k_oracle_f1


DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
DETAIL_OUT_PATH = DATA_DIR / "experiment_33_evalset_reconstruction_results.csv"
SUMMARY_OUT_PATH = DATA_DIR / "experiment_33_evalset_reconstruction_summary.csv"
MANIFEST_OUT_PATH = DATA_DIR / "experiment_33_evalset_reconstruction_manifest.csv"
LOG_PATH = DATA_DIR / "experiment_33_evalset_reconstruction.log"

RNG_SEED = 20260707
WORKERS = int(os.environ.get("EVALSET_RECON_WORKERS", "4"))

CONFIGS = [
    ("rocket_256_knn3", 256, 3),
]

THRESHOLD_METHODS = [
    "count_cap_2pct",
    "count_cap_3pct",
    "adaptive_v0",
]

TRAIN_VARIANTS = [
    "clean_test_only",
    "clean_train_test",
]


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logger = logging.getLogger("Experiment33EvalsetReconstruction")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False
logger.addHandler(logging.FileHandler(LOG_PATH))
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(LOG_FORMAT))


def blob_digest(blob):
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


def stable_key(seed, dataset_name, role, digest, instance_id):
    payload = f"{seed}|{dataset_name}|{role}|{digest}|{instance_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_family(dataset_name):
    if dataset_name == "CornellWhaleChallenge":
        return "CornellWhaleChallenge"
    if "_normal_" in dataset_name:
        return dataset_name.rsplit("_normal_", 1)[0]
    return dataset_name


def ids_to_text(ids):
    return ";".join(str(value) for value in ids)


def text_to_ids(value):
    if not value:
        return []
    return [int(part) for part in str(value).split(";") if part]


def first_unique_by_hash(rows):
    seen = set()
    selected = []
    duplicate_count = 0
    for row in rows:
        digest = row["hash"]
        if digest in seen:
            duplicate_count += 1
            continue
        seen.add(digest)
        selected.append(row)
    return selected, duplicate_count


def load_dataset_rows(conn, dataset_limit=None):
    query = """
        SELECT
            d.id AS dataset_id,
            d.name AS dataset_name,
            d.series_length AS series_length,
            i.id AS instance_id,
            i.split AS split,
            i.instance_index AS instance_index,
            i.label AS label,
            i.values_blob AS values_blob
        FROM datasets d
        JOIN instances i ON i.dataset_id = d.id
        ORDER BY d.id, CASE WHEN i.split = 'TRAIN' THEN 0 ELSE 1 END, i.instance_index
    """
    datasets = []
    current_id = None
    current_rows = []
    current_meta = None
    for row in conn.execute(query):
        dataset_id, name, series_length, instance_id, split, index, label, blob = row
        if current_id is None:
            current_id = dataset_id
            current_meta = {"dataset_id": dataset_id, "dataset_name": name, "series_length": int(series_length)}
        if dataset_id != current_id:
            datasets.append((current_meta, current_rows))
            if dataset_limit is not None and len(datasets) >= dataset_limit:
                return datasets
            current_id = dataset_id
            current_meta = {"dataset_id": dataset_id, "dataset_name": name, "series_length": int(series_length)}
            current_rows = []
        actual_length = len(blob) // 4
        current_rows.append(
            {
                "instance_id": int(instance_id),
                "split": split,
                "instance_index": int(index),
                "label": str(label),
                "hash": blob_digest(blob),
                "actual_length": int(actual_length),
            }
        )
    if current_id is not None and (dataset_limit is None or len(datasets) < dataset_limit):
        datasets.append((current_meta, current_rows))
    return datasets


def select_stable(rows, count, seed, dataset_name, role):
    ordered = sorted(
        rows,
        key=lambda row: stable_key(seed, dataset_name, role, row["hash"], row["instance_id"]),
    )
    return ordered[:count]


def make_manifest_row(meta, rows, manifest_variant, clean_normals, clean_anomalies, seed, duplicate_count):
    dataset_name = meta["dataset_name"]
    train_rows = [row for row in rows if row["split"] == "TRAIN" and row["label"] == "0"]
    clean_train_rows, train_duplicate_count = first_unique_by_hash(train_rows)

    if manifest_variant == "balanced_2pct":
        anomaly_count = min(len(clean_anomalies), len(clean_normals) // 49)
        normal_count = 49 * anomaly_count
        clean_normals = select_stable(clean_normals, normal_count, seed, dataset_name, "balanced_normal")
        clean_anomalies = select_stable(clean_anomalies, anomaly_count, seed, dataset_name, "balanced_anomaly")

    clean_normals = sorted(clean_normals, key=lambda row: row["instance_index"])
    clean_anomalies = sorted(clean_anomalies, key=lambda row: row["instance_index"])
    clean_test_rows = clean_normals + clean_anomalies
    test_total = len(clean_test_rows)
    anomaly_rate = len(clean_anomalies) / test_total if test_total else 0.0
    actual_lengths = [row["actual_length"] for row in rows]
    clean_test_lengths = [row["actual_length"] for row in clean_test_rows]

    return {
        "dataset_id": meta["dataset_id"],
        "dataset_name": dataset_name,
        "family": parse_family(dataset_name),
        "manifest_variant": manifest_variant,
        "series_length": meta["series_length"],
        "actual_len_min": min(actual_lengths) if actual_lengths else 0,
        "actual_len_median": float(np.median(actual_lengths)) if actual_lengths else 0.0,
        "actual_len_max": max(actual_lengths) if actual_lengths else 0,
        "clean_test_actual_len_median": float(np.median(clean_test_lengths)) if clean_test_lengths else 0.0,
        "original_train_count": len(train_rows),
        "clean_train_count": len(clean_train_rows),
        "removed_train_duplicate_normals": train_duplicate_count,
        "original_test_normal_count": sum(1 for row in rows if row["split"] == "TEST" and row["label"] == "0"),
        "original_test_anomaly_count": sum(1 for row in rows if row["split"] == "TEST" and row["label"] == "1"),
        "clean_test_normal_count": len(clean_normals),
        "clean_test_anomaly_count": len(clean_anomalies),
        "clean_test_total_count": test_total,
        "anomaly_rate": anomaly_rate,
        "removed_train_overlap_normals": sum(
            1
            for row in rows
            if row["split"] == "TEST" and row["label"] == "0" and row["hash"] in {train["hash"] for train in train_rows}
        ),
        "removed_test_duplicate_normals": duplicate_count,
        "is_eligible": int(len(clean_normals) > 0 and len(clean_anomalies) > 0),
        "train_instance_ids": [row["instance_id"] for row in train_rows],
        "clean_train_instance_ids": [row["instance_id"] for row in clean_train_rows],
        "clean_test_normal_instance_ids": [row["instance_id"] for row in clean_normals],
        "clean_test_anomaly_instance_ids": [row["instance_id"] for row in clean_anomalies],
        "clean_test_instance_ids": [row["instance_id"] for row in clean_test_rows],
    }


def build_manifests(db_path=DB_PATH, seed=RNG_SEED, dataset_limit=None):
    conn = sqlite3.connect(str(db_path))
    datasets = load_dataset_rows(conn, dataset_limit=dataset_limit)
    conn.close()
    manifests = []
    for meta, rows in datasets:
        train_rows = [row for row in rows if row["split"] == "TRAIN" and row["label"] == "0"]
        train_hashes = {row["hash"] for row in train_rows}
        test_normals = [row for row in rows if row["split"] == "TEST" and row["label"] == "0"]
        test_anomalies = [row for row in rows if row["split"] == "TEST" and row["label"] == "1"]

        non_overlap_normals = [row for row in test_normals if row["hash"] not in train_hashes]
        clean_normals, duplicate_count = first_unique_by_hash(non_overlap_normals)
        clean_anomalies, _ = first_unique_by_hash(test_anomalies)

        strict = make_manifest_row(
            meta,
            rows,
            "strict_unbalanced",
            clean_normals,
            clean_anomalies,
            seed,
            duplicate_count,
        )
        balanced = make_manifest_row(
            meta,
            rows,
            "balanced_2pct",
            clean_normals,
            clean_anomalies,
            seed,
            duplicate_count,
        )
        balanced["is_eligible"] = int(
            balanced["clean_test_normal_count"] > 0 and balanced["clean_test_anomaly_count"] > 0
        )
        manifests.extend([strict, balanced])
    return manifests


def manifest_csv_row(row):
    converted = {}
    for key, value in row.items():
        converted[key] = ids_to_text(value) if isinstance(value, list) else value
    return converted


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_rows(path, rows, fieldnames):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_arrays_by_ids(db_path, instance_ids, target_len):
    if not instance_ids:
        return np.empty((0, target_len), dtype=np.float32)
    placeholders = ",".join("?" for _ in instance_ids)
    query = f"""
        SELECT id, values_blob
        FROM instances
        WHERE id IN ({placeholders})
    """
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(query, instance_ids).fetchall()
    conn.close()
    by_id = {int(row[0]): sanitize_series(np.frombuffer(row[1], dtype=np.float32)) for row in rows}
    series = [by_id[instance_id] for instance_id in instance_ids if instance_id in by_id]
    return align_series_lengths(series, target_len)


def count_cap_threshold(train_scores, rate):
    train_scores = np.asarray(train_scores, dtype=np.float64)
    train_scores = train_scores[np.isfinite(train_scores)]
    if len(train_scores) == 0:
        train_scores = np.array([0.0], dtype=np.float64)
    cap = int(np.floor(float(rate) * len(train_scores)))
    cap = max(0, min(cap, len(train_scores) - 1))
    threshold = float(np.sort(train_scores)[len(train_scores) - cap - 1])
    return threshold, cap / max(1, len(train_scores)), cap


def threshold_rate(method, manifest):
    if method == "count_cap_2pct":
        return 0.02
    if method == "count_cap_3pct":
        return 0.03
    if method == "adaptive_v0":
        median_len = float(manifest["clean_test_actual_len_median"])
        guarded_mid_length = 513 <= median_len <= 1024
        return 0.03 if manifest["clean_test_total_count"] > 50 and not guarded_mid_length else 0.02
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


def run_manifest_task(task):
    db_path, manifest, train_variant = task
    train_ids = (
        manifest["train_instance_ids"]
        if train_variant == "clean_test_only"
        else manifest["clean_train_instance_ids"]
    )
    test_normal_ids = manifest["clean_test_normal_instance_ids"]
    test_anomaly_ids = manifest["clean_test_anomaly_instance_ids"]
    test_ids = test_normal_ids + test_anomaly_ids
    y_test = np.array([0] * len(test_normal_ids) + [1] * len(test_anomaly_ids), dtype=np.int64)
    if len(train_ids) == 0 or len(test_ids) == 0 or len(np.unique(y_test)) < 2:
        return []

    target_len = int(manifest["series_length"]) if int(manifest["series_length"]) > 0 else int(manifest["actual_len_median"])
    X_train = load_arrays_by_ids(db_path, train_ids, target_len)
    X_test = load_arrays_by_ids(db_path, test_ids, target_len)
    if len(X_train) == 0 or len(X_test) == 0:
        return []
    X_train = z_normalize(X_train).astype(np.float32)
    X_test = z_normalize(X_test).astype(np.float32)

    rows = []
    for config_name, num_kernels, neighbors in CONFIGS:
        train_features, test_features = rocket_feature_pair(X_train, X_test, target_len, num_kernels)
        train_scores, test_scores = knn_score_pair(train_features, test_features, neighbors)
        metrics = score_metrics(y_test, test_scores)
        for method in THRESHOLD_METHODS:
            rate = threshold_rate(method, manifest)
            threshold, q_effective, cap_target = count_cap_threshold(train_scores, rate)
            train_exceed_count, train_exceed_rate = train_false_positive_stats(train_scores, threshold)
            rows.append(
                {
                    "dataset_name": manifest["dataset_name"],
                    "family": manifest["family"],
                    "manifest_variant": manifest["manifest_variant"],
                    "train_variant": train_variant,
                    "config_name": config_name,
                    "num_kernels": num_kernels,
                    "knn_neighbors": neighbors,
                    "threshold_method": method,
                    "threshold_family": "adaptive_count_cap" if method == "adaptive_v0" else "count_cap_rate",
                    "sequence_length": target_len,
                    "actual_len_median": manifest["actual_len_median"],
                    "clean_test_actual_len_median": manifest["clean_test_actual_len_median"],
                    "original_train_count": manifest["original_train_count"],
                    "clean_train_count": manifest["clean_train_count"],
                    "train_score_count": len(train_scores),
                    "original_test_normal_count": manifest["original_test_normal_count"],
                    "original_test_anomaly_count": manifest["original_test_anomaly_count"],
                    "clean_test_normal_count": len(test_normal_ids),
                    "clean_test_anomaly_count": len(test_anomaly_ids),
                    "clean_test_total_count": len(test_ids),
                    "removed_train_overlap_normals": manifest["removed_train_overlap_normals"],
                    "removed_test_duplicate_normals": manifest["removed_test_duplicate_normals"],
                    "q_effective": q_effective,
                    "cap_target": cap_target,
                    "threshold": threshold,
                    "train_exceed_count": train_exceed_count,
                    "train_exceed_rate": train_exceed_rate,
                    **evaluate_threshold(y_test, test_scores, threshold, metrics),
                }
            )
    return rows


def summarize(rows):
    summary = []
    keys = sorted(
        {
            (
                row["manifest_variant"],
                row["train_variant"],
                row["config_name"],
                row["threshold_method"],
            )
            for row in rows
        }
    )
    for manifest_variant, train_variant, config_name, method in keys:
        subset = [
            row
            for row in rows
            if row["manifest_variant"] == manifest_variant
            and row["train_variant"] == train_variant
            and row["config_name"] == config_name
            and row["threshold_method"] == method
        ]
        f1s = [float(row["f1"]) for row in subset]
        summary.append(
            {
                "manifest_variant": manifest_variant,
                "train_variant": train_variant,
                "config_name": config_name,
                "threshold_method": method,
                "num_datasets": len(subset),
                "mean_auc_roc": np.mean([float(row["auc_roc"]) for row in subset]),
                "mean_auc_pr": np.mean([float(row["auc_pr"]) for row in subset]),
                "mean_f1": np.mean(f1s),
                "median_f1": float(np.median(f1s)),
                "zero_f1_count": sum(1 for value in f1s if value == 0),
                "mean_predicted_count": np.mean([int(row["predicted_count"]) for row in subset]),
                "mean_tp": np.mean([int(row["tp"]) for row in subset]),
                "mean_fp": np.mean([int(row["fp"]) for row in subset]),
                "mean_fn": np.mean([int(row["fn"]) for row in subset]),
                "mean_train_exceed_rate": np.mean([float(row["train_exceed_rate"]) for row in subset]),
                "mean_oracle_f1": np.mean([float(row["oracle_f1"]) for row in subset]),
                "mean_removed_train_overlap_normals": np.mean(
                    [int(row["removed_train_overlap_normals"]) for row in subset]
                ),
                "mean_removed_test_duplicate_normals": np.mean(
                    [int(row["removed_test_duplicate_normals"]) for row in subset]
                ),
            }
        )
    return sorted(summary, key=lambda row: row["mean_f1"], reverse=True)


def build_tasks(manifests, db_path, manifest_variants=None, train_variants=None):
    manifest_variants = set(manifest_variants or ["strict_unbalanced", "balanced_2pct"])
    train_variants = set(train_variants or TRAIN_VARIANTS)
    tasks = []
    for manifest in manifests:
        if manifest["manifest_variant"] not in manifest_variants:
            continue
        if int(manifest["is_eligible"]) != 1:
            continue
        for train_variant in TRAIN_VARIANTS:
            if train_variant in train_variants:
                tasks.append((str(db_path), manifest, train_variant))
    return tasks


def run_experiment(args):
    for path in [DETAIL_OUT_PATH, SUMMARY_OUT_PATH, MANIFEST_OUT_PATH]:
        if path.exists() and not args.keep_existing:
            path.unlink()

    manifests = build_manifests(args.db_path, seed=args.seed, dataset_limit=args.dataset_limit)
    write_csv(MANIFEST_OUT_PATH, [manifest_csv_row(row) for row in manifests])
    strict_eligible = sum(
        1 for row in manifests if row["manifest_variant"] == "strict_unbalanced" and row["is_eligible"]
    )
    balanced_eligible = sum(
        1 for row in manifests if row["manifest_variant"] == "balanced_2pct" and row["is_eligible"]
    )
    logger.info(
        "Manifest written: %s rows=%d strict_eligible=%d balanced_eligible=%d",
        MANIFEST_OUT_PATH,
        len(manifests),
        strict_eligible,
        balanced_eligible,
    )
    if args.manifest_only:
        return []

    tasks = build_tasks(
        manifests,
        args.db_path,
        manifest_variants=args.manifest_variant,
        train_variants=args.train_variant,
    )
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]
    logger.info("Starting experiment 33 on %d manifest/train tasks with %d workers.", len(tasks), args.workers)

    detail_rows = []
    fieldnames = None
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_manifest_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            manifest = task[1]
            try:
                rows = future.result()
            except Exception as exc:
                logger.error(
                    "Error evaluating %s/%s/%s: %s",
                    manifest["dataset_name"],
                    manifest["manifest_variant"],
                    task[2],
                    exc,
                    exc_info=True,
                )
                rows = []
            completed += 1
            if rows:
                detail_rows.extend(rows)
                fieldnames = fieldnames or list(rows[0].keys())
                append_rows(DETAIL_OUT_PATH, rows, fieldnames)
            if completed % 25 == 0 or completed == len(tasks):
                summary_rows = summarize(detail_rows) if detail_rows else []
                write_csv(SUMMARY_OUT_PATH, summary_rows)
                best = summary_rows[0] if summary_rows else None
                if best:
                    logger.info(
                        "Progress: [%4d/%4d] rows=%d | best=%s/%s/%s/%s meanF1=%.4f medianF1=%.4f",
                        completed,
                        len(tasks),
                        len(detail_rows),
                        best["manifest_variant"],
                        best["train_variant"],
                        best["config_name"],
                        best["threshold_method"],
                        best["mean_f1"],
                        best["median_f1"],
                    )
    logger.info("Experiment 33 finished.")
    return detail_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 33 evalset reconstruction validation")
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument(
        "--manifest-variant",
        action="append",
        choices=["strict_unbalanced", "balanced_2pct"],
        help="Limit evaluation to one or more manifest variants. Repeat flag to include multiple.",
    )
    parser.add_argument(
        "--train-variant",
        action="append",
        choices=TRAIN_VARIANTS,
        help="Limit evaluation to one or more train variants. Repeat flag to include multiple.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
